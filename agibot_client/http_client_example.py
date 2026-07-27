#!/usr/bin/env python3
import time
import cv2
import json_numpy
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

json_numpy.patch()

try:
    from genie_msgs.msg import ArmState, EndState, HeadState, WaistState  # type: ignore
    HAVE_GENIE = True
except Exception:
    HAVE_GENIE = False

def decode_image_msg(msg: Image) -> np.ndarray:
    h = msg.height
    w = msg.width
    step = msg.step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != h * step:
        try:
            channels = data.size // (h * w)
            if channels >= 1:
                return data.reshape((h, w, channels))[:, :, :3].copy()
        except Exception:
            pass
        raise ValueError(f"Unexpected image buffer size {data.size}, expected {h*step}")
    row = data.reshape((h, step))
    channels = step // w
    arr = row[:, : w * channels].reshape((h, w, channels))
    enc = (msg.encoding or "").lower()
    if enc == "bgr8" and channels >= 3:
        arr = arr[:, :, :3][..., ::-1]
    elif enc == "rgb8" and channels >= 3:
        arr = arr[:, :, :3]
    elif enc == "mono8" or channels == 1:
        gray = arr[:, :, 0]
        arr = np.stack([gray, gray, gray], axis=2)
    elif enc == "rgba8" and channels >= 4:
        arr = arr[:, :, :3]
    else:
        if channels >= 3:
            arr = arr[:, :, :3]
        else:
            gray = arr[:, :, 0]
            arr = np.stack([gray, gray, gray], axis=2)
    return arr.copy()

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def normalize_effector_value(v):
    if v is None:
        return 0.0
    try:
        val = float(v)
    except Exception:
        return 0.0
    if abs(val) > 2.0:
        return clamp01(val / 120.0)
    else:
        return clamp01((val + np.pi) / (2 * np.pi))

def to_degrees_if_rad(v):
    if v is None:
        return 0.0
    try:
        val = float(v)
    except Exception:
        return 0.0
    if abs(val) <= 6.5:
        return np.degrees(val)
    return val

class CameraAndStateSubscriber(Node):
    def __init__(self):
        super().__init__('camera_state_subscriber')
        self.images = {'head': None, 'hand_left': None, 'hand_right': None}
        self.create_subscription(Image, '/camera/head_color', lambda msg: self._image_cb(msg, 'head'), 10)
        self.create_subscription(Image, '/camera/hand_left_color', lambda msg: self._image_cb(msg, 'hand_left'), 10)
        self.create_subscription(Image, '/camera/hand_right_color', lambda msg: self._image_cb(msg, 'hand_right'), 10)

        self.left_arm = np.zeros(7, dtype=np.float64)
        self.right_arm = np.zeros(7, dtype=np.float64)
        self.left_eff = 0.0
        self.right_eff = 0.0
        self.head_pos = np.zeros(2, dtype=np.float64)
        self.waist_pos = np.zeros(2, dtype=np.float64)
       
        self.arm_joint_names = []

        # Publishers for control
        self.arm_cmd_pub = self.create_publisher(JointState, '/wbc/arm_command', 10)
        self.left_ee_cmd_pub = self.create_publisher(JointState, '/wbc/left_ee_command', 10)
        self.right_ee_cmd_pub = self.create_publisher(JointState, '/wbc/right_ee_command', 10)

        self.create_subscription(JointState, '/hal/arm_joint_state', self._arm_joint_state_cb, 10)

        if HAVE_GENIE:
            try:
                self.create_subscription(ArmState, '/hal/left_arm_data', self._left_arm_data_cb, 10)
                self.create_subscription(ArmState, '/hal/right_arm_data', self._right_arm_data_cb, 10)
                self.create_subscription(EndState, '/hal/left_ee_data', self._left_ee_cb, 10)
                self.create_subscription(EndState, '/hal/right_ee_data', self._right_ee_cb, 10)
                self.create_subscription(HeadState, '/hal/neck_state', self._neck_cb, 10)
                self.create_subscription(WaistState, '/hal/waist_state', self._waist_cb, 10)
            except Exception as e:
                self.get_logger().warning(f"some genie_msgs subscriptions failed: {e}")

    def _image_cb(self, msg: Image, name: str):
        try:
            img = decode_image_msg(msg)
            img = cv2.resize(img, (848, 480))
            self.images[name] = img
        except Exception as e:
            self.get_logger().warning(f"decode {name} failed: {e}")

    def _arm_joint_state_cb(self, msg: JointState):
        try:
            if msg.name and len(msg.name) >= 14:
                self.arm_joint_names = msg.name

            pos = np.array(msg.position, dtype=np.float64)
            if pos.size >= 14:
                self.left_arm = pos[:7].astype(np.float64)
                self.right_arm = pos[7:14].astype(np.float64)
            elif pos.size >= 7:
                self.left_arm = pos[:7].astype(np.float64)
        except Exception as e:
            self.get_logger().warning(f"arm_joint_state parse failed: {e}")

    def _left_arm_data_cb(self, msg):
        try:
            motor_states = getattr(msg, 'motor_states', None)
            if motor_states:
                positions = [getattr(m, 'position', 0.0) for m in motor_states]
                if len(positions) >= 7:
                    self.left_arm = np.array(positions[:7], dtype=np.float64)
        except Exception as e:
            self.get_logger().warning(f"left_arm_data parse failed: {e}")

    def _right_arm_data_cb(self, msg):
        try:
            motor_states = getattr(msg, 'motor_states', None)
            if motor_states:
                positions = [getattr(m, 'position', 0.0) for m in motor_states]
                if len(positions) >= 7:
                    self.right_arm = np.array(positions[:7], dtype=np.float64)
        except Exception as e:
            self.get_logger().warning(f"right_arm_data parse failed: {e}")

    def _left_ee_cb(self, msg):
        try:
            if hasattr(msg, 'position'):
                pos = getattr(msg, 'position')
                if isinstance(pos, (list, tuple)) and len(pos) >= 1:
                    v = float(pos[0])
                else:
                    v = float(pos)
            else:
                v = 0.0
            self.left_eff = normalize_effector_value(v)
        except Exception as e:
            self.get_logger().warning(f"left_ee parse failed: {e}")

    def _right_ee_cb(self, msg):
        try:
            if hasattr(msg, 'position'):
                pos = getattr(msg, 'position')
                if isinstance(pos, (list, tuple)) and len(pos) >= 1:
                    v = float(pos[0])
                else:
                    v = float(pos)
            else:
                v = 0.0
            self.right_eff = normalize_effector_value(v)
        except Exception as e:
            self.get_logger().warning(f"right_ee parse failed: {e}")

    def _neck_cb(self, msg):
        try:
            ms = getattr(msg, 'motor_states', None)
            if ms and len(ms) >= 2:
                yaw = getattr(ms[0], 'position', 0.0)
                pitch = getattr(ms[1], 'position', 0.0)
            else:
                js = getattr(msg, 'joint_states', None) or getattr(msg, 'position', None)
                if js and len(js) >= 2:
                    yaw = js[0]
                    pitch = js[1]
                else:
                    yaw = 0.0
                    pitch = 0.0
            yaw_deg = to_degrees_if_rad(yaw)
            pitch_deg = to_degrees_if_rad(pitch)
            yaw_norm = clamp01((yaw_deg + 90.0) / 180.0)
            pitch_norm = clamp01((pitch_deg + 20.0) / 45.0)
            self.head_pos = np.array([yaw_norm, pitch_norm], dtype=np.float64)
        except Exception as e:
            self.get_logger().warning(f"neck parse failed: {e}")

    def _waist_cb(self, msg):
        try:
            ms = getattr(msg, 'motor_states', None)
            if ms and len(ms) >= 2:
                body_pitch = getattr(ms[0], 'position', 0.0)
                lift = getattr(ms[1], 'position', 0.0)
            else:
                js = getattr(msg, 'joint_states', None) or getattr(msg, 'position', None)
                if js and len(js) >= 2:
                    body_pitch = js[0]
                    lift = js[1]
                else:
                    body_pitch = 0.0
                    lift = 0.0
            body_pitch_deg = to_degrees_if_rad(body_pitch)
            body_norm = clamp01(body_pitch_deg / 90.0)
            lift_norm = clamp01(lift / 50.0)
            self.waist_pos = np.array([body_norm, lift_norm], dtype=np.float64)
        except Exception as e:
            self.get_logger().warning(f"waist parse failed: {e}")

    def get_images(self):
        return self.images

    def get_states(self):
        return {
            'left_arm': self.left_arm.copy(),
            'right_arm': self.right_arm.copy(),
            'left_eff': float(self.left_eff),
            'right_eff': float(self.right_eff),
            'head_pos': self.head_pos.copy(),
            'waist_pos': self.waist_pos.copy(),
        }
   
    def publish_action(self, action_dict):
        try:
            # --- Arm Control with Smooth Interpolation ---
            if 'action.left_arm_joint_position' in action_dict and 'action.right_arm_joint_position' in action_dict:
                left_action = action_dict['action.left_arm_joint_position']
                right_action = action_dict['action.right_arm_joint_position']
               
                if len(left_action) > 0 and len(right_action) > 0:
                    # 1. Get Model's Ideal Target
                    model_target_pos = np.concatenate([left_action[0], right_action[0]])
                   
                    # 2. Get Current Physical Position
                    current_pos = np.concatenate([self.left_arm, self.right_arm])
                   
                    # 3. Calculate Difference
                    diff = model_target_pos - current_pos
                   
                    # 4. Safety Clip (Velocity Limiter)
                    # Limit max movement per step to 0.05 rad (~2.8 degrees)
                    # This prevents "Safety Limit Exceeded" errors while allowing the robot
                    # to smoothly converge to the model's target pose.
                    STEP_LIMIT = 0.05
                    safe_step = np.clip(diff, -STEP_LIMIT, STEP_LIMIT)
                   
                    # 5. Calculate Final Safe Command
                    final_cmd_pos = current_pos + safe_step

                    # 6. Publish
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    if self.arm_joint_names:
                        msg.name = self.arm_joint_names
                    msg.position = final_cmd_pos.tolist()
                    self.arm_cmd_pub.publish(msg)
                   
                    # Optional debug: Print if we are still "catching up"
                    if np.max(np.abs(diff)) > STEP_LIMIT:
                        print(f"Catching up... dist: {np.max(np.abs(diff)):.3f}")

            # --- End Effector Control ---
            if 'action.left_effector_position' in action_dict:
                raw_val = action_dict['action.left_effector_position'][0]
                left_eff_val = float(raw_val.item() if hasattr(raw_val, 'item') else raw_val)
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = ["gripper"]
                msg.position = [left_eff_val * 120.0]
                self.left_ee_cmd_pub.publish(msg)

            if 'action.right_effector_position' in action_dict:
                raw_val = action_dict['action.right_effector_position'][0]
                right_eff_val = float(raw_val.item() if hasattr(raw_val, 'item') else raw_val)
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = ["gripper"]
                msg.position = [right_eff_val * 120.0]
                self.right_ee_cmd_pub.publish(msg)
               
        except Exception as e:
            self.get_logger().error(f"Failed to publish action: {e}")


def main():
    rclpy.init()
    node = CameraAndStateSubscriber()

    print("Waiting for images & joint states...")
    while any(v is None for v in node.images.values()) or not node.arm_joint_names:
        rclpy.spin_once(node, timeout_sec=0.1)
   
    print(f"Ready! Captured {len(node.arm_joint_names)} joint names.")
    print("Starting Control Loop (Soft Start Enabled)...")

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            imgs = node.get_images()
            states = node.get_states()

            obs = {
                "video.top_head": imgs['head'][np.newaxis, ...].astype(np.uint8),
                "video.hand_left": imgs['hand_left'][np.newaxis, ...].astype(np.uint8),
                "video.hand_right": imgs['hand_right'][np.newaxis, ...].astype(np.uint8),
                "state.left_arm_joint_position": states['left_arm'].reshape(1, 7).astype(np.float64),
                "state.right_arm_joint_position": states['right_arm'].reshape(1, 7).astype(np.float64),
                "state.left_effector_position": np.array([[states['left_eff']]], dtype=np.float64),
                "state.right_effector_position": np.array([[states['right_eff']]], dtype=np.float64),
                "state.head_position": states['head_pos'].reshape(1, 2).astype(np.float64),
                "state.waist_position": states['waist_pos'].reshape(1, 2).astype(np.float64),
                "annotation.language.action_text": ["open the door"],
            }

            try:
                response = requests.post("http://127.0.0.1:6789/act", json={"observation": obs})
               
                if response.status_code == 200:
                    action_dict = response.json()
                    node.publish_action(action_dict)
                else:
                    print(f"Server error: {response.status_code}")

            except Exception as e:
                print(f"Request failed: {e}")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, exiting...")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()

