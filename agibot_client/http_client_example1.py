#!/usr/bin/env python3
import time
import cv2
import json_numpy
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

# 必须打补丁，否则 json 序列化 numpy 会报错
json_numpy.patch()

# ----------------- 配置区 -----------------
SERVER_URL = "http://127.0.0.1:6789/act"
GRIPPER_MAX = 120.0     # 硬件定义：120mm 为闭合
GRIPPER_OPEN_CMD = 0.0  # 硬件定义：0mm 为张开
DT = 0.1               # 50Hz = 0.02s (请确保这与您的训练数据频率一致)
# -----------------------------------------

try:
    from genie_msgs.msg import EndState, HeadState, WaistState
    HAVE_GENIE = True
except ImportError:
    HAVE_GENIE = False

def decode_image_msg(msg: Image) -> np.ndarray:
    """将 ROS Image 消息解码为 numpy 数组 (BGR)"""
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != h * msg.step:
        return np.zeros((h, w, 3), dtype=np.uint8)
    row = data.reshape((h, msg.step))
    # 简单的 BGR8 解码
    return row[:, :w*3].reshape((h, w, 3))[..., ::-1].copy()

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def to_degrees(v):
    # 简单的弧度转角度
    val = float(v) if v is not None else 0.0
    return np.degrees(val) if abs(val) <= 6.5 else val

class SimpleRobotClient(Node):
    def __init__(self):
        super().__init__('simple_robot_client')
       
        # 数据缓存
        self.images = {'head': None, 'hand_left': None, 'hand_right': None}
        self.state = {
            'left_arm': np.zeros(7), 'right_arm': np.zeros(7),
            'left_eff': 0.0, 'right_eff': 0.0,
            'head': np.zeros(2), 'waist': np.zeros(2)
        }
        self.arm_joint_names = []

        # 1. 创建发布者
        self.arm_pub = self.create_publisher(JointState, '/wbc/arm_command', 10)
        self.l_ee_pub = self.create_publisher(JointState, '/wbc/left_ee_command', 10)
        self.r_ee_pub = self.create_publisher(JointState, '/wbc/right_ee_command', 10)

        # 2. 创建订阅者 (Reliable 模式)
        self.create_subscription(Image, '/camera/head_color', lambda m: self.img_cb(m, 'head'), 10)
        self.create_subscription(Image, '/camera/hand_left_color', lambda m: self.img_cb(m, 'hand_left'), 10)
        self.create_subscription(Image, '/camera/hand_right_color', lambda m: self.img_cb(m, 'hand_right'), 10)
        self.create_subscription(JointState, '/hal/arm_joint_state', self.arm_cb, 10)

        if HAVE_GENIE:
            self.create_subscription(EndState, '/hal/left_ee_data', lambda m: self.ee_cb(m, 'left'), 10)
            self.create_subscription(EndState, '/hal/right_ee_data', lambda m: self.ee_cb(m, 'right'), 10)
            self.create_subscription(HeadState, '/hal/neck_state', self.neck_cb, 10)
            self.create_subscription(WaistState, '/hal/waist_state', self.waist_cb, 10)

        # 3. 初始化：等待传感器数据
        print("Waiting for sensors...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if all(v is not None for v in self.images.values()) and len(self.arm_joint_names) > 0:
                break
        print("Sensors Ready!")

        # 4. 初始化：强制张开夹爪 (避免模型误判)
        self.force_open_gripper()

    # --- 极简回调函数 ---
    def img_cb(self, msg, name):
        try:
            img = decode_image_msg(msg)
            self.images[name] = cv2.resize(img, (640, 480))
        except: pass

    def arm_cb(self, msg):
        self.arm_joint_names = msg.name
        p = np.array(msg.position)
        if len(p) >= 14:
            self.state['left_arm'] = p[:7]
            self.state['right_arm'] = p[7:14]

    def ee_cb(self, msg, side):
        try:
            pos = getattr(msg, 'position', [0.0])
            val = float(pos[0]) if isinstance(pos, (list, tuple)) else float(pos)
           
            # [修改 1] 输入归一化：硬件(0~120) -> 模型(0~1)
            self.state[f'{side}_eff'] = clamp01(val / GRIPPER_MAX)
           
        except: pass

    def neck_cb(self, msg):
        try:
            ms = msg.motor_states
            # 智元头部状态归一化逻辑
            self.state['head'] = np.array([
                clamp01((to_degrees(ms[0].position) + 90)/180),
                clamp01((to_degrees(ms[1].position) + 20)/45)
            ])
        except: pass

    def waist_cb(self, msg):
        try:
            ms = msg.motor_states
            # 智元腰部状态归一化逻辑
            self.state['waist'] = np.array([
                clamp01(to_degrees(ms[0].position)/90),
                clamp01(ms[1].position/50)
            ])
        except: pass

    def force_open_gripper(self):
        print(f">>> Forcing Gripper to OPEN ({GRIPPER_OPEN_CMD} mm)...")
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper"]
        msg.position = [GRIPPER_OPEN_CMD] # 发送 0.0
       
        # 连续发几次确保收到
        for _ in range(5):
            self.l_ee_pub.publish(msg)
            self.r_ee_pub.publish(msg)
            time.sleep(0.1)

    # --- 主逻辑 ---
    def run(self):
        print(">>> Starting Control Loop...")
       
        while rclpy.ok():
            # 1. 确保拿到最新数据 (刷新一下 buffer)
            rclpy.spin_once(self, timeout_sec=0.0)

            # 2. 构造 Obs (注意：此时 state['left_eff'] 已经是 0~1 了)
            obs = {
                "video.top_head": self.images['head'][None, ...],
                "video.hand_left": self.images['hand_left'][None, ...],
                "video.hand_right": self.images['hand_right'][None, ...],
                "state.left_arm_joint_position": self.state['left_arm'].reshape(1, 7),
                "state.right_arm_joint_position": self.state['right_arm'].reshape(1, 7),
                "state.left_effector_position": np.array([[self.state['left_eff']]]),
                "state.right_effector_position": np.array([[self.state['right_eff']]]),
                "state.head_position": self.state['head'].reshape(1, 2),
                "state.waist_position": self.state['waist'].reshape(1, 2),
                "annotation.language.action_text": ["pick up the watter bottle"],
            }

            # 3. 推理 (Blocking)
            try:
                resp = requests.post(SERVER_URL, json={"observation": obs})
                if resp.status_code != 200:
                    print(f"Server error: {resp.status_code}")
                    continue
                action_chunk = resp.json()
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(0.5)
                continue

            # 4. 执行 16 帧 (Blocking)
            chunk_len = len(action_chunk['action.left_arm_joint_position'])
           
            for i in range(chunk_len):
                # --- Arm ---
                l_cmd = np.array(action_chunk['action.left_arm_joint_position'][i])
                r_cmd = np.array(action_chunk['action.right_arm_joint_position'][i])
               
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.arm_joint_names
                msg.position = np.concatenate([l_cmd, r_cmd]).tolist()
                self.arm_pub.publish(msg)

                # --- Gripper ---
                # 模型输出 0~1 -> 乘以 120 -> 硬件毫米值
                # [修改 2] 保持原来的乘法，但增加打印
                val_l, val_r = 0.0, 0.0
               
                if 'action.left_effector_position' in action_chunk:
                    val_l = float(action_chunk['action.left_effector_position'][i])
                    self.pub_gripper(self.l_ee_pub, val_l * GRIPPER_MAX)

                if 'action.right_effector_position' in action_chunk:
                    val_r = float(action_chunk['action.right_effector_position'][i])
                    self.pub_gripper(self.r_ee_pub, val_r * GRIPPER_MAX)
               
                # [修改 3] 打印调试信息 (只打印每块的第一帧，避免刷屏)
                if i == 0:
                    print(f"Grip Model: L={val_l:.2f} R={val_r:.2f} -> Cmd: L={val_l*GRIPPER_MAX:.1f}mm")

                # --- Pacing ---
                time.sleep(DT)

    def pub_gripper(self, pub, val):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper"]
        msg.position = [val]
        pub.publish(msg)

def main():
    rclpy.init()
    node = SimpleRobotClient()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
