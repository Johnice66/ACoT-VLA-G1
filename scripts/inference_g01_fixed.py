import numpy as np
import cv2
import time
import os
import sys
import threading
from datetime import datetime
import argparse

try:
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def make_interp_spline(x, y, k=3):
        from scipy.interpolate import interp1d
        return interp1d(x, y, kind='cubic', bounds_error=False, fill_value="extrapolate")

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    Image = None
    JointState = None
    CvBridge = None
    ROS2_AVAILABLE = False

try:
    from genie_msgs.msg import EndState
except ImportError:
    EndState = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'openpi-client', 'src'))
from openpi_client import websocket_client_policy


DEFAULT_TASK_DESCRIPTION = "Fixed-point Non-generalized Door Opening"
G01_EFFECTIVE_DIM = 16


def build_g01_observation(
    latest_images,
    latest_joint_state,
    latest_left_gripper,
    latest_right_gripper,
    task_description=DEFAULT_TASK_DESCRIPTION,
):
    """Build the direct WebSocket input expected by G01ACOTInputs."""

    required_cameras = ("top_head", "hand_left", "hand_right")
    missing_cameras = [name for name in required_cameras if name not in latest_images]
    if missing_cameras:
        raise ValueError(f"缺少相机图像: {missing_cameras}")

    images = {}
    for name in required_cameras:
        image = np.asarray(latest_images[name])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"{name} 图像应为 HWC 三通道，实际 shape={image.shape}"
            )
        images[name] = np.ascontiguousarray(
            cv2.resize(image, (640, 480)),
            dtype=np.uint8,
        )

    joints = np.asarray(latest_joint_state, dtype=np.float32).reshape(-1)

    # /hal/arm_joint_state normally provides 14 arm joints directly.
    # Also support a full G01 state vector, where arm joints are at 28:35 and 35:42.
    if joints.size == 14:
        left_arm = joints[:7]
        right_arm = joints[7:14]
    elif joints.size >= 42:
        left_arm = joints[28:35]
        right_arm = joints[35:42]
    else:
        raise ValueError(
            "G01关节状态应为14维双臂状态，或至少42维的完整状态；"
            f"实际 shape={joints.shape}"
        )

    if latest_left_gripper is None or latest_right_gripper is None:
        raise ValueError("夹爪状态尚未就绪")

    state = np.concatenate(
        [
            left_arm,
            right_arm,
            np.asarray([latest_left_gripper], dtype=np.float32),
            np.asarray([latest_right_gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    if state.shape != (G01_EFFECTIVE_DIM,):
        raise ValueError(f"G01 state应为(16,)，实际为{state.shape}")

    return {
        "images": images,
        "state": np.ascontiguousarray(state),
        "prompt": str(task_description),
    }


def adapt_g01_action(response):
    """Convert G01 policy output into the split structure used by this controller."""

    if not isinstance(response, dict):
        raise TypeError(f"服务端响应应为dict，实际为{type(response)!r}")

    split_keys = {
        "left_arm_joint_position",
        "right_arm_joint_position",
        "left_effector_position",
        "right_effector_position",
    }
    if split_keys.issubset(response):
        return response

    if "actions" not in response:
        raise KeyError(
            f"服务端响应缺少'actions'，实际keys={list(response.keys())}"
        )

    actions = np.asarray(response["actions"], dtype=np.float32)

    # Accept [1, T, D] or [T, D].
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(
                f"只支持batch=1的动作，实际shape={actions.shape}"
            )
        actions = actions[0]

    if actions.ndim != 2:
        raise ValueError(
            f"G01 actions应为[T,D]或[1,T,D]，实际shape={actions.shape}"
        )
    if actions.shape[-1] < G01_EFFECTIVE_DIM:
        raise ValueError(
            f"G01 actions最后一维至少为16，实际shape={actions.shape}"
        )

    actions = np.ascontiguousarray(actions[:, :G01_EFFECTIVE_DIM])

    adapted = {
        "left_arm_joint_position": actions[None, :, 0:7],
        "right_arm_joint_position": actions[None, :, 7:14],
        "left_effector_position": actions[None, :, 14:15],
        "right_effector_position": actions[None, :, 15:16],
    }

    # Preserve optional diagnostics returned by the server.
    for key in ("coarse_actions", "server_timing", "policy_timing"):
        if key in response:
            adapted[key] = response[key]

    return adapted


class MockInferenceNode:
    """模拟推理节点，用于测试模式"""

    def __init__(self):
        self.latest_images = {}
        self.latest_joint_state = np.zeros(14, dtype=np.float32)
        self.latest_left_gripper = 0.0
        self.latest_right_gripper = 0.0

        self.last_chunk_left_arm = None
        self.last_chunk_right_arm = None
        self.last_chunk_left_gripper = None
        self.last_chunk_right_gripper = None

        self.is_holding = False
        self.hold_left_arm = None
        self.hold_right_arm = None
        self.hold_left_gripper = None
        self.hold_right_gripper = None
        self.hold_lock = threading.Lock()
        self._hold_thread_running = False

        self._generate_mock_data()

    def _generate_mock_data(self):
        """生成模拟数据"""
        self.latest_images['top_head'] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        self.latest_images['hand_left'] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        self.latest_images['hand_right'] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        self.latest_joint_state = np.random.randn(14).astype(np.float32) * 0.5
        self.latest_left_gripper = np.random.uniform(0, 100)
        self.latest_right_gripper = np.random.uniform(0, 100)

    def is_data_ready(self):
        return True

    def get_data_status(self):
        return "模拟数据: ✓"

    def get_missing_data(self):
        return []

    def get_observation(self, task_description=DEFAULT_TASK_DESCRIPTION):
        return build_g01_observation(
            latest_images=self.latest_images,
            latest_joint_state=self.latest_joint_state,
            latest_left_gripper=self.latest_left_gripper,
            latest_right_gripper=self.latest_right_gripper,
            task_description=task_description,
        )

    def execute_action_step(self, left_arm, right_arm, left_gripper, right_gripper):
        """模拟执行动作"""
        self.latest_joint_state[:7] = left_arm
        self.latest_joint_state[7:14] = right_arm
        self.latest_left_gripper = left_gripper
        self.latest_right_gripper = right_gripper

    def execute_action_chunk_with_interpolation(self, action,
                                                 execution_horizon=8,
                                                 control_freq=20,
                                                 interpolation_factor=5,
                                                 transition_interpolation_factor=5):
        """模拟执行动作Chunk"""
        with self.hold_lock:
            self.is_holding = False

        print(f"\n🎬 开始执行 Action Chunk...")
        print(f"   执行前{execution_horizon}步, Chunk内{interpolation_factor}倍插值, Chunk间{transition_interpolation_factor}倍插值")

        dt = 1.0 / control_freq

        left_arm_orig = action['left_arm_joint_position'][0][:execution_horizon]
        right_arm_orig = action['right_arm_joint_position'][0][:execution_horizon]
        left_gripper_orig = action['left_effector_position'][0, :execution_horizon, 0]
        right_gripper_orig = action['right_effector_position'][0, :execution_horizon, 0]

        print(f"\n  📥 提取前 {execution_horizon} 步原始动作")
        print(f"     左臂J0: {left_arm_orig[0, 0]:+.4f} → {left_arm_orig[-1, 0]:+.4f}")
        print(f"     右臂J0: {right_arm_orig[0, 0]:+.4f} → {right_arm_orig[-1, 0]:+.4f}")

        if (self.last_chunk_left_arm is not None and
            self.last_chunk_right_arm is not None):

            dist_left = np.linalg.norm(left_arm_orig[0] - self.last_chunk_left_arm)
            dist_right = np.linalg.norm(right_arm_orig[0] - self.last_chunk_right_arm)

            print(f"\n  🔗 跨Chunk过渡:")
            print(f"     上个Chunk最后: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")
            print(f"     新Chunk第一步:  左J0={left_arm_orig[0, 0]:+.4f}, 右J0={right_arm_orig[0, 0]:+.4f}")
            print(f"     跳变距离: 左={dist_left:.4f} rad, 右={dist_right:.4f} rad")

            left_transition_traj = np.vstack([
                self.last_chunk_left_arm,
                left_arm_orig[0]
            ])

            right_transition_traj = np.vstack([
                self.last_chunk_right_arm,
                right_arm_orig[0]
            ])

            t_orig = np.linspace(0, 1, 2)
            t_new = np.linspace(0, 1, transition_interpolation_factor + 1)

            def interpolate_quintic(orig_traj, t_orig, t_new):
                N, D = orig_traj.shape
                interp_traj = np.zeros((len(t_new), D))
                for d in range(D):
                    if N == 2:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=1)
                    else:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=min(5, N-1))
                    interp_traj[:, d] = spl(t_new)
                return interp_traj

            left_transition_interp = interpolate_quintic(left_transition_traj, t_orig, t_new)
            right_transition_interp = interpolate_quintic(right_transition_traj, t_orig, t_new)

            gripper_t_orig = np.array([0, 1])
            gripper_t_new = t_new
            left_gripper_transition = np.interp(
                gripper_t_new,
                gripper_t_orig,
                [self.last_chunk_left_gripper, left_gripper_orig[0]]
            )
            right_gripper_transition = np.interp(
                gripper_t_new,
                gripper_t_orig,
                [self.last_chunk_right_gripper, right_gripper_orig[0]]
            )

            num_transition_steps = len(t_new) - 1

            print(f"  🔀 执行 {num_transition_steps} 步跨Chunk过渡插值 ({transition_interpolation_factor}倍)")

            for i in range(num_transition_steps):
                self.execute_action_step(
                    left_transition_interp[i],
                    right_transition_interp[i],
                    left_gripper_transition[i],
                    right_gripper_transition[i]
                )

                if i % 5 == 0 or i == num_transition_steps - 1:
                    progress = (i + 1) / num_transition_steps * 100
                    print(f"    过渡步 {i+1:2d}/{num_transition_steps} ({progress:5.1f}%): "
                          f"左J0={left_transition_interp[i, 0]:+.4f}, 右J0={right_transition_interp[i, 0]:+.4f}")

                time.sleep(dt)

        original_num_steps = execution_horizon
        new_num_steps = execution_horizon * interpolation_factor

        t_orig = np.linspace(0, 1, original_num_steps)
        t_new = np.linspace(0, 1, new_num_steps)

        def interpolate_quintic_chunk(orig_traj, t_orig, t_new):
            N, D = orig_traj.shape
            interp_traj = np.zeros((len(t_new), D))
            for d in range(D):
                spl = make_interp_spline(t_orig, orig_traj[:, d], k=5)
                interp_traj[:, d] = spl(t_new)
            return interp_traj

        left_arm_interp = interpolate_quintic_chunk(left_arm_orig, t_orig, t_new)
        right_arm_interp = interpolate_quintic_chunk(right_arm_orig, t_orig, t_new)

        left_gripper_interp = np.interp(t_new, t_orig, left_gripper_orig)
        right_gripper_interp = np.interp(t_new, t_orig, right_gripper_orig)

        print(f"\n  🔧 Chunk内插值: {original_num_steps}步 → {new_num_steps}步 ({interpolation_factor}倍)")

        print(f"  ▶️  执行 {new_num_steps} 步Chunk内插值动作 (频率: {control_freq}Hz, 耗时: {new_num_steps/control_freq:.2f}秒)")

        for step in range(new_num_steps):
            left_arm = left_arm_interp[step]
            right_arm = right_arm_interp[step]
            left_gripper = left_gripper_interp[step]
            right_gripper = right_gripper_interp[step]

            self.execute_action_step(left_arm, right_arm, left_gripper, right_gripper)

            if step % 10 == 0 or step == new_num_steps - 1:
                progress = (step + 1) / new_num_steps * 100
                print(f"    Chunk步 {step+1:3d}/{new_num_steps} ({progress:5.1f}%): "
                      f"左J0={left_arm[0]:+.4f}, 右J0={right_arm[0]:+.4f}")

            time.sleep(dt)

        print(f"  ✅ Action Chunk 执行完成")

        self.last_chunk_left_arm = np.array(left_arm_orig[-1])
        self.last_chunk_right_arm = np.array(right_arm_orig[-1])
        self.last_chunk_left_gripper = float(left_gripper_orig[-1])
        self.last_chunk_right_gripper = float(right_gripper_orig[-1])

        print(f"  💾 保存本次Chunk最后位置: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")

        with self.hold_lock:
            self.hold_left_arm = left_arm_interp[-1].copy()
            self.hold_right_arm = right_arm_interp[-1].copy()
            self.hold_left_gripper = left_gripper_interp[-1]
            self.hold_right_gripper = right_gripper_interp[-1]
            self.is_holding = True

        print(f"  🔒 已切换到保持模式")

    def shutdown(self):
        print("\n🛑 正在关闭节点...")
        print("✅ 节点已关闭")


class RobotInferenceNode(Node):

    def __init__(self, enable_control=False):
        super().__init__('robot_inference_node')
        self.bridge = CvBridge()
        self.enable_control = enable_control

        self.latest_images = {}
        self.latest_joint_state = None
        self.latest_left_gripper = None
        self.latest_right_gripper = None

        self.last_chunk_left_arm = None
        self.last_chunk_right_arm = None
        self.last_chunk_left_gripper = None
        self.last_chunk_right_gripper = None

        self.is_holding = False
        self.hold_left_arm = None
        self.hold_right_arm = None
        self.hold_left_gripper = None
        self.hold_right_gripper = None
        self.hold_lock = threading.Lock()
        self._hold_thread_running = False

        self.data_ready = {
            'head_image': False,
            'left_hand_image': False,
            'right_hand_image': False,
            'joint_state': False,
            'left_gripper': False,
            'right_gripper': False,
        }

        print("📡 创建ROS2订阅...")

        self.create_subscription(Image, '/camera/head_color', self._head_callback, 70)
        self.create_subscription(Image, '/camera/hand_left_color', self._left_hand_callback, 70)
        self.create_subscription(Image, '/camera/hand_right_color', self._right_hand_callback, 70)

        self.create_subscription(JointState, '/hal/arm_joint_state', self._joint_callback, 70)

        if EndState is not None:
            self.create_subscription(EndState, '/hal/left_ee_data', self._left_gripper_callback, 70)
            self.create_subscription(EndState, '/hal/right_ee_data', self._right_gripper_callback, 70)
            print("✅ 夹爪订阅已创建")

        if self.enable_control:
            self.arm_pub = self.create_publisher(JointState, '/wbc/arm_command', 10)
            self.left_gripper_pub = self.create_publisher(JointState, '/wbc/left_ee_command', 10)
            self.right_gripper_pub = self.create_publisher(JointState, '/wbc/right_ee_command', 10)
            print("✅ 控制发布者已创建")

            self._hold_thread_running = True
            self.hold_thread = threading.Thread(
                target=self._hold_position_loop,
                daemon=True
            )
            self.hold_thread.start()
            print("✅ 后台位置保持线程已启动 (20Hz)")

        print("✅ ROS2订阅已创建")
        print("⏳ 等待ROS2连接建立 (2秒)...")
        time.sleep(2.0)

    def _hold_position_loop(self):
        rate = 20
        dt = 1.0 / rate

        print("🔄 位置保持线程开始运行...")

        while self._hold_thread_running and rclpy.ok():
            with self.hold_lock:
                if self.is_holding and self.hold_left_arm is not None:
                    self.execute_action_step(
                        self.hold_left_arm,
                        self.hold_right_arm,
                        self.hold_left_gripper,
                        self.hold_right_gripper
                    )
            time.sleep(dt)

        print("🔄 位置保持线程已退出")

    def _head_callback(self, msg):
        try:
            self.latest_images['top_head'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['head_image'] = True
        except Exception:
            try:
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images['top_head'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['head_image'] = True
            except Exception:
                pass

    def _left_hand_callback(self, msg):
        try:
            self.latest_images['hand_left'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['left_hand_image'] = True
        except Exception:
            try:
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images['hand_left'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['left_hand_image'] = True
            except Exception:
                pass

    def _right_hand_callback(self, msg):
        try:
            self.latest_images['hand_right'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['right_hand_image'] = True
        except Exception:
            try:
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images['hand_right'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['right_hand_image'] = True
            except Exception:
                pass

    def _joint_callback(self, msg):
        self.latest_joint_state = msg.position
        self.data_ready['joint_state'] = True

    def _left_gripper_callback(self, msg):
        try:
            if len(msg.end_state) > 0:
                self.latest_left_gripper = float(msg.end_state[0].position)
                self.data_ready['left_gripper'] = True
        except Exception as e:
            print(f"⚠️ 左夹爪数据解析错误: {e}")

    def _right_gripper_callback(self, msg):
        try:
            if len(msg.end_state) > 0:
                self.latest_right_gripper = float(msg.end_state[0].position)
                self.data_ready['right_gripper'] = True
        except Exception as e:
            print(f"⚠️ 右夹爪数据解析错误: {e}")

    # def is_data_ready(self):
    #     return True

    def is_data_ready(self):
        required_images = {
            "top_head",
            "hand_left",
            "hand_right",
        }

        return (
            required_images.issubset(self.latest_images.keys())
            and self.latest_joint_state is not None
            and self.latest_left_gripper is not None
            and self.latest_right_gripper is not None
            and all(self.data_ready.values())
        )

    def get_data_status(self):
        status_items = []
        status_items.append(f"头部图像: {'✓' if self.data_ready['head_image'] else '✗'}")
        status_items.append(f"左手图像: {'✓' if self.data_ready['left_hand_image'] else '✗'}")
        status_items.append(f"右手图像: {'✓' if self.data_ready['right_hand_image'] else '✗'}")
        status_items.append(f"关节状态: {'✓' if self.data_ready['joint_state'] else '✗'}")

        left_grip_status = f"✓({self.latest_left_gripper:.1f}mm)" if self.data_ready['left_gripper'] else "✗"
        right_grip_status = f"✓({self.latest_right_gripper:.1f}mm)" if self.data_ready['right_gripper'] else "✗"
        status_items.append(f"左夹爪: {left_grip_status}")
        status_items.append(f"右夹爪: {right_grip_status}")

        return " | ".join(status_items)

    def get_missing_data(self):
        missing = []
        data_names = {
            'head_image': '头部图像',
            'left_hand_image': '左手图像',
            'right_hand_image': '右手图像',
            'joint_state': '关节状态',
            'left_gripper': '左夹爪状态',
            'right_gripper': '右夹爪状态',
        }

        for key, ready in self.data_ready.items():
            if not ready:
                missing.append(data_names[key])

        return missing

    def get_observation(self, task_description=DEFAULT_TASK_DESCRIPTION):
        if not self.is_data_ready():
            return None

        return build_g01_observation(
            latest_images=self.latest_images,
            latest_joint_state=self.latest_joint_state,
            latest_left_gripper=self.latest_left_gripper,
            latest_right_gripper=self.latest_right_gripper,
            task_description=task_description,
        )

    def execute_action_step(self, left_arm, right_arm, left_gripper, right_gripper):
        if not self.enable_control:
            return

        arm_msg = JointState()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.position = left_arm.tolist() + right_arm.tolist()
        self.arm_pub.publish(arm_msg)

        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = self.get_clock().now().to_msg()
        left_gripper_msg.name = ['left_gripper_joint1']
        left_gripper_msg.position = [float(left_gripper)]
        self.left_gripper_pub.publish(left_gripper_msg)

        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = self.get_clock().now().to_msg()
        right_gripper_msg.name = ['right_gripper_joint1']
        right_gripper_msg.position = [float(right_gripper)]
        self.right_gripper_pub.publish(right_gripper_msg)

    def execute_action_chunk_with_interpolation(self, action,
                                                 execution_horizon=8,
                                                 control_freq=20,
                                                 interpolation_factor=5,
                                                 transition_interpolation_factor=5):
        if not self.enable_control:
            return

        with self.hold_lock:
            self.is_holding = False

        print(f"\n🎬 开始执行 Action Chunk...")
        print(f"   执行前{execution_horizon}步, Chunk内{interpolation_factor}倍插值, Chunk间{transition_interpolation_factor}倍插值")

        dt = 1.0 / control_freq

        left_arm_orig = action['left_arm_joint_position'][0][:execution_horizon]
        right_arm_orig = action['right_arm_joint_position'][0][:execution_horizon]
        left_gripper_orig = action['left_effector_position'][0, :execution_horizon, 0]
        right_gripper_orig = action['right_effector_position'][0, :execution_horizon, 0]

        print(f"\n  📥 提取前 {execution_horizon} 步原始动作")
        print(f"     左臂J0: {left_arm_orig[0, 0]:+.4f} → {left_arm_orig[-1, 0]:+.4f}")
        print(f"     右臂J0: {right_arm_orig[0, 0]:+.4f} → {right_arm_orig[-1, 0]:+.4f}")

        if (self.last_chunk_left_arm is not None and
            self.last_chunk_right_arm is not None):

            dist_left = np.linalg.norm(left_arm_orig[0] - self.last_chunk_left_arm)
            dist_right = np.linalg.norm(right_arm_orig[0] - self.last_chunk_right_arm)

            print(f"\n  🔗 跨Chunk过渡:")
            print(f"     上个Chunk最后: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")
            print(f"     新Chunk第一步:  左J0={left_arm_orig[0, 0]:+.4f}, 右J0={right_arm_orig[0, 0]:+.4f}")
            print(f"     跳变距离: 左={dist_left:.4f} rad, 右={dist_right:.4f} rad")

            left_transition_traj = np.vstack([
                self.last_chunk_left_arm,
                left_arm_orig[0]
            ])

            right_transition_traj = np.vstack([
                self.last_chunk_right_arm,
                right_arm_orig[0]
            ])

            t_orig = np.linspace(0, 1, 2)
            t_new = np.linspace(0, 1, transition_interpolation_factor + 1)

            def interpolate_quintic(orig_traj, t_orig, t_new):
                N, D = orig_traj.shape
                interp_traj = np.zeros((len(t_new), D))
                for d in range(D):
                    if N == 2:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=1)
                    else:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=min(5, N-1))
                    interp_traj[:, d] = spl(t_new)
                return interp_traj

            left_transition_interp = interpolate_quintic(left_transition_traj, t_orig, t_new)
            right_transition_interp = interpolate_quintic(right_transition_traj, t_orig, t_new)

            gripper_t_orig = np.array([0, 1])
            gripper_t_new = t_new
            left_gripper_transition = np.interp(
                gripper_t_new,
                gripper_t_orig,
                [self.last_chunk_left_gripper, left_gripper_orig[0]]
            )
            right_gripper_transition = np.interp(
                gripper_t_new,
                gripper_t_orig,
                [self.last_chunk_right_gripper, right_gripper_orig[0]]
            )

            num_transition_steps = len(t_new) - 1

            print(f"  🔀 执行 {num_transition_steps} 步跨Chunk过渡插值 ({transition_interpolation_factor}倍)")

            for i in range(num_transition_steps):
                self.execute_action_step(
                    left_transition_interp[i],
                    right_transition_interp[i],
                    left_gripper_transition[i],
                    right_gripper_transition[i]
                )

                if i % 5 == 0 or i == num_transition_steps - 1:
                    progress = (i + 1) / num_transition_steps * 100
                    print(f"    过渡步 {i+1:2d}/{num_transition_steps} ({progress:5.1f}%): "
                          f"左J0={left_transition_interp[i, 0]:+.4f}, 右J0={right_transition_interp[i, 0]:+.4f}")

                time.sleep(dt)
                rclpy.spin_once(self, timeout_sec=0.001)

        original_num_steps = execution_horizon
        new_num_steps = execution_horizon * interpolation_factor

        t_orig = np.linspace(0, 1, original_num_steps)
        t_new = np.linspace(0, 1, new_num_steps)

        def interpolate_quintic_chunk(orig_traj, t_orig, t_new):
            N, D = orig_traj.shape
            interp_traj = np.zeros((len(t_new), D))
            for d in range(D):
                spl = make_interp_spline(t_orig, orig_traj[:, d], k=5)
                interp_traj[:, d] = spl(t_new)
            return interp_traj

        left_arm_interp = interpolate_quintic_chunk(left_arm_orig, t_orig, t_new)
        right_arm_interp = interpolate_quintic_chunk(right_arm_orig, t_orig, t_new)

        left_gripper_interp = np.interp(t_new, t_orig, left_gripper_orig)
        right_gripper_interp = np.interp(t_new, t_orig, right_gripper_orig)

        print(f"\n  🔧 Chunk内插值: {original_num_steps}步 → {new_num_steps}步 ({interpolation_factor}倍)")

        print(f"  ▶️  执行 {new_num_steps} 步Chunk内插值动作 (频率: {control_freq}Hz, 耗时: {new_num_steps/control_freq:.2f}秒)")

        for step in range(new_num_steps):
            left_arm = left_arm_interp[step]
            right_arm = right_arm_interp[step]
            left_gripper = left_gripper_interp[step]
            right_gripper = right_gripper_interp[step]

            self.execute_action_step(left_arm, right_arm, left_gripper, right_gripper)

            if step % 10 == 0 or step == new_num_steps - 1:
                progress = (step + 1) / new_num_steps * 100
                print(f"    Chunk步 {step+1:3d}/{new_num_steps} ({progress:5.1f}%): "
                      f"左J0={left_arm[0]:+.4f}, 右J0={right_arm[0]:+.4f}")

            time.sleep(dt)
            rclpy.spin_once(self, timeout_sec=0.001)

        print(f"  ✅ Action Chunk 执行完成")

        self.last_chunk_left_arm = np.array(left_arm_orig[-1])
        self.last_chunk_right_arm = np.array(right_arm_orig[-1])
        self.last_chunk_left_gripper = float(left_gripper_orig[-1])
        self.last_chunk_right_gripper = float(right_gripper_orig[-1])

        print(f"  💾 保存本次Chunk最后位置: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")

        with self.hold_lock:
            self.hold_left_arm = left_arm_interp[-1].copy()
            self.hold_right_arm = right_arm_interp[-1].copy()
            self.hold_left_gripper = left_gripper_interp[-1]
            self.hold_right_gripper = right_gripper_interp[-1]
            self.is_holding = True

        print(f"  🔒 已切换到保持模式（后台线程接管）")

        for i in range(5):
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

    def shutdown(self):
        print("\n🛑 正在关闭节点...")

        self._hold_thread_running = False
        if hasattr(self, 'hold_thread'):
            self.hold_thread.join(timeout=2.0)

        print("✅ 节点已关闭")


def wait_for_data(node, timeout=30):
    print(f"\n⏳ 等待所有必需数据 (超时: {timeout}秒)...")
    print("   必需: 3个图像 + 关节状态 + 2个夹爪状态")

    start_time = time.time()
    last_print_time = 0

    while time.time() - start_time < timeout:
        if ROS2_AVAILABLE:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.is_data_ready():
            print(f"\n✅ 所有数据就绪! (耗时: {time.time() - start_time:.1f}秒)")
            return True

        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            elapsed = current_time - start_time
            current_status = node.get_data_status()
            print(f"  [{elapsed:.1f}s] {current_status}")
            last_print_time = current_time

    print(f"\n❌ 超时! 未接收到完整数据")
    print(f"   最终状态: {node.get_data_status()}")

    missing = node.get_missing_data()
    if missing:
        print(f"   ❌ 缺失数据: {', '.join(missing)}")

    return False


def save_observation_images(obs, save_dir="observation_samples"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = []
    for camera_name in ("top_head", "hand_left", "hand_right"):
        if camera_name not in obs["images"]:
            continue

        image_rgb = np.asarray(obs["images"][camera_name])
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filename = f"{save_dir}/{timestamp}_{camera_name}.png"
        cv2.imwrite(filename, image_bgr)
        saved_files.append(filename)

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="ACoT-VLA 推理脚本")
    parser.add_argument("--test-mode", action="store_true", help="使用模拟数据进行测试，不依赖ROS2")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="推理服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="推理服务器端口")
    parser.add_argument("--enable-control", action="store_true", help="启用控制模式")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 ACoT-VLA - Action Chunking + 统一插值策略")
    print("=" * 80)

    ENABLE_CONTROL = args.enable_control
    CONTROL_FREQ = 20
    EXECUTION_HORIZON = 8
    INTERPOLATION_FACTOR = 5
    TRANSITION_INTERPOLATION_FACTOR = 5
    ACTION_HORIZON = 16
    SERVER_HOST = args.host
    SERVER_PORT = args.port

    transition_steps = TRANSITION_INTERPOLATION_FACTOR
    chunk_steps = EXECUTION_HORIZON * INTERPOLATION_FACTOR
    total_steps_per_cycle = transition_steps + chunk_steps
    cycle_time = total_steps_per_cycle / CONTROL_FREQ

    if ENABLE_CONTROL and not args.test_mode:
        print("\n⚠️  控制已启用! 机器人将执行推理动作!")
        print(f"\n📋 控制参数:")
        print(f"   控制频率: {CONTROL_FREQ}Hz")
        print(f"   模型预测步数: {ACTION_HORIZON}步")
        print(f"   执行步数: 前{EXECUTION_HORIZON}步 (丢弃{ACTION_HORIZON-EXECUTION_HORIZON}步)")
        print(f"\n   📊 插值策略（统一）:")
        print(f"   ├─ Chunk间过渡: 2步 → {TRANSITION_INTERPOLATION_FACTOR+1}步 ({TRANSITION_INTERPOLATION_FACTOR}倍插值, 执行{TRANSITION_INTERPOLATION_FACTOR}步)")
        print(f"   └─ Chunk内动作: {EXECUTION_HORIZON}步 → {chunk_steps}步 ({INTERPOLATION_FACTOR}倍插值)")
        print(f"\n   ⏱️  时间分析:")
        print(f"   ├─ 跨Chunk过渡: {transition_steps}步 × {1000/CONTROL_FREQ:.0f}ms = {transition_steps/CONTROL_FREQ:.2f}秒")
        print(f"   ├─ Chunk内执行: {chunk_steps}步 × {1000/CONTROL_FREQ:.0f}ms = {chunk_steps/CONTROL_FREQ:.2f}秒")
        print(f"   └─ 总耗时: {cycle_time:.2f}秒/轮 (推理频率: ~{1/cycle_time:.2f}Hz)")
        print(f"\n✨ 核心优势:")
        print(f"   1. Action Chunking: 高频感知-行动闭环")
        print(f"   2. 统一插值策略: Chunk间和Chunk内使用相同插值倍数")
        print(f"   3. 可调节参数: 独立控制两种插值倍数")
        print(f"   4. 后台保持线程: 无停顿")
        print("\n   请确保机器人周围安全!")
        response = input("\n   继续? (yes/no): ")
        if response.lower() != 'yes':
            print("已取消")
            return

    if args.test_mode:
        print("\n[1/3] 使用测试模式 (模拟数据)...")
        node = MockInferenceNode()
        print("✅ 模拟节点已创建")
    else:
        if not ROS2_AVAILABLE:
            print("❌ ROS2不可用! 请使用 --test-mode 参数进行测试，或安装ROS2环境")
            return

        print("\n[1/3] 初始化ROS2...")
        rclpy.init()
        node = RobotInferenceNode(enable_control=ENABLE_CONTROL)

    print("\n[2/3] 连接推理服务器...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=SERVER_HOST,
            port=SERVER_PORT
        )
        print(f"✅ 连接成功! 服务器元数据: {client.get_server_metadata()}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        node.shutdown()
        if not args.test_mode and ROS2_AVAILABLE:
            rclpy.shutdown()
        return

    print("\n[3/3] 等待所有必需数据...")
    if not wait_for_data(node, timeout=30):
        print("\n💡 调试提示:")
        print("   1. 检查机器人是否启动")
        print("   2. 检查copilot模式")
        node.shutdown()
        if not args.test_mode and ROS2_AVAILABLE:
            rclpy.shutdown()
        return

    obs = node.get_observation()
    if obs:
        print("\n📊 初始状态:")
        print(f"   state shape: {obs['state'].shape}, dtype: {obs['state'].dtype}")
        print(f"   左臂J0: {obs['state'][0]:.4f} rad")
        print(f"   右臂J0: {obs['state'][7]:.4f} rad")
        save_observation_images(obs)

    print("\n" + "=" * 80)
    print("🎯 开始 Action Chunking + 统一插值 循环")
    print(f"   流程: 推理 → 跨Chunk过渡({TRANSITION_INTERPOLATION_FACTOR}倍) → Chunk内执行({INTERPOLATION_FACTOR}倍) → 保持 → 推理")
    print("   特性: 统一插值策略，可独立调节Chunk间和Chunk内插值倍数")
    print("   按 Ctrl+C 退出")
    print("=" * 80)

    try:
        count = 0
        total_steps_executed = 0

        while True:
            if not args.test_mode and ROS2_AVAILABLE and not rclpy.ok():
                break

            cycle_start = time.time()

            if not args.test_mode and ROS2_AVAILABLE:
                rclpy.spin_once(node, timeout_sec=0.001)
            obs = node.get_observation()

            if obs is None:
                time.sleep(0.1)
                continue

            count += 1

            print(f"\n\n{'#'*80}")
            print(f"# Action Chunking 循环 #{count}")
            print(f"{'#'*80}")

            pre_inference_joints = np.array(node.latest_joint_state)
            print(f"\n📍 当前状态:")
            print(f"   左臂J0: {pre_inference_joints[0]:+.4f} rad")
            print(f"   右臂J0: {pre_inference_joints[7]:+.4f} rad")
            print(f"   已执行总步数: {total_steps_executed}")

            with node.hold_lock:
                hold_status = "🔒 保持中" if node.is_holding else "⏸️  空闲"
            print(f"   保持状态: {hold_status}")

            if count == 1:
                print("\n🔎 发送给G01服务端的观测:")
                print(f"   keys: {list(obs.keys())}")
                print(
                    f"   state: shape={obs['state'].shape}, "
                    f"dtype={obs['state'].dtype}"
                )
                for camera_name, image in obs["images"].items():
                    print(
                        f"   {camera_name}: "
                        f"shape={image.shape}, dtype={image.dtype}"
                    )
                print(f"   prompt: {obs['prompt']}")

            print(f"\n🧠 正在推理 (预测{ACTION_HORIZON}步)...")
            inference_start = time.time()
            raw_action = client.infer(obs)
            action = adapt_g01_action(raw_action)
            inference_time = time.time() - inference_start

            print(f"✅ 推理完成 (耗时: {inference_time*1000:.1f}ms)")

            if 'server_timing' in action:
                print(f"   服务器推理时间: {action['server_timing'].get('infer_ms', 'N/A')}ms")
            if 'policy_timing' in action:
                print(f"   策略推理时间: {action['policy_timing'].get('infer_ms', 'N/A')}ms")

            target_left_j0 = action['left_arm_joint_position'][0, EXECUTION_HORIZON-1, 0]
            target_right_j0 = action['right_arm_joint_position'][0, EXECUTION_HORIZON-1, 0]
            print(f"\n🎯 本轮目标 (原始第{EXECUTION_HORIZON}步):")
            print(f"   左臂J0: {target_left_j0:+.4f} rad")
            print(f"   右臂J0: {target_right_j0:+.4f} rad")

            if ENABLE_CONTROL or args.test_mode:
                node.execute_action_chunk_with_interpolation(
                    action,
                    execution_horizon=EXECUTION_HORIZON,
                    control_freq=CONTROL_FREQ,
                    interpolation_factor=INTERPOLATION_FACTOR,
                    transition_interpolation_factor=TRANSITION_INTERPOLATION_FACTOR
                )

                if count > 1:
                    total_steps_executed += transition_steps + chunk_steps
                else:
                    total_steps_executed += chunk_steps

                if not args.test_mode and ROS2_AVAILABLE:
                    rclpy.spin_once(node, timeout_sec=0.01)
                final_joints = np.array(node.latest_joint_state)

                print(f"\n📊 执行结果:")
                print(f"   目标: 左J0={target_left_j0:+.4f}, 右J0={target_right_j0:+.4f}")
                print(f"   实际: 左J0={final_joints[0]:+.4f}, 右J0={final_joints[7]:+.4f}")
                print(f"   误差: 左J0={abs(final_joints[0]-target_left_j0):.4f}, "
                      f"右J0={abs(final_joints[7]-target_right_j0):.4f} rad")
            else:
                print("  ⏭️  跳过执行 (控制未启用)")

            cycle_time_actual = time.time() - cycle_start
            actual_freq = 1.0 / cycle_time_actual if cycle_time_actual > 0 else 0

            print(f"\n⏱️  本轮耗时: {cycle_time_actual:.2f}秒 (预期: {cycle_time:.2f}秒, 频率: {actual_freq:.2f}Hz)")

            if cycle_time_actual > cycle_time * 1.3:
                print(f"⚠️  警告: 本轮耗时超过预期30%")

            print(f"💡 后台线程正在保持最后位置，等待下次推理...")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    finally:
        node.shutdown()
        if not args.test_mode and ROS2_AVAILABLE:
            rclpy.shutdown()
        print("🔌 已关闭")


if __name__ == "__main__":
    main()
