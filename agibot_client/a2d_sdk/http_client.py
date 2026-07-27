#!/usr/bin/env python3
"""
Agibot 策略执行脚本
使用真实机器人图像和状态，执行模型推理并控制机器人
"""
import sys
sys.path.insert(0, '/home/nwrobot/openpi-main/agibot_client')

import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from openpi_client import websocket_client_policy as client
import cv2

try:
    from a2d_sdk.robot import RobotController
    from genie_msgs.msg import EndState
    HAVE_SDK = True
except:
    HAVE_SDK = False
    EndState = None
    print("⚠️ 警告:  未找到机器人SDK，将运行模拟模式")

GRIPPER_MAX = 120.0
DT = 0.15  # 每步时间间隔（秒）

def decode_image(msg:  Image) -> np.ndarray:
    """解码 ROS Image 消息为 numpy 数组 (H, W, C) 格式"""
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != h * msg.step:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = data.reshape((h, msg.step))[: , :w*3]. reshape((h, w, 3))
    img = img[..., ::-1].copy()  # BGR → RGB
    return cv2.resize(img, (640, 480))

class AgibotPolicyNode(Node):
    def __init__(self):
        super().__init__('agibot_policy_node')
       
        # 初始化状态
        self.images = {'head': None, 'hand_left': None, 'hand_right': None}
        self. arm_state = np.zeros(14, dtype=np.float32)
        self.gripper_state = np.zeros(2, dtype=np.float32)
        self.has_real_state = False
       
        # 初始化机器人控制器
        if HAVE_SDK:
            try:
                self.robot = RobotController()
                print("✓ 机器人控制器初始化成功")
            except Exception as e:
                print(f"✗ 机器人控制器初始化失败: {e}")
                self.robot = None
        else:
            self.robot = None
       
        # 订阅 topics
        self.create_subscription(Image, '/camera/head_color',
                                lambda m: self._img_cb(m, 'head'), 10)
        self.create_subscription(Image, '/camera/hand_left_color',
                                lambda m:  self._img_cb(m, 'hand_left'), 10)
        self.create_subscription(Image, '/camera/hand_right_color',
                                lambda m: self._img_cb(m, 'hand_right'), 10)
        self.create_subscription(JointState, '/hal/arm_joint_state', self._arm_cb, 10)
       
        if HAVE_SDK and EndState:
            self.create_subscription(EndState, '/hal/left_ee_data',
                                    lambda m: self._gripper_cb(m, 0), 10)
            self.create_subscription(EndState, '/hal/right_ee_data',
                                    lambda m: self._gripper_cb(m, 1), 10)
       
        # 等待传感器就绪
        print("等待传感器数据...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            # 检查图像和关节状态
            if (all(v is not None for v in self.images.values()) and
                np.any(self.arm_state != 0)):  # 确保关节数据不全为 0
                self.has_real_state = True
                break
       
        print("✓ 传感器就绪")
        print(f"  初始关节状态: {self.arm_state[: 3]} ...  (前3个关节)")
        print(f"  初始夹爪状态: {self.gripper_state}\n")

    def _img_cb(self, msg, name):
        """图像回调"""
        try:
            self.images[name] = decode_image(msg)
        except:
            pass
   
    def _arm_cb(self, msg):
        """关节状态回调"""
        p = np.array(msg.position, dtype=np.float32)
        if len(p) >= 14:
            self.arm_state = p[: 14]
            if not self.has_real_state and np.any(self.arm_state != 0):
                self. has_real_state = True
   
    def _gripper_cb(self, msg, idx):
        """夹爪回调"""
        try:
            pos = getattr(msg, 'position', [0.0])
            val = float(pos[0]) if isinstance(pos, (list, tuple)) else float(pos)
            self.gripper_state[idx] = np.clip(val / GRIPPER_MAX, 0, 1)
        except:
            pass
   
    def get_observation(self, prompt):
        """
        获取当前观察
       
        返回的图像格式为 (H, W, C)，服务器端会自动处理
        """
        rclpy.spin_once(self, timeout_sec=0.01)
       
        state = np.concatenate([self.arm_state, self.gripper_state]).astype(np.float32)
       
        # ✅ 处理图像为 None 的情况
        def ensure_image(img):
            """确保图像存在，格式为 (H, W, C)"""
            if img is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return img
       
        return {
            "observation.images.top_head": ensure_image(self.images['head']),
            "observation.images.hand_left": ensure_image(self.images['hand_left']),
            "observation.images.hand_right": ensure_image(self.images['hand_right']),
            "observation.state": state,
            "prompt": prompt
        }
   
    def execute_actions(self, actions:  np.ndarray):
        """
        执行动作序列
       
        Args:
            actions: (N, 16) 或 (N, 22) 的动作数组，前16维有效
        """
        if not HAVE_SDK or self.robot is None:
            print(f"  [模拟模式] 跳过执行 {len(actions)} 步动作")
            return
       
        # 只取前16维
        actions = actions[:, :16] if actions.shape[1] > 16 else actions
       
        robot_actions = []
        for action in actions:
            # 关节位置（已经是绝对位置）
            l_arm = action[: 7]. tolist()
            r_arm = action[7:14].tolist()
           
            # 夹爪位置（裁剪到 [0, 1]，然后转为 [0, 120]）
            l_grip = np.clip(float(action[14]), 0.0, 1.0) * GRIPPER_MAX
            r_grip = np.clip(float(action[15]), 0.0, 1.0) * GRIPPER_MAX
           
            robot_actions.append({
                "left_arm": {"action_data": l_arm, "control_type": "ABS_JOINT"},
                "right_arm": {"action_data": r_arm, "control_type":  "ABS_JOINT"},
                "gripper": {
                    "action_data": [l_grip, r_grip],
                    "control_type":  "ABS_JOINT"
                }
            })
       
        # 发送轨迹
        try:
            self.robot.trajectory_tracking_control(
                infer_timestamp=int(time.time() * 1e9),
                robot_states={
                    "head": [0, 0],
                    "waist": [0, 0],
                    "arm":  self.arm_state. tolist()
                },
                robot_actions=robot_actions,
                robot_link="base_link",
                trajectory_reference_time=len(actions) * DT
            )
            print(f"  ✓ 执行 {len(actions)} 步 ({len(actions)*DT:.1f}s)")
        except Exception as e:
            print(f"  ✗ 执行失败:  {e}")

def main():
    rclpy.init()
   
    # 初始化节点
    node = AgibotPolicyNode()
   
    # 连接策略服务器
    print("连接策略服务器...")
    try:
        policy = client.WebsocketClientPolicy(host="localhost", port=8000)
        print("✓ 策略服务器连接成功\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
   
    task_prompt = "Pick up the red cube on the desk, place it into the box!"
   
    print("="*80)
    print("开始策略执行")
    print(f"任务: {task_prompt}")
    print("="*80)
    print("提示: 按 Ctrl+C 停止\n")
   
    step = 0
    try:
        while rclpy. ok():
            # 获取观察
            obs = node.get_observation(task_prompt)
           
            # 推理（计时）
            start_time = time. time()
            result = policy.infer(obs)
            actions = result["actions"]
            infer_time = time.time() - start_time
           
            # 计算轨迹时间
            trajectory_time = len(actions) * DT
           
            # 打印状态
            print(f"Step {step:3d} | "
                  f"推理: {infer_time*1000:5.1f}ms | "
                  f"轨迹: {trajectory_time:.2f}s | "
                  f"状态: L={obs['observation.state'][0]:.2f} R={obs['observation.state'][7]:.2f} | "
                  f"夹爪: {obs['observation.state'][15]:.2f}")
           
            # 执行动作
            node.execute_actions(actions)
           
            # 动态等待：确保机器人完成当前轨迹
            elapsed = time.time() - start_time
            sleep_time = max(0, trajectory_time - elapsed)
            time.sleep(sleep_time)
           
            step += 1
           
    except KeyboardInterrupt:
        print(f"\n\n用户中断 (执行了 {step} 步)")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback. print_exc()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()
