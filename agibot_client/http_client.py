#!/usr/bin/env python3
"""
Agibot 推理诊断脚本（只推理，不执行动作）
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

GRIPPER_MAX = 120. 0

def decode_image(msg:  Image) -> np.ndarray:
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != h * msg.step:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = data.reshape((h, msg.step))[: , :w*3]. reshape((h, w, 3))
    img = img[..., ::-1].copy()
    return cv2.resize(img, (640, 480))

class DiagnosticNode(Node):
    def __init__(self):
        super().__init__('diagnostic_node')
       
        self.images = {'head': None, 'hand_left': None, 'hand_right': None}
        self. arm_state = np.zeros(14, dtype=np.float32)
        self.gripper_state = np.zeros(2, dtype=np.float32)
       
        self.create_subscription(Image, '/camera/head_color',
                                lambda m: self._img_cb(m, 'head'), 10)
        self.create_subscription(Image, '/camera/hand_left_color',
                                lambda m: self._img_cb(m, 'hand_left'), 10)
        self.create_subscription(Image, '/camera/hand_right_color',
                                lambda m:  self._img_cb(m, 'hand_right'), 10)
        self.create_subscription(JointState, '/hal/arm_joint_state', self._arm_cb, 10)
       
        if HAVE_SDK and EndState:
            self.create_subscription(EndState, '/hal/left_ee_data',
                                    lambda m: self._gripper_cb(m, 0), 10)
            self.create_subscription(EndState, '/hal/right_ee_data',
                                    lambda m:  self._gripper_cb(m, 1), 10)
       
        print("等待传感器数据...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if all(v is not None for v in self.images.values()):
                break
        print("✓ 传感器就绪\n")
   
    def _img_cb(self, msg, name):
        try:
            self.images[name] = decode_image(msg)
        except:
            pass
   
    def _arm_cb(self, msg):
        p = np.array(msg.position, dtype=np.float32)
        if len(p) >= 14:
            self.arm_state = p[: 14]
   
    def _gripper_cb(self, msg, idx):
        try:
            pos = getattr(msg, 'position', [0.0])
            val = float(pos[0]) if isinstance(pos, (list, tuple)) else float(pos)
            self.gripper_state[idx] = np.clip(val / GRIPPER_MAX, 0, 1)
        except:
            pass
   
    def get_observation(self, prompt):
        rclpy.spin_once(self, timeout_sec=0.0)
        state = np.concatenate([self.arm_state, self.gripper_state]).astype(np.float32)
        return {
            "observation.images.top_head": self.images['head'],
            "observation.images.hand_left": self.images['hand_left'],
            "observation.images.hand_right": self.images['hand_right'],
            "observation.state": state,
            "prompt": prompt
        }

def main():
    rclpy.init()
    node = DiagnosticNode()
   
    # 连接策略服务器
    print("连接策略服务器...")
    try:
        policy = client.WebsocketClientPolicy(host="localhost", port=8000)
        print("✓ 策略服务器连接成功\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
   
    task_prompt = "Pick up the red cube on the desk, place it into the box!"
   
    print("=" * 80)
    print("开始诊断推理 (运行 10 步)")
    print("=" * 80)
   
    try:
        for step in range(10):
            # 获取观察
            obs = node.get_observation(task_prompt)
           
            # 打印当前状态
            print(f"\n{'='*80}")
            print(f"Step {step}")
            print(f"{'='*80}")
            print(f"当前机器人状态 (observation.state, 16维):")
            print(f"  左臂 (7): {obs['observation.state'][:7]}")
            print(f"  右臂 (7): {obs['observation.state'][7:14]}")
            print(f"  夹爪 (2): {obs['observation.state'][14:16]}")
            print(f"  范围:  [{obs['observation.state']. min():.3f}, {obs['observation.state'].max():.3f}]")
           
            # 推理
            start = time.time()
            result = policy.infer(obs)
            actions = result["actions"]  # (10, 16)
            infer_time = (time.time() - start) * 1000
           
            print(f"\n推理结果 (actions, shape={actions.shape}):")
            print(f"  推理时间: {infer_time:.1f}ms")
            print(f"  动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"\n第1步动作 (actions[0]):")
            print(f"  左臂 (7): {actions[0][:7]}")
            print(f"  右臂 (7): {actions[0][7:14]}")
            print(f"  夹爪 (2): {actions[0][14:16]}")
           
            print(f"\n第10步动作 (actions[9]):")
            print(f"  左臂 (7): {actions[9][:7]}")
            print(f"  右臂 (7): {actions[9][7:14]}")
            print(f"  夹爪 (2): {actions[9][14:16]}")
           
            # 检查异常值
            if np.any(np.abs(actions) > 3. 0):
                print(f"\n⚠️ 警告: 动作值过大!")
                large_values = np.where(np.abs(actions) > 3.0)
                print(f"  异常位置: {large_values}")
                print(f"  异常值:  {actions[large_values]}")
           
            time.sleep(1. 0)  # 等待1秒再继续
           
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()
