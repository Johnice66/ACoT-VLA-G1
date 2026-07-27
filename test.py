# robot_observation_action_chunking_unified.py
# GR00T Action Chunking + 统一插值策略（Chunk间和Chunk内使用相同插值倍数）

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from gr00t.policy.server_client import PolicyClient
import time
import os
import threading
from datetime import datetime
from scipy.interpolate import make_interp_spline
try:
    from genie_msgs.msg import EndState
except ImportError:
    EndState = None


class RobotObservationNode(Node):
    """机器人观测采集节点 - Action Chunking + 统一插值策略"""
   
    def __init__(self, enable_control=False):
        super().__init__('robot_observation_node')
        self.bridge = CvBridge()
        self.enable_control = enable_control
       
        # 存储最新数据
        self.latest_images = {}
        self.latest_joint_state = None
        self.latest_left_gripper = None
        self.latest_right_gripper = None

        # ⭐⭐⭐ 存储上一个 Chunk 的**原始**最后一步（用于跨Chunk过渡）⭐⭐⭐
        self.last_chunk_left_arm = None       # 上一个Chunk的原始最后位置
        self.last_chunk_right_arm = None
        self.last_chunk_left_gripper = None
        self.last_chunk_right_gripper = None
       
        # 位置保持相关
        self.is_holding = False
        self.hold_left_arm = None
        self.hold_right_arm = None
        self.hold_left_gripper = None
        self.hold_right_gripper = None
        self.hold_lock = threading.Lock()
        self._hold_thread_running = False
       
        # 数据接收标志
        self.data_ready = {
            'head_image': False,
            'left_hand_image': False,
            'right_hand_image': False,
            'joint_state': False,
            'left_gripper': False,
            'right_gripper': False,
        }
       
        print("📡 创建ROS2订阅...")
       
        # 订阅相机
        self.create_subscription(Image, '/camera/head_color', self._head_callback, 70)
        self.create_subscription(Image, '/camera/hand_left_color', self._left_hand_callback, 70)
        self.create_subscription(Image, '/camera/hand_right_color', self._right_hand_callback, 70)
       
        # 订阅关节状态
        self.create_subscription(JointState, '/hal/arm_joint_state', self._joint_callback, 70)
       
        # 订阅夹爪状态
        if EndState is not None:
            self.create_subscription(EndState, '/hal/left_ee_data', self._left_gripper_callback, 70)
            self.create_subscription(EndState, '/hal/right_ee_data', self._right_gripper_callback, 70)
            print("✅ 夹爪订阅已创建 (必需)")
       
        # 控制发布者
        if self.enable_control:
            self.arm_pub = self.create_publisher(JointState, '/wbc/arm_command', 10)
            self.left_gripper_pub = self.create_publisher(JointState, '/wbc/left_ee_command', 10)
            self.right_gripper_pub = self.create_publisher(JointState, '/wbc/right_ee_command', 10)
            print("✅ 控制发布者已创建")
            
            # 启动后台保持线程
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
        """后台线程：以20Hz持续发送保持指令"""
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
   
    def is_data_ready(self):
        return True
        return all(self.data_ready.values())
   
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
   
    def get_observation(self, task_description="Pick up the banana from the fruit basket on the right side of the table and place it into the box on the left"):
        if not self.is_data_ready():
            return None
       
        # 处理图像
        video_obs = {}
        for key in ['top_head', 'hand_left', 'hand_right']:
            img = self.latest_images[key]
            resized = cv2.resize(img, (640, 480))
            correct_key = f"observation.images.{key}"
            video_obs[correct_key] = resized[np.newaxis, np.newaxis, ...]
       
        # 处理关节状态
        joints = np.array(self.latest_joint_state, dtype=np.float32)
        left_arm = joints[:7]
        right_arm = joints[7:14]
       
        state_obs = {
            'left_arm_joint_position': left_arm[np.newaxis, np.newaxis, :],
            'right_arm_joint_position': right_arm[np.newaxis, np.newaxis, :],
            'left_effector_position': np.array([[[self.latest_left_gripper]]], dtype=np.float32),
            'right_effector_position': np.array([[[self.latest_right_gripper]]], dtype=np.float32),
        }
       
        language_obs = {
            'annotation.human.task_description': [[task_description]]
        }
       
        return {
            'video': video_obs,
            'state': state_obs,
            'language': language_obs,
        }
   
    def execute_action_step(self, left_arm, right_arm, left_gripper, right_gripper):
        """执行单步动作"""
        if not self.enable_control:
            return
       
        # 1. 发布手臂控制
        arm_msg = JointState()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.position = left_arm.tolist() + right_arm.tolist()
        self.arm_pub.publish(arm_msg)
       
        # 2. 发布左夹爪控制
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = self.get_clock().now().to_msg()
        left_gripper_msg.name = ['left_gripper_joint1']
        left_gripper_msg.position = [float(left_gripper)]
        self.left_gripper_pub.publish(left_gripper_msg)
       
        # 3. 发布右夹爪控制
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
        """
        执行 Action Chunk + 统一插值策略
        
        关键改进：
        1. Chunk间过渡插值：使用固定的 transition_interpolation_factor
        2. Chunk内插值：使用固定的 interpolation_factor
        3. 两种插值都使用五次样条，保持一致性
        
        Args:
            action: GR00T 输出的完整动作 (32步)
            execution_horizon: 实际执行的原始步数 (推荐8步)
            control_freq: 控制频率 (Hz)
            interpolation_factor: Chunk内插值倍数 (推荐5倍)
            transition_interpolation_factor: Chunk间过渡插值倍数 (推荐5倍)
        """
        if not self.enable_control:
            return

        # ⭐ 步骤1: 停止保持模式
        with self.hold_lock:
            self.is_holding = False
        
        print(f"\n🎬 开始执行 Action Chunk...")
        print(f"   执行前{execution_horizon}步, Chunk内{interpolation_factor}倍插值, Chunk间{transition_interpolation_factor}倍插值")

        dt = 1.0 / control_freq

        # === 步骤2: 提取原始数据（只取前 execution_horizon 步）===
        left_arm_orig = action['left_arm_joint_position'][0][:execution_horizon]      # (8, 7)
        right_arm_orig = action['right_arm_joint_position'][0][:execution_horizon]    # (8, 7)
        left_gripper_orig = action['left_effector_position'][0, :execution_horizon, 0]   # (8,)
        right_gripper_orig = action['right_effector_position'][0, :execution_horizon, 0] # (8,)

        print(f"\n  📥 提取前 {execution_horizon} 步原始动作")
        print(f"     左臂J0: {left_arm_orig[0, 0]:+.4f} → {left_arm_orig[-1, 0]:+.4f}")
        print(f"     右臂J0: {right_arm_orig[0, 0]:+.4f} → {right_arm_orig[-1, 0]:+.4f}")

        # ⭐⭐⭐ 步骤3: 跨Chunk过渡插值（统一策略）⭐⭐⭐
        if (self.last_chunk_left_arm is not None and 
            self.last_chunk_right_arm is not None):
            
            # 计算距离（用于显示）
            dist_left = np.linalg.norm(left_arm_orig[0] - self.last_chunk_left_arm)
            dist_right = np.linalg.norm(right_arm_orig[0] - self.last_chunk_right_arm)
            
            print(f"\n  🔗 跨Chunk过渡:")
            print(f"     上个Chunk最后: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")
            print(f"     新Chunk第一步:  左J0={left_arm_orig[0, 0]:+.4f}, 右J0={right_arm_orig[0, 0]:+.4f}")
            print(f"     跳变距离: 左={dist_left:.4f} rad, 右={dist_right:.4f} rad")
            
            # ⭐ 构建2点轨迹：[上个Chunk最后, 新Chunk第一步]
            left_transition_traj = np.vstack([
                self.last_chunk_left_arm,    # (7,)
                left_arm_orig[0]             # (7,)
            ])  # (2, 7)
            
            right_transition_traj = np.vstack([
                self.last_chunk_right_arm,
                right_arm_orig[0]
            ])  # (2, 7)
            
            # ⭐ 使用五次样条插值（与Chunk内相同）
            t_orig = np.linspace(0, 1, 2)  # 2个原始点
            t_new = np.linspace(0, 1, transition_interpolation_factor + 1)  # 插值后N+1个点
            
            def interpolate_quintic(orig_traj, t_orig, t_new):
                """五次样条插值"""
                N, D = orig_traj.shape
                interp_traj = np.zeros((len(t_new), D))
                for d in range(D):
                    # 对于只有2个点的情况，使用3次样条（5次需要至少4个点）
                    if N == 2:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=1)  # 线性插值
                    else:
                        spl = make_interp_spline(t_orig, orig_traj[:, d], k=min(5, N-1))
                    interp_traj[:, d] = spl(t_new)
                return interp_traj
            
            left_transition_interp = interpolate_quintic(left_transition_traj, t_orig, t_new)
            right_transition_interp = interpolate_quintic(right_transition_traj, t_orig, t_new)
            
            # 夹爪：线性插值
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
            
            num_transition_steps = len(t_new) - 1  # 去掉最后一个点（会在Chunk内执行）
            
            print(f"  🔀 执行 {num_transition_steps} 步跨Chunk过渡插值 ({transition_interpolation_factor}倍)")
            
            # 执行过渡动作（不包括最后一步，因为它等于新Chunk的第一步）
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

        # === 步骤4: Chunk内五次样条插值 ===
        original_num_steps = execution_horizon
        new_num_steps = execution_horizon * interpolation_factor
        
        t_orig = np.linspace(0, 1, original_num_steps)
        t_new = np.linspace(0, 1, new_num_steps)

        def interpolate_quintic_chunk(orig_traj, t_orig, t_new):
            """Chunk内五次样条插值"""
            N, D = orig_traj.shape
            interp_traj = np.zeros((len(t_new), D))
            for d in range(D):
                spl = make_interp_spline(t_orig, orig_traj[:, d], k=5)
                interp_traj[:, d] = spl(t_new)
            return interp_traj

        left_arm_interp = interpolate_quintic_chunk(left_arm_orig, t_orig, t_new)
        right_arm_interp = interpolate_quintic_chunk(right_arm_orig, t_orig, t_new)

        # 夹爪：线性插值
        left_gripper_interp = np.interp(t_new, t_orig, left_gripper_orig)
        right_gripper_interp = np.interp(t_new, t_orig, right_gripper_orig)

        print(f"\n  🔧 Chunk内插值: {original_num_steps}步 → {new_num_steps}步 ({interpolation_factor}倍)")

        # === 步骤5: 执行插值后的主动作序列 ===
        print(f"  ▶️  执行 {new_num_steps} 步Chunk内插值动作 (频率: {control_freq}Hz, 耗时: {new_num_steps/control_freq:.2f}秒)")

        for step in range(new_num_steps):
            left_arm = left_arm_interp[step]
            right_arm = right_arm_interp[step]
            left_gripper = left_gripper_interp[step]
            right_gripper = right_gripper_interp[step]

            self.execute_action_step(left_arm, right_arm, left_gripper, right_gripper)

            # 显示进度
            if step % 10 == 0 or step == new_num_steps - 1:
                progress = (step + 1) / new_num_steps * 100
                print(f"    Chunk步 {step+1:3d}/{new_num_steps} ({progress:5.1f}%): "
                      f"左J0={left_arm[0]:+.4f}, 右J0={right_arm[0]:+.4f}")

            time.sleep(dt)
            rclpy.spin_once(self, timeout_sec=0.001)

        print(f"  ✅ Action Chunk 执行完成")

        # ⭐⭐⭐ 步骤6: 保存本次Chunk的**原始**最后一步 ⭐⭐⭐
        self.last_chunk_left_arm = np.array(left_arm_orig[-1])      # 原始第8步
        self.last_chunk_right_arm = np.array(right_arm_orig[-1])
        self.last_chunk_left_gripper = float(left_gripper_orig[-1])
        self.last_chunk_right_gripper = float(right_gripper_orig[-1])

        print(f"  💾 保存本次Chunk最后位置: 左J0={self.last_chunk_left_arm[0]:+.4f}, 右J0={self.last_chunk_right_arm[0]:+.4f}")

        # === 步骤7: 启动保持模式 ===
        with self.hold_lock:
            self.hold_left_arm = left_arm_interp[-1].copy()
            self.hold_right_arm = right_arm_interp[-1].copy()
            self.hold_left_gripper = left_gripper_interp[-1]
            self.hold_right_gripper = right_gripper_interp[-1]
            self.is_holding = True
        
        print(f"  🔒 已切换到保持模式（后台线程接管）")
        
        # 刷新状态
        for i in range(5):
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)
    
    def shutdown(self):
        """清理资源"""
        print("\n🛑 正在关闭节点...")
        
        # 停止保持线程
        self._hold_thread_running = False
        if hasattr(self, 'hold_thread'):
            self.hold_thread.join(timeout=2.0)
        
        print("✅ 节点已关闭")


def wait_for_data(node, timeout=30):
    """等待所有必需数据就绪"""
    print(f"\n⏳ 等待所有必需数据 (超时: {timeout}秒)...")
    print("   必需: 3个图像 + 关节状态 + 2个夹爪状态")
   
    start_time = time.time()
    last_print_time = 0
   
    while time.time() - start_time < timeout:
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
    """保存观测图像"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    saved_files = []
    for key in ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right']:
        if key in obs['video']:
            img = obs['video'][key][0, 0, :, :, :]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            camera_name = key.split('.')[-1]
            filename = f"{save_dir}/{timestamp}_{camera_name}.png"
            cv2.imwrite(filename, img_bgr)
            saved_files.append(filename)
   
    return saved_files


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 AGIBOT G1 - Action Chunking + 统一插值策略")
    print("=" * 80)
   
    # ⭐⭐⭐ 控制参数（统一插值策略）⭐⭐⭐
    ENABLE_CONTROL = True
    CONTROL_FREQ = 20                      # 20Hz控制频率
    EXECUTION_HORIZON = 8                  # 每次执行8步（GR00T推荐）
    INTERPOLATION_FACTOR = 5               # Chunk内插值倍数（8步 → 40步）
    TRANSITION_INTERPOLATION_FACTOR = 5    # Chunk间过渡插值倍数（2步 → 6步）
    ACTION_HORIZON = 32                    # 模型预测32步
   
    # 计算总执行时间
    transition_steps = TRANSITION_INTERPOLATION_FACTOR  # 不包括最后一步
    chunk_steps = EXECUTION_HORIZON * INTERPOLATION_FACTOR
    total_steps_per_cycle = transition_steps + chunk_steps
    cycle_time = total_steps_per_cycle / CONTROL_FREQ
    
    if ENABLE_CONTROL:
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
   
    # 初始化ROS2
    print("\n[1/3] 初始化ROS2...")
    rclpy.init()
    node = RobotObservationNode(enable_control=ENABLE_CONTROL)
   
    # 连接推理服务器
    print("\n[2/3] 连接推理服务器...")
    try:
        client = PolicyClient(host="127.0.0.1", port=5555)
        print("✅ 连接成功!")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        node.shutdown()
        rclpy.shutdown()
        return
   
    # 等待数据就绪
    print("\n[3/3] 等待所有必需数据...")
    if not wait_for_data(node, timeout=30):
        print("\n💡 调试提示:")
        print("   1. 检查机器人是否启动")
        print("   2. 检查copilot模式")
        node.shutdown()
        rclpy.shutdown()
        return
   
    # 显示初始状态
    obs = node.get_observation()
    if obs:
        print("\n📊 初始状态:")
        print(f"   左臂J0: {obs['state']['left_arm_joint_position'][0,0,0]:.4f} rad")
        print(f"   右臂J0: {obs['state']['right_arm_joint_position'][0,0,0]:.4f} rad")
        save_observation_images(obs)
   
    # 推理循环
    print("\n" + "=" * 80)
    print("🎯 开始 Action Chunking + 统一插值 循环")
    print(f"   流程: 推理 → 跨Chunk过渡({TRANSITION_INTERPOLATION_FACTOR}倍) → Chunk内执行({INTERPOLATION_FACTOR}倍) → 保持 → 推理")
    print("   特性: 统一插值策略，可独立调节Chunk间和Chunk内插值倍数")
    print("   按 Ctrl+C 退出")
    print("=" * 80)
   
    try:
        count = 0
        total_steps_executed = 0
       
        while rclpy.ok():
            cycle_start = time.time()
           
            # 1. 获取观测
            rclpy.spin_once(node, timeout_sec=0.001)
            obs = node.get_observation()
           
            if obs is None:
                time.sleep(0.1)
                continue
           
            count += 1
           
            print(f"\n\n{'#'*80}")
            print(f"# Action Chunking 循环 #{count}")
            print(f"{'#'*80}")
           
            # 记录推理前的状态
            pre_inference_joints = np.array(node.latest_joint_state)
            print(f"\n📍 当前状态:")
            print(f"   左臂J0: {pre_inference_joints[0]:+.4f} rad")
            print(f"   右臂J0: {pre_inference_joints[7]:+.4f} rad")
            print(f"   已执行总步数: {total_steps_executed}")
            
            # 显示保持状态
            with node.hold_lock:
                hold_status = "🔒 保持中" if node.is_holding else "⏸️  空闲"
            print(f"   保持状态: {hold_status}")
           
            # 2. 推理
            print(f"\n🧠 正在推理 (预测{ACTION_HORIZON}步)...")
            inference_start = time.time()
            action, _ = client.get_action(obs)
            inference_time = time.time() - inference_start
           
            print(f"✅ 推理完成 (耗时: {inference_time*1000:.1f}ms)")
           
            # 显示目标位置
            target_left_j0 = action['left_arm_joint_position'][0, EXECUTION_HORIZON-1, 0]
            target_right_j0 = action['right_arm_joint_position'][0, EXECUTION_HORIZON-1, 0]
            print(f"\n🎯 本轮目标 (原始第{EXECUTION_HORIZON}步):")
            print(f"   左臂J0: {target_left_j0:+.4f} rad")
            print(f"   右臂J0: {target_right_j0:+.4f} rad")
           
            # 3. 执行 Action Chunk + 统一插值
            if ENABLE_CONTROL:
                node.execute_action_chunk_with_interpolation(
                    action, 
                    execution_horizon=EXECUTION_HORIZON,
                    control_freq=CONTROL_FREQ,
                    interpolation_factor=INTERPOLATION_FACTOR,
                    transition_interpolation_factor=TRANSITION_INTERPOLATION_FACTOR
                )
                
                # 统计（只统计第一轮后的完整循环）
                if count > 1:
                    total_steps_executed += transition_steps + chunk_steps
                else:
                    total_steps_executed += chunk_steps  # 第一轮没有过渡
               
                # 验证执行后位置
                rclpy.spin_once(node, timeout_sec=0.01)
                final_joints = np.array(node.latest_joint_state)
               
                print(f"\n📊 执行结果:")
                print(f"   目标: 左J0={target_left_j0:+.4f}, 右J0={target_right_j0:+.4f}")
                print(f"   实际: 左J0={final_joints[0]:+.4f}, 右J0={final_joints[7]:+.4f}")
                print(f"   误差: 左J0={abs(final_joints[0]-target_left_j0):.4f}, "
                      f"右J0={abs(final_joints[7]-target_right_j0):.4f} rad")
            else:
                print("  ⏭️  跳过执行 (控制未启用)")
           
            # 统计
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
        rclpy.shutdown()
        print("🔌 已关闭")


if __name__ == "__main__":
    main()
