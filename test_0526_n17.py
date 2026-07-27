"""
AGIBOT G1 GR00T N1.7 v6 (正确异步: wait到位后拍观测)
正确异步时机:
  执行完 → wait到位(精确位置) → 拍观测(精确位置) → 推理(轨迹从精确位置出发)
  → 推理后台跑 → blend+执行下一轮 → 不阻塞 → 不回弹 ✅
"""

import numpy as np
import cv2
import msgpack
import msgpack_numpy as mnp
import rclpy
import sys
import zmq
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from genie_msgs.msg import EndState
except ImportError:
    EndState = None

REQUIRED_KEYS = ("joint_position", "left_effector_position", "right_effector_position")
IMG_SIZE = (640, 480)


def key_tail(key):
    return str(key).split(".")[-1].lower()

def canonical_key(key):
    lk = str(key).lower()
    tail = key_tail(lk)
    if tail == "joint_position" or lk.endswith(".joint_position"):
        return "joint_position"
    if tail == "left_effector_position" or lk.endswith(".left_effector_position"):
        return "left_effector_position"
    if tail == "right_effector_position" or lk.endswith(".right_effector_position"):
        return "right_effector_position"
    return str(key)

def normalize_action(action):
    norm = {}
    for k, v in action.items():
        nk = k[len("action."):] if k.startswith("action.") else k
        norm[canonical_key(nk)] = np.asarray(v, dtype=np.float32)
    if "joint_position" not in norm:
        if "left_arm_joint_position" in norm and "right_arm_joint_position" in norm:
            norm["joint_position"] = np.concatenate(
                [norm["left_arm_joint_position"], norm["right_arm_joint_position"]], axis=-1)
        else:
            raise KeyError(f"缺少 joint_position, keys: {list(norm.keys())}")
    if norm["joint_position"].shape[-1] != 14:
        raise ValueError(f"joint_position 需要14维, 实际 {norm['joint_position'].shape}")
    for k in REQUIRED_KEYS:
        if k not in norm: raise KeyError(f"缺少 {k}")
    return norm


def bezier_chunk(chunk, keys, substeps, ei=0.3, eo=0.3):
    out = {}
    for key in keys:
        if key not in chunk: continue
        arr = chunk[key][0]; H, D = arr.shape
        total = H * substeps
        res = np.zeros((total, D), dtype=arr.dtype)
        for i in range(H - 1):
            p0, p1 = arr[i], arr[i+1]
            d = p1 - p0; c0 = p0 + d * ei; c1 = p1 - d * eo
            idx = i * substeps
            for j in range(substeps):
                t = j / substeps; u = 1 - t
                res[idx + j] = u*u*u*p0 + 3*u*u*t*c0 + 3*u*t*t*c1 + t*t*t*p1
        res[(H-1)*substeps:] = arr[-1]
        out[key] = res[np.newaxis, ...]
    return out


class EMAFilter:
    def __init__(self, dim, alpha=0.3):
        self.alpha = alpha; self.v = np.zeros(dim, dtype=np.float32); self.ok = False
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if not self.ok: self.v = x.copy(); self.ok = True; return self.v
        self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v


class PolicyClient:
    def __init__(self, host="127.0.0.1", port=5555):
        self.context = zmq.Context(); self.host = host; self.port = port
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 20000)
        self.socket.setsockopt(zmq.SNDTIMEO, 20000)
        self.socket.setsockopt(zmq.SNDBUF, 65536)
        self.socket.setsockopt(zmq.RCVBUF, 65536)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def get_action(self, obs, max_steps=8):
        payload = msgpack.packb(
            {"endpoint": "get_action", "data": {"observation": obs, "options": None}},
            default=mnp.encode, use_bin_type=True)
        self.socket.send(payload); raw = self.socket.recv()
        if raw == b"ERROR": raise RuntimeError("Server error")
        resp = msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp: raise RuntimeError(f"Server error: {resp['error']}")
        action, _ = resp
        action = normalize_action(action)
        # 截断到 max_steps 步（模型默认可能输出 16/32 步）
        for k in REQUIRED_KEYS:
            if action[k].shape[1] > max_steps:
                action[k] = action[k][:, :max_steps, :]
        return action


class RobotNode(Node):
    def __init__(self, enable_control=False):
        super().__init__("robot_n17_v6")
        self.bridge = CvBridge(); self.enable_control = enable_control
        self.latest_images = {}
        self.latest_joint_state = np.zeros(14, dtype=np.float32)
        self.latest_left_gripper = 0.0; self.latest_right_gripper = 0.0
        self._ready = {"head": False, "joint": False}
        qos = 1
        self.create_subscription(Image, "/camera/head_color", self._cb_head, qos)
        self.create_subscription(Image, "/camera/hand_left_color", self._cb_left, qos)
        self.create_subscription(Image, "/camera/hand_right_color", self._cb_right, qos)
        self.create_subscription(JointState, "/hal/arm_joint_state", self._cb_joint, qos)
        if EndState is not None:
            self.create_subscription(EndState, "/hal/left_ee_data", self._cb_gl, qos)
            self.create_subscription(EndState, "/hal/right_ee_data", self._cb_gr, qos)
        if self.enable_control:
            self.arm_pub = self.create_publisher(JointState, "/wbc/arm_command", qos)
            self.lg_pub = self.create_publisher(JointState, "/wbc/left_ee_command", qos)
            self.rg_pub = self.create_publisher(JointState, "/wbc/right_ee_command", qos)
            self._arm_msg = JointState(); self._lg_msg = JointState(); self._rg_msg = JointState()
            self._lg_msg.name = ["left_gripper_joint1"]; self._rg_msg.name = ["right_gripper_joint1"]
        time.sleep(1.5)

    def _decode_img(self, msg):
        try: return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except:
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except: return None

    def _cb_head(self, msg):
        img = self._decode_img(msg)
        if img is not None: self.latest_images["top_head"] = img; self._ready["head"] = True
    def _cb_left(self, msg):
        img = self._decode_img(msg)
        if img is not None: self.latest_images["hand_left"] = img
    def _cb_right(self, msg):
        img = self._decode_img(msg)
        if img is not None: self.latest_images["hand_right"] = img
    def _cb_joint(self, msg):
        if len(msg.position) >= 14:
            self.latest_joint_state = np.array(msg.position[:14], dtype=np.float32)
            self._ready["joint"] = True
    def _cb_gl(self, msg):
        if len(msg.end_state) > 0: self.latest_left_gripper = float(msg.end_state[0].position)
    def _cb_gr(self, msg):
        if len(msg.end_state) > 0: self.latest_right_gripper = float(msg.end_state[0].position)

    @property
    def ready(self): return self._ready["head"] and self._ready["joint"]

    def get_obs(self, task="Grasp the thermometer gun"):
        if not self.ready: return None
        video = {}
        for k in ("top_head", "hand_left", "hand_right"):
            if k not in self.latest_images: return None
            img = cv2.resize(self.latest_images[k], IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            video[f"observation.images.{k}"] = img[np.newaxis, np.newaxis, ...]
        j = self.latest_joint_state
        state = {
            "left_arm_joint_position": j[:7][np.newaxis, np.newaxis, :],
            "right_arm_joint_position": j[7:14][np.newaxis, np.newaxis, :],
            "left_effector_position": np.array([[[self.latest_left_gripper]]], dtype=np.float32),
            "right_effector_position": np.array([[[self.latest_right_gripper]]], dtype=np.float32),
        }
        lang = {"annotation.human.task_description": [[task]]}
        return {"video": video, "state": state, "language": lang}

    def execute_step(self, jp, lg, rg):
        if not self.enable_control: return
        stamp = self.get_clock().now().to_msg()
        self._arm_msg.header.stamp = stamp; self._arm_msg.position = jp.tolist()
        self.arm_pub.publish(self._arm_msg)
        self._lg_msg.header.stamp = stamp; self._lg_msg.position = [float(lg)]
        self.lg_pub.publish(self._lg_msg)
        self._rg_msg.header.stamp = stamp; self._rg_msg.position = [float(rg)]
        self.rg_pub.publish(self._rg_msg)


# ========== 后台推理 ==========

def infer_job(client, obs):
    try: return client.get_action(obs)
    except: return None


# ========== 主函数 ==========

def main():
    print("=" * 62)
    print("  GR00T N1.7 v6 (正确异步 | 60Hz | EMA)")
    print("  wait到位后拍观测 ")
    print("=" * 62)

    ENABLE_CONTROL = True
    CONTROL_FREQ = 60
    INTERP_FACTOR = 3
    EMA_ALPHA = 0.3
    POS_TOLERANCE = 0.03
    TASK = "Partial Discharge Detection"

    if ENABLE_CONTROL:
        print("\n[!] 控制已启用!")
        # 非交互式运行时自动跳过确认（stdin 不可用或为 DEVNULL）
        try:
            if not sys.stdin.isatty():
                print("  非交互式终端，自动跳过确认")
            elif input("  继续? (yes/no): ").lower() != "yes":
                return
        except (EOFError, OSError):
            print("  stdin 不可用，自动跳过确认")

    print("[1] ROS2..."); rclpy.init()
    node = RobotNode(enable_control=ENABLE_CONTROL)

    print("[2] 推理服务器...")
    try: client = PolicyClient(); print("  OK")
    except Exception as e: print(f"  FAIL: {e}"); rclpy.shutdown(); return

    print("[3] 传感器...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready: print(f"  就绪 ({time.perf_counter()-t0:.1f}s)"); break
    else: print("  超时!"); rclpy.shutdown(); return

    executor = ThreadPoolExecutor(max_workers=1)

    obs = node.get_obs(task=TASK)
    if obs is None: print("  观测失败"); rclpy.shutdown(); return

    j0 = node.latest_joint_state
    print(f"  当前: J0={j0[0]:+.3f} J7={j0[7]:+.3f}")

    t0 = time.perf_counter()
    current_action = client.get_action(obs)
    print(f"  首次推理: {(time.perf_counter()-t0)*1000:.0f}ms")

    node.last_joint_pos = None
    node.last_left_gripper = None; node.last_right_gripper = None

    ema = EMAFilter(14, alpha=EMA_ALPHA)
    dt = 1.0 / CONTROL_FREQ
    action_keys = ["joint_position", "left_effector_position", "right_effector_position"]

    # 预先启动第二轮推理(精确位置)
    obs2 = node.get_obs(task=TASK)
    infer_future = executor.submit(infer_job, client, obs2) if obs2 is not None else None

    print(f"\n{'='*62}")
    print(f"  主循环: blend+执行 | wait到位后拍观测+异步推理")
    print(f"{'='*62}")

    cycle = 1
    INFERENCE_START = time.perf_counter()
    gripper_open_start = None
    try:
        while rclpy.ok():
            cs = time.perf_counter()

            # 拿后台预取结果(已就绪, 不阻塞)
            if infer_future is not None:
                try:
                    r = infer_future.result(timeout=10.0)
                    if r is not None: current_action = r
                except: pass
                infer_future = None

            action = current_action
            H = action["joint_position"].shape[1]

            interp = bezier_chunk(action, action_keys, substeps=INTERP_FACTOR, ei=0.3, eo=0.3)
            total = H * INTERP_FACTOR
            traj = interp["joint_position"][0]

            # blend (从上一轮目标位置出发, 同rhc1)
            if node.last_joint_pos is not None:
                num_blend = 14
                t_vals = np.linspace(0, 1, num_blend + 2)[1:-1]
                s = t_vals * t_vals * (3 - 2 * t_vals)
                jp0 = node.last_joint_pos; jp1 = traj[0]
                for i in range(num_blend):
                    blend = (1 - s[i]) * jp0 + s[i] * jp1
                    node.execute_step(blend, node.last_left_gripper, node.last_right_gripper)
                    time.sleep(dt)
                    if i % 4 == 0: rclpy.spin_once(node, timeout_sec=0.001)

            # 执行主动作
            seq_start = time.perf_counter()
            for si in range(total):
                lg = interp["left_effector_position"][0, si, 0]
                rg = interp["right_effector_position"][0, si, 0]
                node.execute_step(ema(traj[si]), lg, rg)
                wait_until = seq_start + (si + 1) * dt
                remaining = wait_until - time.perf_counter()
                if remaining > 0.002: time.sleep(remaining)
                if si % 20 == 0: rclpy.spin_once(node, timeout_sec=0.001)

            # 更新last
            node.last_joint_pos = action["joint_position"][0, -1, :].copy()
            node.last_left_gripper = float(action["left_effector_position"][0, -1, 0])
            node.last_right_gripper = float(action["right_effector_position"][0, -1, 0])
            target_pos = action["joint_position"][0, -1, :]

            # wait到位
            tw = time.perf_counter() + 15.0
            while time.perf_counter() < tw:
                rclpy.spin_once(node, timeout_sec=0.02)
                j = node.latest_joint_state
                if j is not None:
                    err = float(np.max(np.abs(j - target_pos)))
                    if err < POS_TOLERANCE: break
                time.sleep(0.02)

            # wait到位后→拍观测(精确位置)→启动后台推理

            # 检查：推理开始10s后，右夹爪打开>3s → 认为推理结束
            elapsed = time.perf_counter() - INFERENCE_START
            if elapsed > 10.0 and node.latest_right_gripper < 10.0:
                if gripper_open_start is None:
                    gripper_open_start = time.perf_counter()
                    print(f"  🟢 右夹爪打开 (pos={node.latest_right_gripper:.1f}, elapsed={elapsed:.0f}s)")
                elif time.perf_counter() - gripper_open_start > 3.0:
                    print(f"  ✅ 右夹爪已打开超过3s (pos={node.latest_right_gripper:.1f}, elapsed={elapsed:.0f}s)")
                    print(f"  🏁 推理完成，退出")
                    break
            else:
                gripper_open_start = None

            next_obs = node.get_obs(task=TASK)
            if next_obs is not None:
                infer_future = executor.submit(infer_job, client, next_obs)

            ct = time.perf_counter() - cs
            j = node.latest_joint_state
            if cycle <= 3 or cycle % 30 == 0:
                print(f"  #{cycle:03d}: {ct*1000:.0f}ms | J0={j[0]:+.3f} J7={j[7]:+.3f}")
            cycle += 1

    except KeyboardInterrupt: print("\n  中断")
    finally:
        executor.shutdown(wait=False)
        rclpy.shutdown()
        print(f"  共 {cycle-1} 个推理周期")


if __name__ == "__main__":
    main()

