import time
import cv2
import json_numpy
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

json_numpy.patch()

def decode_image_msg(msg: Image) -> np.ndarray:
    """
    把 sensor_msgs.msg.Image 解码成 HxWx3 的 uint8 RGB numpy 数组。
    支持常见 encoding: 'rgb8','bgr8','mono8','rgba8'（会丢弃 alpha）。
    """
    h = msg.height
    w = msg.width
    step = msg.step  # 字节/行
    data = np.frombuffer(msg.data, dtype=np.uint8)
    # 有些实现 data 长度为 h*step
    if data.size != h * step:
        # 尝试容错：直接按 (h, w, channels) reshape（常见情况）
        try:
            channels = data.size // (h * w)
            if channels >= 1:
                return data.reshape((h, w, channels))[:, :, :3].copy()
        except Exception:
            pass
        raise ValueError(f"Unexpected image buffer size {data.size}, expected {h*step}")
    # reshape 到 (h, step) 再截取前 w*channels 字节
    row = data.reshape((h, step))
    channels = step // w
    arr = row[:, : w * channels].reshape((h, w, channels))
    enc = (msg.encoding or "").lower()
    if enc == "bgr8" and channels >= 3:
        arr = arr[:, :, :3][..., ::-1]  # BGR->RGB
    elif enc == "rgb8" and channels >= 3:
        arr = arr[:, :, :3]
    elif enc == "mono8" or channels == 1:
        gray = arr[:, :, 0]
        arr = np.stack([gray, gray, gray], axis=2)
    elif enc == "rgba8" and channels >= 4:
        arr = arr[:, :, :3]
    else:
        # 其它编码尽量取前三通道
        if channels >= 3:
            arr = arr[:, :, :3]
        else:
            # 最后兜底：复制通道到3通道
            gray = arr[:, :, 0]
            arr = np.stack([gray, gray, gray], axis=2)
    return arr.copy()  # 确保是可写的连续数组

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.images = {'head': None, 'hand_left': None, 'hand_right': None}
        self.create_subscription(Image, '/camera/head_color', lambda msg: self._callback(msg, 'head'), 10)
        self.create_subscription(Image, '/camera/hand_left_color', lambda msg: self._callback(msg, 'hand_left'), 10)
        self.create_subscription(Image, '/camera/hand_right_color', lambda msg: self._callback(msg, 'hand_right'), 10)

    def _callback(self, msg: Image, name: str):
        try:
            img = decode_image_msg(msg)  # HxWx3 RGB
            img = cv2.resize(img, (848, 480))  
            self.images[name] = img
        except Exception as e:
            # 简短日志，不中断节点
            self.get_logger().warning(f"decode {name} failed: {e}")

    def get_images(self):
        return self.images

def main():
    rclpy.init()
    node = CameraSubscriber()
    # 等待第一帧到达
    while any(v is None for v in node.images.values()):
        rclpy.spin_once(node, timeout_sec=0.1)
    # 主循环：持续发送带有最新三路相机图的请求（状态暂用随机）
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            imgs = node.get_images()
            obs = {
                "video.top_head": imgs['head'][np.newaxis, ...].astype(np.uint8),
                "video.hand_left": imgs['hand_left'][np.newaxis, ...].astype(np.uint8),
                "video.hand_right": imgs['hand_right'][np.newaxis, ...].astype(np.uint8),
                # 状态部分仍用随机（下一步替换真实状态）
                "state.left_arm_joint_position": np.random.rand(1, 7),
                "state.right_arm_joint_position": np.random.rand(1, 7),
                "state.left_effector_position": np.random.rand(1, 1),
                "state.right_effector_position": np.random.rand(1, 1),
                "state.head_position": np.random.rand(1, 2),
                "state.waist_position": np.random.rand(1, 2),
                "annotation.language.action_text": ["Pick up the object!"],
            }
            t = time.time()
            response = requests.post("http://127.0.0.1:6789/act", json={"observation": obs})
            print(f"used time {time.time() - t}")
            print(response.json())
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
