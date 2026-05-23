"""
Create: 2026.05.24
Author: SG.SUH
Python: 3.8.10
PyTorch: 1.14.0
"""

import threading

import cv2
import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64

from .petr_launcher import PETR_class


REQUEST_TIMEOUT_TICKS = 10  # 0.5s timer * 10 = 5s before re-requesting a shot


class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        self.publisher_ = self.create_publisher(Int64, "reqshot", 10)
        self.subscription = self.create_subscription(
            Image, "image", self.request_handler, 10
        )
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.petr = PETR_class()

        self._image_lock = threading.Lock()
        self._pending_image = None
        self._image_event = threading.Event()
        self._stop_event = threading.Event()

        self._req_pending = False
        self._req_fail_cnt = 0

        self._worker = threading.Thread(target=self._petr_worker, daemon=True)
        self._worker.start()

    def _petr_worker(self):
        while not self._stop_event.is_set():
            if not self._image_event.wait(timeout=0.5):
                continue
            self._image_event.clear()
            with self._image_lock:
                image, self._pending_image = self._pending_image, None
            if image is None:
                continue
            try:
                self.petr.detect_object(image)
            except Exception:
                self.get_logger().exception("PETR detection failed")

    def request_handler(self, msg):
        self._req_pending = False
        self.get_logger().info(
            f"got image {msg.width}x{msg.height} "
            f"(ts: {msg.header.stamp.sec}.{msg.header.stamp.nanosec})"
        )

        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}")
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        # PETR expects RGB. np.frombuffer is read-only, so make a writable copy.
        if msg.encoding == "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:  # rgb8
            frame = frame.copy()

        with self._image_lock:
            self._pending_image = frame
        self._image_event.set()

        # Renders the latest detection result (may lag the new frame by ~1 cycle).
        buf = self.petr.draw_picture(frame)
        annotated = cv2.cvtColor(np.array(PILImage.open(buf)), cv2.COLOR_RGB2BGR)
        cv2.imshow("result", annotated)
        cv2.waitKey(1)

    def timer_callback(self):
        if not self._req_pending:
            msg = Int64()
            msg.data = 0
            self.publisher_.publish(msg)
            self.get_logger().info(f"requesting shot: {msg.data}")
            self._req_pending = True
            return

        self._req_fail_cnt += 1
        if self._req_fail_cnt > REQUEST_TIMEOUT_TICKS:
            self._req_fail_cnt = 0
            self._req_pending = False

    def shutdown(self):
        self._stop_event.set()
        self._image_event.set()
        self._worker.join(timeout=2.0)
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
