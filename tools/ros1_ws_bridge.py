#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS1 → WebSocket 桥接脚本 (运行在机器人电脑 / Docker 容器内).

订阅 master 臂的 PosCmd topic，通过 WebSocket 向仿真电脑实时推送
末端执行器状态，供 IsaacLab-Arena 遥操作使用。

依赖:
    pip install websockets   # (在 ROS1 / Docker 环境中安装)

使用:
    # 默认参数 (端口 5555, 50Hz 推送)
    python ros1_ws_bridge.py

    # 自定义参数
    python ros1_ws_bridge.py --port 8765 --hz 100

    # Docker 运行时需要映射端口:
    # docker run -p 5555:5555 ...

Topic 约定:
    /master1_pos_back  ->  左臂 (arm_control/PosCmd)
    /master2_pos_back  ->  右臂 (arm_control/PosCmd)
"""

from __future__ import print_function

import argparse
import asyncio
import json
import threading
import time

import rospy
from arm_control.msg import PosCmd


# ---------------------------------------------------------------------------
# Shared state: ROS callbacks write, WebSocket server reads
# ---------------------------------------------------------------------------
class ArmState:
    """Thread-safe container for the latest arm states from ROS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._left = None   # type: dict | None
        self._right = None  # type: dict | None

    def update_left(self, msg):
        with self._lock:
            self._left = self._msg_to_dict(msg)

    def update_right(self, msg):
        with self._lock:
            self._right = self._msg_to_dict(msg)

    def snapshot(self):
        """Return the latest state as a JSON-serialisable dict."""
        with self._lock:
            return {
                "timestamp": time.time(),
                "left": self._left,
                "right": self._right,
            }

    @staticmethod
    def _msg_to_dict(msg):
        return {
            "x": msg.x,
            "y": msg.y,
            "z": msg.z,
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": msg.yaw,
            "gripper": msg.gripper,
            "mode1": msg.mode1,
            "mode2": msg.mode2,
        }


# ---------------------------------------------------------------------------
# ROS1 subscriber (runs in a background thread)
# ---------------------------------------------------------------------------
def ros_thread(state, left_topic, right_topic):
    """Initialise ROS node and spin in a daemon thread."""
    rospy.init_node("ws_bridge", anonymous=True, disable_signals=True)

    rospy.Subscriber(left_topic, PosCmd, state.update_left, queue_size=1)
    rospy.Subscriber(right_topic, PosCmd, state.update_right, queue_size=1)

    rospy.loginfo(
        "[ws_bridge] Subscribing: left=%s  right=%s", left_topic, right_topic
    )
    rospy.spin()


# ---------------------------------------------------------------------------
# WebSocket server (main thread, asyncio)
# ---------------------------------------------------------------------------
async def ws_handler(websocket, state, hz):
    """Push arm state to a single connected client at *hz* frequency."""
    interval = 1.0 / hz
    peer = websocket.remote_address
    print(f"[ws_bridge] Client connected: {peer}")
    try:
        while True:
            payload = json.dumps(state.snapshot())
            await websocket.send(payload)
            await asyncio.sleep(interval)
    except Exception:
        print(f"[ws_bridge] Client disconnected: {peer}")


async def serve(state, port, hz):
    import websockets

    handler = lambda ws, path=None: ws_handler(ws, state, hz)
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"[ws_bridge] WebSocket server listening on 0.0.0.0:{port}  ({hz} Hz)")
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ROS1 PosCmd -> WebSocket bridge for remote teleoperation"
    )
    parser.add_argument("--port", type=int, default=5555, help="WebSocket port (default: 5555)")
    parser.add_argument("--hz", type=int, default=50, help="Push frequency in Hz (default: 50)")
    parser.add_argument(
        "--left_topic", type=str, default="/master1_pos_back",
        help="Left arm PosCmd topic (default: /master1_pos_back)",
    )
    parser.add_argument(
        "--right_topic", type=str, default="/master2_pos_back",
        help="Right arm PosCmd topic (default: /master2_pos_back)",
    )
    args = parser.parse_args()

    state = ArmState()

    # Start ROS in a daemon thread
    t = threading.Thread(
        target=ros_thread,
        args=(state, args.left_topic, args.right_topic),
        daemon=True,
    )
    t.start()

    # Give ROS a moment to initialise
    time.sleep(0.5)

    # Run WebSocket server in main thread
    try:
        asyncio.run(serve(state, args.port, args.hz))
    except KeyboardInterrupt:
        print("\n[ws_bridge] Shutting down.")


if __name__ == "__main__":
    main()
