#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS1 -> WebSocket bridge (runs on the robot PC / Docker container).

Subscribes to master arm ROS1 topics and pushes state to the simulation PC
via WebSocket for real-time teleoperation in IsaacLab-Arena.

Supports three modes:
    ee    -- End-effector only  (PosCmd topics)
    joint -- Joint position only (JointInformation topics)
    both  -- Both EE and joint data in one payload

Dependencies:
    pip install websockets   # (inside the ROS1 / Docker environment)

Usage:
    # EE mode (default), 200 Hz
    python ros1_ws_bridge.py

    # Joint mode
    python ros1_ws_bridge.py --mode joint

    # Both, custom port
    python ros1_ws_bridge.py --mode both --port 8765

Topic conventions (defaults):
    /master1_pos_back   -> left arm  EE   (arm_control/PosCmd)
    /master2_pos_back   -> right arm EE   (arm_control/PosCmd)
    /joint_information  -> left arm  joints (arm_control/JointInformation)
    /joint_information2 -> right arm joints (arm_control/JointInformation)
"""

from __future__ import print_function

import argparse
import asyncio
import json
import threading
import time

import rospy


# ---------------------------------------------------------------------------
# Shared state: ROS callbacks write, WebSocket server reads
# ---------------------------------------------------------------------------
class ArmState:
    """Thread-safe container for the latest arm states from ROS."""

    def __init__(self):
        self._lock = threading.Lock()
        # EE data (from PosCmd)
        self._left_ee = None   # type: dict | None
        self._right_ee = None  # type: dict | None
        # Joint data (from JointInformation)
        self._left_joint = None   # type: dict | None
        self._right_joint = None  # type: dict | None

    # -- EE callbacks --
    def update_left_ee(self, msg):
        with self._lock:
            self._left_ee = self._poscmd_to_dict(msg)

    def update_right_ee(self, msg):
        with self._lock:
            self._right_ee = self._poscmd_to_dict(msg)

    # -- Joint callbacks --
    def update_left_joint(self, msg):
        with self._lock:
            self._left_joint = self._jointinfo_to_dict(msg)

    def update_right_joint(self, msg):
        with self._lock:
            self._right_joint = self._jointinfo_to_dict(msg)

    # -- Snapshot builders --
    def snapshot(self, mode):
        """Return the latest state as a JSON-serialisable dict."""
        with self._lock:
            left = {}
            right = {}

            if mode in ("ee", "both") and self._left_ee is not None:
                left.update(self._left_ee)
            if mode in ("ee", "both") and self._right_ee is not None:
                right.update(self._right_ee)

            if mode in ("joint", "both") and self._left_joint is not None:
                left.update(self._left_joint)
            if mode in ("joint", "both") and self._right_joint is not None:
                right.update(self._right_joint)

            return {
                "timestamp": time.time(),
                "left": left if left else None,
                "right": right if right else None,
            }

    # -- Message converters --
    @staticmethod
    def _poscmd_to_dict(msg):
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

    @staticmethod
    def _jointinfo_to_dict(msg):
        return {
            "joint_pos": list(msg.joint_pos),
            "joint_vel": list(msg.joint_vel),
        }


# ---------------------------------------------------------------------------
# ROS1 subscriber (runs in a background thread)
# ---------------------------------------------------------------------------
def ros_thread(state, mode, ee_left_topic, ee_right_topic,
               joint_left_topic, joint_right_topic):
    """Initialise ROS node and spin in a daemon thread."""
    rospy.init_node("ws_bridge", anonymous=True, disable_signals=True)

    if mode in ("ee", "both"):
        from arm_control.msg import PosCmd
        rospy.Subscriber(ee_left_topic, PosCmd, state.update_left_ee, queue_size=1)
        rospy.Subscriber(ee_right_topic, PosCmd, state.update_right_ee, queue_size=1)
        rospy.loginfo("[ws_bridge] EE topics: left=%s  right=%s", ee_left_topic, ee_right_topic)

    if mode in ("joint", "both"):
        from arm_control.msg import JointInformation
        rospy.Subscriber(joint_left_topic, JointInformation, state.update_left_joint, queue_size=1)
        rospy.Subscriber(joint_right_topic, JointInformation, state.update_right_joint, queue_size=1)
        rospy.loginfo("[ws_bridge] Joint topics: left=%s  right=%s", joint_left_topic, joint_right_topic)

    rospy.spin()


# ---------------------------------------------------------------------------
# WebSocket server (main thread, asyncio)
# ---------------------------------------------------------------------------
async def ws_handler(websocket, state, hz, mode):
    """Push arm state to a single connected client at *hz* frequency."""
    interval = 1.0 / hz
    peer = websocket.remote_address
    print(f"[ws_bridge] Client connected: {peer}")
    try:
        while True:
            payload = json.dumps(state.snapshot(mode))
            await websocket.send(payload)
            await asyncio.sleep(interval)
    except Exception:
        print(f"[ws_bridge] Client disconnected: {peer}")


async def serve(state, port, hz, mode):
    import websockets

    handler = lambda ws, path=None: ws_handler(ws, state, hz, mode)
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"[ws_bridge] WebSocket server listening on 0.0.0.0:{port}  "
              f"(mode={mode}, {hz} Hz)")
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ROS1 -> WebSocket bridge for remote teleoperation"
    )
    parser.add_argument("--port", type=int, default=5555,
                        help="WebSocket port (default: 5555)")
    parser.add_argument("--hz", type=int, default=200,
                        help="Push frequency in Hz (default: 200)")
    parser.add_argument("--mode", type=str, default="ee",
                        choices=["ee", "joint", "both"],
                        help="Data mode: ee (EE pose), joint (joint angles), "
                             "both (EE + joint). Default: ee")

    # EE topics
    parser.add_argument("--left_topic", type=str, default="/master1_pos_back",
                        help="Left arm PosCmd topic (default: /master1_pos_back)")
    parser.add_argument("--right_topic", type=str, default="/master2_pos_back",
                        help="Right arm PosCmd topic (default: /master2_pos_back)")

    # Joint topics
    parser.add_argument("--joint_left_topic", type=str, default="/joint_information",
                        help="Left arm JointInformation topic (default: /joint_information)")
    parser.add_argument("--joint_right_topic", type=str, default="/joint_information2",
                        help="Right arm JointInformation topic (default: /joint_information2)")

    args = parser.parse_args()

    state = ArmState()

    # Start ROS in a daemon thread
    t = threading.Thread(
        target=ros_thread,
        args=(state, args.mode,
              args.left_topic, args.right_topic,
              args.joint_left_topic, args.joint_right_topic),
        daemon=True,
    )
    t.start()

    # Give ROS a moment to initialise
    time.sleep(0.5)

    # Run WebSocket server in main thread
    try:
        asyncio.run(serve(state, args.port, args.hz, args.mode))
    except KeyboardInterrupt:
        print("\n[ws_bridge] Shutting down.")


if __name__ == "__main__":
    main()
