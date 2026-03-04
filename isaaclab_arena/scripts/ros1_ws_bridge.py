#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS1 -> WebSocket bridge with optional HDF5 recording.

Subscribes to master arm ROS1 topics and pushes state to the simulation PC
via WebSocket for real-time teleoperation in IsaacLab-Arena.

Supports three modes:
    ee    -- End-effector only  (PosCmd topics)
    joint -- Joint position only (JointControl topics)
    both  -- Both EE and joint data in one payload

Recording (--record):
    Buffers every transmitted snapshot and writes HDF5 files whose layout
    matches the IsaacLab demo format (obs/eef_pos, obs/eef_quat,
    obs/joint_pos, actions, timestamps, etc.) so that real-robot data can
    be directly compared with simulation recordings.

    Terminal commands while recording:
        Enter   = save current episode, start new buffer
        d+Enter = discard current buffer, start fresh
        s+Enter = print status (buffer size, saved count)
        Ctrl+C  = save remaining data and exit

Dependencies:
    pip install websockets              # WebSocket server
    pip install h5py numpy scipy        # Recording (only needed with --record)

Usage:
    # Normal bridge (no recording)
    python ros1_ws_bridge.py --mode both

    # Bridge with recording
    python ros1_ws_bridge.py --mode both --record --output_dir ./real_demos

Topic conventions (defaults):
    /master1_pos_back   -> left arm  EE   (arm_control/PosCmd)
    /master2_pos_back   -> right arm EE   (arm_control/PosCmd)
    /joint_control      -> left arm  joints (arm_control/JointControl)
    /joint_control2     -> right arm joints (arm_control/JointControl)
"""

from __future__ import print_function

import argparse
import asyncio
import json
import os
import sys
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
        # Joint data (from JointControl)
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
# Demo Recorder: buffer snapshots and write HDF5
# ---------------------------------------------------------------------------
class DemoRecorder:
    """Records arm state snapshots to HDF5 files in IsaacLab demo format.

    HDF5 layout per episode file::

        data/
          attrs: total, env_args
          demo_0/
            attrs: num_samples
            obs/
              eef_pos           (N, 3) float32
              eef_quat          (N, 4) float32   [w, x, y, z]
              gripper_pos       (N, 1) float32
              right_eef_pos     (N, 3) float32
              right_eef_quat    (N, 4) float32
              right_gripper_pos (N, 1) float32
              joint_pos         (N, 14) float32
              joint_vel         (N, 14) float32
              timestamps        (N,) float64
              eef_state         (N, 14) float32   [lxyz lrpy lg rxyz rrpy rg]
            actions             (N, 14) float32
    """

    MAX_BUFFER_FRAMES = 200_000  # auto-save safety (~16 min @ 200 Hz)

    def __init__(self, output_dir, prefix="real_episode", record_hz=0):
        import h5py as _h5py
        import numpy as _np
        from scipy.spatial.transform import Rotation as _Rot

        self._h5py = _h5py
        self._np = _np
        self._Rot = _Rot

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = prefix
        self.record_hz = record_hz

        self._lock = threading.Lock()
        self._buffer = []
        self._episode_idx = self._find_next_index()
        self._recording = True
        self.total_saved = 0

    # -- index management --

    def _find_next_index(self):
        idx = 0
        while os.path.exists(
            os.path.join(self.output_dir, f"{self.prefix}{idx}.hdf5")
        ):
            idx += 1
        return idx

    # -- public API --

    @property
    def buffer_size(self):
        with self._lock:
            return len(self._buffer)

    @property
    def episode_idx(self):
        return self._episode_idx

    def record(self, snapshot):
        """Append a valid snapshot to the buffer."""
        left = snapshot.get("left")
        right = snapshot.get("right")
        if left is None or right is None:
            return
        with self._lock:
            if not self._recording:
                return
            self._buffer.append({
                "timestamp": snapshot["timestamp"],
                "left": dict(left),
                "right": dict(right),
            })
            if len(self._buffer) >= self.MAX_BUFFER_FRAMES:
                self._auto_save_locked()

    def save_episode(self):
        """Save buffer to HDF5 with auto-incremented name. Returns filepath or None."""
        with self._lock:
            if not self._buffer:
                return None
            data = list(self._buffer)
            self._buffer = []

        filepath = os.path.join(
            self.output_dir, f"{self.prefix}{self._episode_idx}.hdf5"
        )
        self._write_hdf5(filepath, data)
        self._episode_idx += 1
        self.total_saved += 1
        return filepath

    def save_episode_as(self, name):
        """Save buffer to HDF5 with a specific filename (for synced recording).

        ``name`` is the bare stem, e.g. ``"arx_remote_joint_episode27"``.
        The file is written to ``<output_dir>/<name>.hdf5``.
        """
        with self._lock:
            if not self._buffer:
                return None
            data = list(self._buffer)
            self._buffer = []

        filepath = os.path.join(self.output_dir, f"{name}.hdf5")
        self._write_hdf5(filepath, data)
        self.total_saved += 1
        # Keep episode_idx in sync with the number extracted from the name
        import re
        m = re.search(r"(\d+)$", name)
        if m:
            self._episode_idx = int(m.group(1)) + 1
        else:
            self._episode_idx += 1
        return filepath

    def discard_episode(self):
        """Discard buffer and return number of discarded frames."""
        with self._lock:
            n = len(self._buffer)
            self._buffer = []
        return n

    # -- internal helpers --

    def _auto_save_locked(self):
        """Auto-save when buffer is full. Called with self._lock held."""
        data = list(self._buffer)
        self._buffer = []
        filepath = os.path.join(
            self.output_dir, f"{self.prefix}{self._episode_idx}.hdf5"
        )
        threading.Thread(
            target=self._write_hdf5, args=(filepath, data), daemon=True
        ).start()
        print(f"[Recorder] Auto-saved {len(data)} frames -> {filepath}")
        self._episode_idx += 1
        self.total_saved += 1

    def _rpy_to_quat_wxyz_batch(self, rpy):
        """(N, 3) RPY  ->  (N, 4) quaternion in [w, x, y, z] convention."""
        np = self._np
        q_xyzw = self._Rot.from_euler("xyz", rpy).as_quat()  # [x,y,z,w]
        return np.column_stack([q_xyzw[:, 3], q_xyzw[:, :3]]).astype(np.float32)

    def _write_hdf5(self, filepath, snapshots):
        h5py = self._h5py
        np = self._np
        N = len(snapshots)

        timestamps = np.array(
            [s["timestamp"] for s in snapshots], dtype=np.float64
        )

        # Pre-allocate arrays
        eef_pos = np.zeros((N, 3), dtype=np.float32)
        eef_rpy = np.zeros((N, 3), dtype=np.float64)
        gripper_pos = np.zeros((N, 1), dtype=np.float32)

        r_eef_pos = np.zeros((N, 3), dtype=np.float32)
        r_eef_rpy = np.zeros((N, 3), dtype=np.float64)
        r_gripper_pos = np.zeros((N, 1), dtype=np.float32)

        joint_pos = np.zeros((N, 14), dtype=np.float32)
        joint_vel = np.zeros((N, 14), dtype=np.float32)

        for i, snap in enumerate(snapshots):
            ld = snap["left"]
            rd = snap["right"]

            # -- EE (left) --
            if "x" in ld:
                eef_pos[i] = [ld["x"], ld["y"], ld["z"]]
                eef_rpy[i] = [ld["roll"], ld["pitch"], ld["yaw"]]
                gripper_pos[i, 0] = ld.get("gripper", 0.0)

            # -- EE (right) --
            if "x" in rd:
                r_eef_pos[i] = [rd["x"], rd["y"], rd["z"]]
                r_eef_rpy[i] = [rd["roll"], rd["pitch"], rd["yaw"]]
                r_gripper_pos[i, 0] = rd.get("gripper", 0.0)

            # -- Joints (left: indices 0-6, right: indices 7-13) --
            if "joint_pos" in ld:
                jp = ld["joint_pos"]
                n = min(len(jp), 7)
                joint_pos[i, :n] = jp[:n]
            if "joint_pos" in rd:
                jp = rd["joint_pos"]
                n = min(len(jp), 7)
                joint_pos[i, 7:7 + n] = jp[:n]

            if "joint_vel" in ld:
                jv = ld["joint_vel"]
                n = min(len(jv), 7)
                joint_vel[i, :n] = jv[:n]
            if "joint_vel" in rd:
                jv = rd["joint_vel"]
                n = min(len(jv), 7)
                joint_vel[i, 7:7 + n] = jv[:n]

        # RPY -> quaternion [w,x,y,z]
        eef_quat = self._rpy_to_quat_wxyz_batch(eef_rpy)
        r_eef_quat = self._rpy_to_quat_wxyz_batch(r_eef_rpy)

        # LeRobot-compatible 14D EE state: [lxyz lrpy lg rxyz rrpy rg]
        eef_state = np.zeros((N, 14), dtype=np.float32)
        eef_state[:, 0:3] = eef_pos
        eef_state[:, 3:6] = eef_rpy.astype(np.float32)
        eef_state[:, 6] = gripper_pos[:, 0]
        eef_state[:, 7:10] = r_eef_pos
        eef_state[:, 10:13] = r_eef_rpy.astype(np.float32)
        eef_state[:, 13] = r_gripper_pos[:, 0]

        # Duration / FPS stats
        duration = float(timestamps[-1] - timestamps[0]) if N > 1 else 0.0
        actual_fps = (N - 1) / duration if duration > 0 else 0.0

        with h5py.File(filepath, "w") as f:
            f.attrs["source"] = "real_robot_bridge"
            f.attrs["record_hz"] = self.record_hz
            f.attrs["actual_fps"] = actual_fps
            f.attrs["duration_s"] = duration

            data_grp = f.create_group("data")
            data_grp.attrs["total"] = N
            data_grp.attrs["env_args"] = json.dumps({
                "source": "real_robot",
                "type": 2,
                "sim_args": {
                    "dt": 1.0 / self.record_hz if self.record_hz else 0,
                    "num_envs": 1,
                },
            })

            demo = data_grp.create_group("demo_0")
            demo.attrs["num_samples"] = N

            obs = demo.create_group("obs")
            obs.create_dataset("eef_pos", data=eef_pos, compression="gzip")
            obs.create_dataset("eef_quat", data=eef_quat, compression="gzip")
            obs.create_dataset("gripper_pos", data=gripper_pos, compression="gzip")
            obs.create_dataset("right_eef_pos", data=r_eef_pos, compression="gzip")
            obs.create_dataset("right_eef_quat", data=r_eef_quat, compression="gzip")
            obs.create_dataset("right_gripper_pos", data=r_gripper_pos, compression="gzip")
            obs.create_dataset("joint_pos", data=joint_pos, compression="gzip")
            obs.create_dataset("joint_vel", data=joint_vel, compression="gzip")
            obs.create_dataset("timestamps", data=timestamps, compression="gzip")
            obs.create_dataset("eef_state", data=eef_state, compression="gzip")

            # actions = raw joint commands being sent to simulation
            demo.create_dataset("actions", data=joint_pos.copy(), compression="gzip")

        print(
            f"[Recorder] Wrote {filepath}  "
            f"({N} frames, {duration:.1f}s, {actual_fps:.1f} fps)"
        )


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
        from arm_control.msg import JointControl
        rospy.Subscriber(joint_left_topic, JointControl, state.update_left_joint, queue_size=1)
        rospy.Subscriber(joint_right_topic, JointControl, state.update_right_joint, queue_size=1)
        rospy.loginfo("[ws_bridge] Joint topics: left=%s  right=%s", joint_left_topic, joint_right_topic)

    rospy.spin()


# ---------------------------------------------------------------------------
# WebSocket server (main thread, asyncio)
# ---------------------------------------------------------------------------
async def ws_handler(websocket, state, hz, mode, recorder=None):
    """Push arm state to a connected client and listen for episode signals.

    The sim client can send JSON messages back to control recording:

    * ``{"cmd": "save_episode", "name": "arx_remote_joint_episode27"}``
      - save the current buffer as ``<name>.hdf5``.
    * ``{"cmd": "discard_episode"}``
      - discard the current buffer.
    """
    interval = 1.0 / hz
    peer = websocket.remote_address
    print(f"[ws_bridge] Client connected: {peer}")

    async def _sender():
        while True:
            payload = json.dumps(state.snapshot(mode))
            await websocket.send(payload)
            await asyncio.sleep(interval)

    async def _receiver():
        try:
            async for raw in websocket:
                if recorder is None:
                    continue
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                cmd = msg.get("cmd")
                if cmd == "save_episode":
                    name = msg.get("name", "")
                    if name:
                        path = recorder.save_episode_as(name)
                    else:
                        path = recorder.save_episode()
                    if path:
                        print(f"[Recorder] Synced save: {path}")
                    else:
                        print("[Recorder] Synced save requested but buffer empty")
                elif cmd == "discard_episode":
                    n = recorder.discard_episode()
                    print(f"[Recorder] Remote discard: {n} frames")
        except Exception:
            pass

    try:
        await asyncio.gather(_sender(), _receiver())
    except Exception:
        print(f"[ws_bridge] Client disconnected: {peer}")


async def record_loop(state, recorder, hz, mode):
    """Periodically sample ArmState and feed to DemoRecorder."""
    interval = 1.0 / hz
    frame_count = 0
    while True:
        snapshot = state.snapshot(mode)
        recorder.record(snapshot)
        frame_count += 1
        if frame_count % (hz * 10) == 0:
            buf = recorder.buffer_size
            if buf > 0:
                print(
                    f"[Recorder] Buffer: {buf} frames "
                    f"(~{buf / hz:.1f}s) | saved: {recorder.total_saved}"
                )
        await asyncio.sleep(interval)


def stdin_manager(recorder, hz):
    """Background thread: read terminal commands for episode management."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().lower()

            if cmd in ("", "n"):
                path = recorder.save_episode()
                if path:
                    print(f"[Recorder] Episode saved: {path}")
                else:
                    print("[Recorder] Buffer empty, nothing to save")
                print(
                    f"[Recorder] Recording episode #{recorder.episode_idx}. "
                    f"Press Enter to save."
                )

            elif cmd == "d":
                n = recorder.discard_episode()
                print(
                    f"[Recorder] Discarded {n} frames "
                    f"(~{n / hz:.1f}s). New episode #{recorder.episode_idx}."
                )

            elif cmd == "s":
                buf = recorder.buffer_size
                print(
                    f"[Recorder] Buffer: {buf} frames (~{buf / hz:.1f}s) | "
                    f"Saved episodes: {recorder.total_saved} | "
                    f"Next index: {recorder.episode_idx}"
                )

        except (EOFError, KeyboardInterrupt):
            break


async def serve(state, port, hz, mode, recorder=None):
    import websockets

    handler = lambda ws, path=None: ws_handler(ws, state, hz, mode, recorder)
    async with websockets.serve(handler, "0.0.0.0", port):
        print(
            f"[ws_bridge] WebSocket server listening on 0.0.0.0:{port}  "
            f"(mode={mode}, {hz} Hz)"
        )
        if recorder:
            asyncio.get_event_loop().create_task(
                record_loop(state, recorder, hz, mode)
            )
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
    parser.add_argument("--joint_left_topic", type=str, default="/joint_control",
                        help="Left arm JointControl topic (default: /joint_control)")
    parser.add_argument("--joint_right_topic", type=str, default="/joint_control2",
                        help="Right arm JointControl topic (default: /joint_control2)")

    # Recording
    parser.add_argument("--record", action="store_true",
                        help="Enable HDF5 recording of all transmitted data")
    parser.add_argument("--output_dir", type=str, default="./real_demos",
                        help="Output directory for HDF5 files (default: ./real_demos)")
    parser.add_argument("--record_prefix", type=str, default="real_episode",
                        help="Filename prefix for episode files (default: real_episode)")

    args = parser.parse_args()

    state = ArmState()

    # -- Setup recorder (if enabled) --
    recorder = None
    if args.record:
        try:
            recorder = DemoRecorder(
                output_dir=args.output_dir,
                prefix=args.record_prefix,
                record_hz=args.hz,
            )
        except ImportError as e:
            print(
                f"[Recorder] ERROR: missing dependency ({e}).\n"
                "  Install with: pip install h5py numpy scipy"
            )
            sys.exit(1)

        print("=" * 60)
        print(f"[Recorder] ENABLED  |  output: {args.output_dir}/")
        print(f"[Recorder] Recording at {args.hz} Hz in '{args.mode}' mode")
        print(f"[Recorder] HDF5 format: obs/{{eef_pos, eef_quat, joint_pos, ...}}")
        print(f"[Recorder] Starting episode #{recorder.episode_idx}")
        print("[Recorder] Commands:")
        print("           Enter   = save episode & start new")
        print("           d+Enter = discard & start new")
        print("           s+Enter = show status")
        print("           Ctrl+C  = save & exit")
        print("=" * 60)

    # -- Start ROS in a daemon thread --
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

    # -- Start stdin manager for episode control --
    if recorder:
        stdin_t = threading.Thread(
            target=stdin_manager, args=(recorder, args.hz), daemon=True
        )
        stdin_t.start()

    # -- Run WebSocket server in main thread --
    try:
        asyncio.run(serve(state, args.port, args.hz, args.mode, recorder))
    except KeyboardInterrupt:
        if recorder:
            path = recorder.save_episode()
            if path:
                print(f"\n[Recorder] Final episode saved: {path}")
            print(
                f"[Recorder] Session done. "
                f"Total episodes saved: {recorder.total_saved}"
            )
        print("\n[ws_bridge] Shutting down.")


if __name__ == "__main__":
    main()
