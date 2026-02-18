#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WebSocket 诊断工具 -- 实时打印远端机械臂数据.

用法:
    python tools/ws_debug_client.py --ip 10.100.21.249 --port 5555
"""

import argparse
import json
import time

import websocket


def main():
    parser = argparse.ArgumentParser(description="WebSocket teleop data debugger")
    parser.add_argument("--ip", type=str, default="10.100.21.249")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    url = f"ws://{args.ip}:{args.port}"
    print(f"Connecting to {url} ...")
    ws = websocket.create_connection(url, timeout=5)
    print("Connected!\n")

    prev_left = None
    prev_right = None
    count = 0

    try:
        while True:
            raw = ws.recv()
            data = json.loads(raw)

            left = data.get("left")
            right = data.get("right")

            count += 1
            if count % 25 == 0:  # 每 0.5 秒打印一次 (50Hz)
                print("=" * 80)
                print(f"Frame #{count}")

                if left:
                    print(f"  LEFT  (master1):  x={left['x']:.4f}  y={left['y']:.4f}  z={left['z']:.4f}")
                    print(f"          roll={left['roll']:.4f}  pitch={left['pitch']:.4f}  yaw={left['yaw']:.4f}")
                    print(f"          gripper={left['gripper']:.4f}  mode1={left.get('mode1')}  mode2={left.get('mode2')}")
                    if prev_left:
                        dx = left['x'] - prev_left['x']
                        dy = left['y'] - prev_left['y']
                        dz = left['z'] - prev_left['z']
                        print(f"          delta_pos: dx={dx:.6f}  dy={dy:.6f}  dz={dz:.6f}")
                else:
                    print("  LEFT  (master1):  *** NO DATA ***")

                if right:
                    print(f"  RIGHT (master2):  x={right['x']:.4f}  y={right['y']:.4f}  z={right['z']:.4f}")
                    print(f"          roll={right['roll']:.4f}  pitch={right['pitch']:.4f}  yaw={right['yaw']:.4f}")
                    print(f"          gripper={right['gripper']:.4f}  mode1={right.get('mode1')}  mode2={right.get('mode2')}")
                    if prev_right:
                        dx = right['x'] - prev_right['x']
                        dy = right['y'] - prev_right['y']
                        dz = right['z'] - prev_right['z']
                        print(f"          delta_pos: dx={dx:.6f}  dy={dy:.6f}  dz={dz:.6f}")
                else:
                    print("  RIGHT (master2):  *** NO DATA ***")

                prev_left = left.copy() if left else None
                prev_right = right.copy() if right else None

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
