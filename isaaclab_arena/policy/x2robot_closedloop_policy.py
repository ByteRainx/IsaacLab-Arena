# Copyright (c) 2025, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0

"""
X2Robot Desktop Closedloop Policy for IsaacLab-Arena

This module integrates the x2robot_client (Desktop dual-arm) inference system 
with IsaacLab-Arena simulation. Only supports dual-arm control (left arm, right arm, grippers).
"""

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# WebSocket client dependencies
try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    from websockets.sync.client import connect as ws_connect
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from isaaclab_arena.policy.policy_base import PolicyBase


class SimpleWebSocketClient:
    """Simple WebSocket client for X2Robot inference server."""
    
    def __init__(self, address: str, port: int):
        self.uri = f"ws://{address}:{port}"
        self.connection = None
        self.metadata = {}
        
    def connect_sync(self) -> dict:
        """Connect to the server and receive metadata."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package is required. Install with: pip install websockets")
        if not HAS_MSGPACK:
            raise ImportError("msgpack and msgpack-numpy are required. Install with: pip install msgpack msgpack-numpy")
            
        self.connection = ws_connect(self.uri, max_size=None)
        metadata_bytes = self.connection.recv()
        self.metadata = msgpack.unpackb(metadata_bytes, raw=False)
        return self.metadata
    
    def predict_sync(self, observation: dict) -> dict:
        """Send observation and receive action prediction."""
        if self.connection is None:
            raise RuntimeError("Not connected to server. Call connect_sync() first.")
        
        obs_bytes = msgpack.packb(observation, use_bin_type=True)
        self.connection.send(obs_bytes)
        
        response_bytes = self.connection.recv()
        if isinstance(response_bytes, str):
            raise RuntimeError(f"Server error: {response_bytes}")
        
        return msgpack.unpackb(response_bytes, raw=False)
    
    def close(self):
        """Close the connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None


@dataclass
class X2RobotPolicyConfig:
    """Configuration for X2Robot Desktop policy (dual-arm only)."""
    
    model_address: str = "localhost"
    model_port: int = 8000
    instruction: str = "pick up the object"
    control_mode: str = "end_pose"
    action_horizon: int = 16
    action_chunk_length: int = 4
    camera_left: str = "left_wrist_cam"
    camera_right: str = "right_wrist_cam"
    camera_front: str = None
    target_image_size: tuple = (480, 640, 3)


class X2RobotClosedloopPolicy(PolicyBase):
    """Desktop dual-arm closedloop policy using X2Robot inference server."""
    
    def __init__(
        self,
        config: X2RobotPolicyConfig | Dict[str, Any] | str,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        if isinstance(config, str):
            import yaml
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = X2RobotPolicyConfig(**config_dict)
        elif isinstance(config, dict):
            self.config = X2RobotPolicyConfig(**config)
        else:
            self.config = config
        
        self.num_envs = num_envs
        self.device = device
        
        self.client: Optional[SimpleWebSocketClient] = None
        self._init_inference_client()
        
        self.action_dim = 14
        
        self.current_action_chunk = torch.zeros(
            (num_envs, self.config.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    def _init_inference_client(self):
        address = self.config.model_address
        port = self.config.model_port
        uri = f"ws://{address}:{port}"
        
        self.client = SimpleWebSocketClient(address, port)
        try:
            metadata = self.client.connect_sync()
            print(f"[X2RobotPolicy] Connected to server at {uri}")
            if metadata:
                print(f"[X2RobotPolicy] Server metadata: {metadata}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to X2Robot server at {uri}: {e}")
    
    @staticmethod
    def _compress_image_to_base64(image: np.ndarray) -> Optional[str]:
        if image is None:
            return None
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            return None
        return base64.b64encode(encoded).decode('utf-8')
    
    def _collect_observations(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        policy_obs = observation.get("policy", {})
        
        def get_camera_base64(cam_name: Optional[str]) -> Optional[str]:
            if cam_name is None:
                return None
            cam_obs = observation.get("camera_obs", {})
            rgb = cam_obs.get(cam_name) or policy_obs.get(cam_name)
            if rgb is None:
                return None
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if len(rgb.shape) == 4:
                rgb = rgb[0]
            h, w = rgb.shape[:2]
            th, tw = self.config.target_image_size[:2]
            if h != th or w != tw:
                rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
            return self._compress_image_to_base64(rgb)
        
        camera_left = get_camera_base64(self.config.camera_left)
        camera_right = get_camera_base64(self.config.camera_right)
        camera_front = get_camera_base64(self.config.camera_front)
        
        def to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.array(x)
        
        def build_arm_state(eef_pos, eef_quat, gripper) -> np.ndarray:
            eef_pos = to_numpy(eef_pos)
            eef_quat = to_numpy(eef_quat)
            gripper = to_numpy(gripper)
            
            if eef_pos is None:
                return np.zeros(7, dtype=np.float32)
            
            if len(eef_pos.shape) > 1:
                eef_pos = eef_pos[0]
            if eef_quat is not None and len(eef_quat.shape) > 1:
                eef_quat = eef_quat[0]
            
            if eef_quat is not None:
                euler = Rotation.from_quat(eef_quat).as_euler('xyz')
            else:
                euler = np.zeros(3)
            
            if gripper is not None:
                if len(gripper.shape) > 0:
                    gripper_val = float(gripper[0])
                else:
                    gripper_val = float(gripper)
            else:
                gripper_val = 0.0
            
            return np.concatenate([eef_pos, euler, [gripper_val]]).astype(np.float32)
        
        follow1_pos = build_arm_state(
            policy_obs.get("eef_pos"),
            policy_obs.get("eef_quat"),
            policy_obs.get("gripper_pos"),
        )
        
        follow2_pos = build_arm_state(
            policy_obs.get("right_eef_pos"),
            policy_obs.get("right_eef_quat"),
            policy_obs.get("right_gripper_pos"),
        )
        
        x2robot_obs = {
            "ACTION_FOLLOW1_POS": follow1_pos,
            "ACTION_FOLLOW2_POS": follow2_pos,
            "instruction": self.config.instruction,
        }
        
        if camera_left is not None:
            x2robot_obs["CAMERA_LEFT"] = camera_left
        if camera_right is not None:
            x2robot_obs["CAMERA_RIGHT"] = camera_right
        if camera_front is not None:
            x2robot_obs["CAMERA_FRONT"] = camera_front
        
        return x2robot_obs
    
    def _convert_action_to_tensor(self, response: Dict[str, Any]) -> torch.Tensor:
        follow1 = response.get("FOLLOW1_POS") or response.get("FOLLOW1_JOINTS")
        follow2 = response.get("FOLLOW2_POS") or response.get("FOLLOW2_JOINTS")
        
        if follow1 is None or follow2 is None:
            print(f"[X2RobotPolicy] Missing action keys: {response.keys()}")
            return torch.zeros(
                (self.config.action_horizon, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
        
        follow1 = np.array(follow1)
        follow2 = np.array(follow2)
        
        if len(follow1.shape) == 1:
            follow1 = follow1.reshape(1, -1)
        if len(follow2.shape) == 1:
            follow2 = follow2.reshape(1, -1)
        
        action_np = np.concatenate([follow1, follow2], axis=-1)
        
        return torch.from_numpy(action_np).float().to(self.device)
    
    def get_action(self, env: gym.Env, observation: Dict[str, Any]) -> torch.Tensor:
        if any(self.env_requires_new_action_chunk):
            action_chunk = self._query_server(observation)
            
            mask = self.env_requires_new_action_chunk
            self.current_action_chunk[mask] = action_chunk[mask]
            self.current_action_index[mask] = 0
            self.env_requires_new_action_chunk[mask] = False
        
        action = self.current_action_chunk[
            torch.arange(self.num_envs, device=self.device),
            self.current_action_index
        ]
        
        self.current_action_index += 1
        
        exhausted = self.current_action_index >= self.config.action_chunk_length
        self.env_requires_new_action_chunk[exhausted] = True
        self.current_action_index[exhausted] = -1
        
        return action
    
    def _query_server(self, observation: Dict[str, Any]) -> torch.Tensor:
        x2robot_obs = self._collect_observations(observation)
        
        try:
            response = self.client.predict_sync(x2robot_obs)
        except Exception as e:
            print(f"[X2RobotPolicy] Inference failed: {e}")
            return torch.zeros(
                (self.num_envs, self.config.action_horizon, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
        
        action_tensor = self._convert_action_to_tensor(response)
        
        if len(action_tensor.shape) == 2:
            action_tensor = action_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        if action_tensor.shape[1] < self.config.action_horizon:
            pad_size = self.config.action_horizon - action_tensor.shape[1]
            padding = action_tensor[:, -1:, :].repeat(1, pad_size, 1)
            action_tensor = torch.cat([action_tensor, padding], dim=1)
        
        return action_tensor[:, :self.config.action_horizon, :]
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = slice(None)
        
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True
    
    def close(self):
        if self.client is not None:
            try:
                self.client.close()
                print("[X2RobotPolicy] Connection closed")
            except Exception as e:
                print(f"[X2RobotPolicy] Error closing: {e}")
    
    def __del__(self):
        self.close()
