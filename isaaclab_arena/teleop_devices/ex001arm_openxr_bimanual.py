# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3RelRetargeterCfg

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class Ex001ArmOpenXRBimanualTeleopDevice(TeleopDeviceBase):
    """
    OpenXR bimanual teleop device for EX001Arm.

    Retargeter order (14D total):
      left arm SE(3) (6) + left gripper (1) + right arm SE(3) (6) + right gripper (1)
    """

    name = "ex001arm_openxr_bimanual"

    def __init__(
        self,
        sim_device: str | None = None,
        delta_pos_scale_factor: float = 10.0,
        delta_rot_scale_factor: float = 10.0,
        alpha_pos: float = 0.5,
        alpha_rot: float = 0.5,
        zero_out_xy_rotation: bool = False,
        use_wrist_rotation: bool = True,
        use_wrist_position: bool = True,
        enable_visualization: bool = False,
    ):
        super().__init__(sim_device=sim_device)
        self.delta_pos_scale_factor = delta_pos_scale_factor
        self.delta_rot_scale_factor = delta_rot_scale_factor
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.zero_out_xy_rotation = zero_out_xy_rotation
        self.use_wrist_rotation = use_wrist_rotation
        self.use_wrist_position = use_wrist_position
        self.enable_visualization = enable_visualization

    def _make_se3_cfg(self, bound_hand: OpenXRDevice.TrackingTarget) -> Se3RelRetargeterCfg:
        return Se3RelRetargeterCfg(
            bound_hand=bound_hand,
            zero_out_xy_rotation=self.zero_out_xy_rotation,
            use_wrist_rotation=self.use_wrist_rotation,
            use_wrist_position=self.use_wrist_position,
            delta_pos_scale_factor=self.delta_pos_scale_factor,
            delta_rot_scale_factor=self.delta_rot_scale_factor,
            alpha_pos=self.alpha_pos,
            alpha_rot=self.alpha_rot,
            enable_visualization=self.enable_visualization,
            sim_device=self.sim_device or "cpu",
        )

    def _make_gripper_cfg(self, bound_hand: OpenXRDevice.TrackingTarget) -> GripperRetargeterCfg:
        return GripperRetargeterCfg(bound_hand=bound_hand, sim_device=self.sim_device or "cpu")

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                self.name: OpenXRDeviceCfg(
                    retargeters=[
                        self._make_se3_cfg(OpenXRDevice.TrackingTarget.HAND_LEFT),
                        self._make_gripper_cfg(OpenXRDevice.TrackingTarget.HAND_LEFT),
                        self._make_se3_cfg(OpenXRDevice.TrackingTarget.HAND_RIGHT),
                        self._make_gripper_cfg(OpenXRDevice.TrackingTarget.HAND_RIGHT),
                    ],
                    sim_device=self.sim_device,
                ),
            }
        )
