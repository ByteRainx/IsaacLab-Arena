      
import os
import numpy as np
import argparse
from pathlib import Path
import shutil
import torch
import torchvision

#from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
#from lerobot.datasets.utils import (
#    check_timestamps_sync,
#    get_episode_data_index,
#    validate_episode_buffer,
#    validate_frame,
#    write_episode,
#    write_episode_stats,
#    write_info,
#)
from lerobot.datasets.utils import check_timestamps_sync, get_episode_data_index, validate_episode_buffer, validate_frame, write_episode, write_episode_stats, write_info
#from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.compute_stats import get_feature_stats
import pandas as pd

import re
import numpy as np
from collections import defaultdict
import json
import os

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from lerobot.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices


_DOF_DIM_MAPPING={
        "follow_left_position": 3,
        "follow_left_rotation": 3,
        "follow_left_gripper": 1,
        "follow_right_position": 3,
        "follow_right_rotation": 3,
        "follow_right_gripper": 1,
        'head_rotation': 2,
        "lifting_mechanism_position": 1,
        'car_pose': 3, 
    }


# 统一的action key映射：模型key -> 原始数据key
_ACTION_KEY_FULL_MAPPING = {
    # ARX系列
    'follow_right_arm_joint_pos': 'follow_right_joint_pos',
    'follow_right_arm_joint_dev': 'follow_right_joint_dev',
    'follow_right_arm_joint_cur': 'follow_right_joint_cur',
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'master_right_arm_joint_pos': 'master_right_joint_pos',
    'master_right_arm_joint_dev': 'master_right_joint_dev',
    'master_right_arm_joint_cur': 'master_right_joint_cur',
    'master_right_ee_cartesian_pos': 'master_right_position',
    'master_right_ee_rotation': 'master_right_rotation',
    'master_right_gripper': 'master_right_gripper',
    'follow_left_arm_joint_pos': 'follow_left_joint_pos',
    'follow_left_arm_joint_dev': 'follow_left_joint_dev',
    'follow_left_arm_joint_cur': 'follow_left_joint_cur',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper',
    'master_left_arm_joint_pos': 'master_left_joint_pos',
    'master_left_arm_joint_dev': 'master_left_joint_dev',
    'master_left_arm_joint_cur': 'master_left_joint_cur',
    'master_left_ee_cartesian_pos': 'master_left_position',
    'master_left_ee_rotation': 'master_left_rotation',
    'master_left_gripper': 'master_left_gripper',
    
    # JAKA系列 - 添加原始数据映射
    'follow_left_ee_cartesian_pos_jaka': 'follow_left_position',
    'follow_right_ee_cartesian_pos_jaka': 'follow_right_position',
    'follow_left_ee_rotation_jaka': 'follow_left_rotation',
    'follow_right_ee_rotation_jaka': 'follow_right_rotation',
    'follow_left_arm_joint_pos_jaka': 'follow_left_joint_pos',
    'follow_right_arm_joint_pos_jaka': 'follow_right_joint_pos',
    'follow_left_arm_joint_cur_jaka': 'follow_left_joint_cur',
    'follow_right_arm_joint_cur_jaka': 'follow_right_joint_cur',
    
    # 手部控制
    'follow_left_hand_joint_pos': 'follow_left_hand_joint_pos',
    'follow_left_hand_joint_dev': 'follow_left_hand_joint_dev',
    'follow_right_hand_joint_pos': 'follow_right_hand_joint_pos',
    'follow_right_hand_joint_dev': 'follow_right_hand_joint_dev',
    
    # 夹爪力控 - 从arm_joint_cur的最后一个关节提取
    'follow_left_gripper_cur': 'follow_left_joint_cur[-1]',
    'follow_right_gripper_cur': 'follow_right_joint_cur[-1]',
    
    # 其他
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    'head_actions': 'head_rotation',
    "height":"lifting_mechanism_position",
}


_ACTION_KEY_FULL_MAPPING_INV = {v:k for k,v in _ACTION_KEY_FULL_MAPPING.items()}



def process_action(file_path, raw_key2model_key=_ACTION_KEY_FULL_MAPPING_INV, filter_angle_outliers=True):
    # ======== Step 1: 统一输入类型 ========
    mappings = raw_key2model_key
    # 如果传入的是单个字典，转换为单元素列表以统一处理
    if isinstance(mappings, dict):
        mappings = [mappings]
        return_list = False
    else:
        return_list = True
    
    # ======== Step 2: 加载原始动作数据 ========
    if isinstance(file_path, str):
        file_name = os.path.basename(file_path)
        action_path = os.path.join(file_path, f"{file_name}.json")
        with open(action_path, 'r') as file:
            actions = json.load(file)
    else:
        actions = file_path
    
    data = actions.get('data', [])
    if not data:
        raise ValueError("No 'data' field found in action file")
    
    # ======== Step 3: 提前编译正则表达式 ========
    index_pattern = re.compile(r'^(.+)\[(-?\d+)\]$')
    
    # ======== Step 4: 收集所有映射所需原始键 ========
    all_raw_keys = set()
    special_mappings_info = []  # 存储特殊映射的元信息
    
    for mapping_idx, mapping_dict in enumerate(mappings):
        for raw_key in mapping_dict:
            # 处理字符串键（可能包含索引）
            if isinstance(raw_key, str):
                match = index_pattern.match(raw_key)
                if match:
                    base_key = match.group(1)
                    all_raw_keys.add(base_key)
                    # 记录特殊映射信息
                    special_mappings_info.append({
                        'mapping_idx': mapping_idx,
                        'base_raw_key': base_key,
                        'index': int(match.group(2)),
                        'model_key': mapping_dict[raw_key],
                        'original_key': raw_key
                    })
                else:
                    all_raw_keys.add(raw_key)
            # 处理非字符串键
            else:
                all_raw_keys.add(raw_key)
    
    # ======== Step 5: 统一提取原始数据 ========
    aggregated_raw_data = defaultdict(list)
    for action in data:
        for key in all_raw_keys:
            if key in action:
                aggregated_raw_data[key].append(action[key])
            else:
                # 用NaN填充缺失值确保数组形状一致
                aggregated_raw_data[key].append([float('nan') for i in range(_DOF_DIM_MAPPING[key])])
    
    # 转换为NumPy数组
    for key in aggregated_raw_data:
        aggregated_raw_data[key] = np.array(aggregated_raw_data[key], dtype=np.float32)
    
    # ======== Step 6: 预构建轨迹存储对象 ========
    trajectories_list = [dict() for _ in range(len(mappings))]
    
    # ======== Step 7: 处理普通映射 ========
    for mapping_idx, mapping_dict in enumerate(mappings):
        for raw_key, model_key in mapping_dict.items():
            # 跳过特殊映射键
            if isinstance(raw_key, str) and index_pattern.match(raw_key):
                continue
            
            if raw_key in aggregated_raw_data:
                # 克隆数据避免原地修改
                trajectories_list[mapping_idx][model_key] = aggregated_raw_data[raw_key].copy()
    
    # ======== Step 8: 处理特殊映射 ========
    for info in special_mappings_info:
        base_key = info['base_raw_key']
        idx = info['mapping_idx']
        
        if base_key in aggregated_raw_data:
            data_arr = aggregated_raw_data[base_key]
            # 只处理2D数组（N×M）
            if data_arr.ndim == 2:
                # 处理负索引
                if info['index'] < 0:
                    target_index = data_arr.shape[1] + info['index']
                else:
                    target_index = info['index']
                
                # 安全提取索引
                if 0 <= target_index < data_arr.shape[1]:
                    trajectories_list[idx][info['model_key']] = data_arr[:, target_index:target_index+1]
                else:
                    print(f"Warning: Invalid index {info['index']} for key {info['original_key']}")
            else:
                print(f"Warning: Base key {base_key} is not 2D array for index access")
        else:
            print(f"Warning: Base key {base_key} not found in raw data")
    
    # ======== Step 9: 后处理 ========
    final_trajectories = []
    for traj in trajectories_list:
        # 四元数转欧拉角
        traj = quat2euler(traj)
        # 计算底盘速度
        traj = calculate_base_velocity(traj)
        if filter_angle_outliers:
            processed = smooth_action(traj)
            final_trajectories.append(processed)
        else:
            final_trajectories.append(traj)
    
    # ======== Step 10: 返回结果 ========
    return final_trajectories if return_list else final_trajectories[0]



def quat2euler(traj):
    if "follow_right_ee_rotation" in traj and traj["follow_right_ee_rotation"].shape[-1]==4:
        traj["follow_right_ee_rotation"] = R.from_quat(traj["follow_right_ee_rotation"]).as_euler('xyz')
    if "follow_left_ee_rotation" in traj and traj["follow_left_ee_rotation"].shape[-1]==4:
        traj["follow_left_ee_rotation"] = R.from_quat(traj["follow_left_ee_rotation"]).as_euler('xyz')
    if "master_right_ee_rotation" in traj and traj["master_right_ee_rotation"].shape[-1]==4:
        traj["master_right_ee_rotation"] = R.from_quat(traj["master_right_ee_rotation"]).as_euler('xyz')
    if "master_left_ee_rotation" in traj and traj["master_left_ee_rotation"].shape[-1]==4:
        traj["master_left_ee_rotation"] = R.from_quat(traj["master_left_ee_rotation"]).as_euler('xyz')
    return traj


def calculate_base_velocity(traj):
    if "car_pose" in traj and traj["car_pose"].shape[-1]==3 and not np.isnan(traj["car_pose"]).any():
        velocity = process_car_pose_to_base_velocity(traj["car_pose"])
        if velocity["valid"]:
            traj["car_pose"] = velocity["base_velocity_decomposed"]
        else:
            print("Warning: car_pose is invalid, pop it", flush=True)
            traj.pop("car_pose")
    return traj


def smooth_action(action):
    def _filter(traj, threshold = 3, alpha = 0.05, window=10):
        # Convert to pandas Series but preserve the original dtype
        orig_dtype = traj.dtype
        data = pd.Series(traj)
        derivatives = np.diff(data)

        spike_indices = np.where(abs(derivatives) > threshold)[0]
        if len(spike_indices) > 0:
            ema = data.ewm(alpha=alpha, adjust=True).mean()
            
            # Fix: Ensure the slice indices are within bounds
            start_idx = max(0, spike_indices[0] - window)
            end_idx = min(len(data), spike_indices[-1] + window + 1)
            
            # Get the corresponding segment from the EMA
            modified_seg = ema.iloc[start_idx:end_idx]
            
            # Ensure the lengths match before assignment and explicitly convert to the original dtype
            if len(modified_seg) > 0:
                # Convert values back to the original dtype before assignment
                data.iloc[start_idx:end_idx] = modified_seg.values.astype(orig_dtype)
                
        return data.to_numpy().astype(orig_dtype)  # Ensure we return the same dtype

    for key in ['follow_right_ee_rotation', 'follow_left_ee_rotation']:
        if key in action:  # Check if the key exists in the action dictionary
            try:
                # Process each dimension separately while preserving dtype
                orig_dtype = action[key].dtype
                filtered_traj = np.stack([_filter(action[key][:,i]) for i in range(3)], axis=1)
                if not np.isnan(filtered_traj).any():
                    action[key] = filtered_traj.astype(orig_dtype)  # Ensure consistent dtype
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not smooth {key} due to error: {e}")
    
    return action




def process_car_pose_to_base_velocity(car_pose, 
                                    outlier_threshold=3, 
                                    jump_threshold=1.0, 
                                    smooth_iterations=3, 
                                    strong_smooth=True):
    """
    🎯 处理car_pose数据，转换为本体坐标系base_velocity_decomposed，与batch_process_json_data.py完全一致。
    包含完整的异常值处理、角度unwrap、过滤和平滑处理，以及本体坐标系速度计算。
    
    参数:
        car_pose: 输入的car_pose数据，shape: (n, 3) [x, y, angle]
        outlier_threshold: 异常值检测阈值，默认为3
        jump_threshold: 跳变检测阈值，默认为1.0
        smooth_iterations: 平滑迭代次数，默认为3
        strong_smooth: 是否使用强平滑模式，默认为True
        
    返回:
        dict: 包含处理后的数据
            - 'base_velocity_decomposed': shape (n, 3) [vx_body, vy_body, vyaw] (本体坐标系)
            - 'valid': bool, 是否有效（通过速度范围检查）
    """
    # 定义速度限制（与data_analysis_filter.py中完全一致）
    velocity_limits = {
        'vx': {'min': -0.5, 'max': 0.5},
        'vy': {'min': -0.5, 'max': 0.5}, 
        'vyaw': {'min': -1.6, 'max': 1.6}
    }
    
    # 处理空数据或单点数据
    if len(car_pose) == 0:
        return {
            'base_velocity_decomposed': np.zeros((0, 3)),
            'valid': False
        }

    if len(car_pose) == 1:
        return {
            'base_velocity_decomposed': np.zeros((1, 3)),
            'valid': True  # 单点数据认为是有效的
        }

    # 🎯 步骤1: 提取位置和角度数据，并进行角度展开
    x_values = car_pose[:, 0].copy()
    y_values = car_pose[:, 1].copy()
    angle_values = car_pose[:, 2].copy()

    # 🎯 角度展开处理，避免跳变（移到函数内部）
    angle_values_unwrapped = np.unwrap(angle_values)

    # 🎯 步骤2-4: 异常值处理、跳变修正、平滑处理
    # 异常值处理
    x_filtered = remove_outliers(x_values, outlier_threshold)
    y_filtered = remove_outliers(y_values, outlier_threshold)
    angle_filtered = remove_outliers(angle_values_unwrapped, outlier_threshold)

    # 跳变修正
    x_filtered = remove_jumps(x_filtered, jump_threshold)
    y_filtered = remove_jumps(y_filtered, jump_threshold)
    angle_filtered = remove_jumps(angle_filtered, jump_threshold)

    # 平滑处理
    window_length = min(51 if strong_smooth else 21, len(x_filtered) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)

    x_smooth = smooth_data(x_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                          iterations=smooth_iterations, strong_smooth=strong_smooth)
    y_smooth = smooth_data(y_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                          iterations=smooth_iterations, strong_smooth=strong_smooth)
    angle_smooth = smooth_data(angle_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                              iterations=smooth_iterations, strong_smooth=strong_smooth)

    # 🎯 步骤5: 使用与data_processor.py完全一致的本体坐标系速度计算方法
    dt = 1/20  # 20Hz采样频率
    
    # 计算全局位移
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    angle_diff = np.diff(angle_smooth)
    
    # 获取当前帧的角度（用于坐标变换）
    current_theta = angle_smooth[:-1]  # shape: (n-1,)
    
    # 🎯 坐标变换：从全局坐标系到本体坐标系（与data_processor.py一致）
    cos_theta = np.cos(current_theta)
    sin_theta = np.sin(current_theta)
    
    # 🎯 计算本体坐标系下的速度
    vx_body = (x_diff * cos_theta + y_diff * sin_theta) / dt   # 前进速度（本体坐标系）
    vy_body = (-x_diff * sin_theta + y_diff * cos_theta) / dt  # 左侧速度（本体坐标系）
    vyaw = angle_diff / dt  # 角速度
    
    # 在开头添加零速度以匹配原始数据长度
    vx_array = np.concatenate([[0], vx_body])
    vy_array = np.concatenate([[0], vy_body])
    vyaw_array = np.concatenate([[0], vyaw])
    
    base_velocity_decomposed = np.stack([vx_array, vy_array, vyaw_array], axis=1)
    
    # 🎯 步骤6: 速度范围检查（与data_analysis_filter.py一致）
    valid = True

    if abs(x_values).max()>6 or abs(y_values).max()>6 or abs(angle_values).max()>6:
        valid = False
    
    # 检查每个速度分量是否在允许范围内
    if valid:
        for vx_val in vx_body:
            if vx_val < velocity_limits['vx']['min'] or vx_val > velocity_limits['vx']['max']:
                valid = False
                break
    
    if valid:  # 只有vx通过检查才继续检查vy
        for vy_val in vy_body:
            if vy_val < velocity_limits['vy']['min'] or vy_val > velocity_limits['vy']['max']:
                valid = False
                break
    
    if valid:  # 只有vx和vy都通过检查才继续检查vyaw
        for vyaw_val in vyaw:
            if vyaw_val < velocity_limits['vyaw']['min'] or vyaw_val > velocity_limits['vyaw']['max']:
                valid = False
                break

    return {
        'base_velocity_decomposed': base_velocity_decomposed,
        'valid': valid
    }
    


def remove_outliers(data, threshold=3):
    """
    使用Z-score方法移除异常值。
    
    参数:
        data: 输入数据数组
        threshold: Z-score阈值，默认为3
        
    返回:
        过滤后的数据
    """
    # 如果数据点太少或全为相同值，直接返回原始数据
    if len(data) < 3 or np.all(data == data[0]):
        return data.copy()

    # 计算标准差，避免灾难性抵消
    std = np.std(data)
    if std < 1e-10:  # 如果标准差接近0，直接返回原始数据
        return data.copy()
        
    # 计算Z-score
    try:
        z_scores = np.abs(stats.zscore(data))
    except:
        # 如果无法计算Z-score（例如所有值相同），直接返回原始数据
        return data.copy()
        
    filtered_data = data.copy()
    
    # 找出异常值
    mask = z_scores > threshold
    
    # 如果异常值太多，可能是数据本身就有较大波动，保持原样
    if np.sum(mask) > len(data) * 0.4:  # 如果超过40%的数据被标记为异常
        return data.copy()
        
    # 标记异常值为NaN
    filtered_data[mask] = np.nan
    
    # 检查是否所有值都变成了NaN
    if np.all(np.isnan(filtered_data)):
        return data.copy()
        
    # 用插值填充NaN值
    nan_mask = np.isnan(filtered_data)
    
    # 确保有非NaN值可以用于插值
    if np.any(~nan_mask):
        filtered_data[nan_mask] = np.interp(
            np.flatnonzero(nan_mask), 
            np.flatnonzero(~nan_mask), 
            filtered_data[~nan_mask]
        )
    
    return filtered_data



def remove_jumps(data, threshold=1.0):
    """
    检测并修正数据中的突然跳变。
    
    参数:
        data: 输入数据数组
        threshold: 跳变检测阈值，默认为1.0
        
    返回:
        修正后的数据
    """
    # 如果数据点太少，无法有效检测跳变，直接返回原始数据
    if len(data) < 3:
        return data.copy()
        
    result = data.copy()
    
    # 计算相邻点之间的差值
    try:
        diffs = np.abs(np.diff(result))
    except:
        # 如果无法计算差值，直接返回原始数据
        return data.copy()
        
    # 找出大于阈值的跳变点
    jump_indices = np.where(diffs > threshold)[0]
    
    # 如果跳变点太多，说明数据本身波动较大，保持原样
    if len(jump_indices) > len(data) * 0.3:  # 如果超过30%的点被标记为跳变
        return data.copy()
    
    # 处理每个跳变点
    for idx in jump_indices:
        # 用前后点的平均值替换跳变
        if idx > 0 and idx < len(result) - 1:
            # 取跳变前后的平均值
            result[idx+1] = (result[idx] + result[idx+2]) / 2
        elif idx == len(result) - 2:  # 如果是倒数第二个点
            result[idx+1] = result[idx]  # 用前一个点的值替换
    
    return result



def smooth_data(data, window_length=None, polyorder=3, iterations=1, strong_smooth=False):
    """
    使用Savitzky-Golay滤波器平滑数据。
    
    参数:
        data: 输入数据数组
        window_length: 窗口长度，如果未指定则自动计算
        polyorder: 多项式阶数，默认为3
        iterations: 平滑迭代次数，默认为1
        strong_smooth: 是否使用强平滑模式，默认为False
        
    返回:
        平滑后的数据
    """
    # 确保数据至少有3个点
    if len(data) < 3:
        return data.copy()
        
    # 计算合适的窗口长度
    if window_length is None:
        if strong_smooth:
            # 强平滑模式下使用更大的窗口
            window_length = min(51, len(data) - 1)
        else:
            window_length = min(21, len(data) - 1)
        
    # 确保窗口长度为奇数且小于等于数据长度
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:  # 确保窗口长度为奇数
        window_length -= 1
    
    # 确保窗口长度至少为3
    window_length = max(3, window_length)
    
    # 如果数据长度小于窗口长度，使用高斯滤波
    if window_length >= len(data):
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)
    
    # 在强平滑模式下使用更低的多项式阶数
    if strong_smooth:
        polyorder = min(2, polyorder)
    
    # 确保多项式阶数小于窗口长度
    polyorder = min(polyorder, window_length - 1)
    
    smooth_data_result = data.copy()
    
    try:
        # 应用多次平滑
        for _ in range(iterations):
            smooth_data_result = savgol_filter(smooth_data_result, window_length, polyorder)
            
        # 如果需要强平滑，再应用高斯滤波
        if strong_smooth:
            smooth_data_result = gaussian_filter1d(smooth_data_result, sigma=2.0)
            
        return smooth_data_result
        
    except Exception as e:
        # 如果savgol_filter失败，使用高斯滤波作为备选
        print(f"Savgol filter failed: {e}, using Gaussian filter instead")
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)
    

def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]

        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img

    return images


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if features[key]["dtype"] in ["image", "video"]:
            value_norm = 1.0 if "depth" in key else 255.0
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / value_norm, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats

def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            continue
            # ep_ft_array = sample_images(data)  # data is a list of image paths
            # axes_to_reduce = (0, 2, 3)  # keep channel dim
            # keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats





class X2LeRobotDataset(LeRobotDataset):

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        # validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        print("task", task, flush=True)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                # img_path = self._get_image_file_path(
                #     episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                # )
                # if frame_index == 0:
                #     img_path.parent.mkdir(parents=True, exist_ok=True)
                # self._save_image(frame[key], img_path)
                img_path = frame[key]
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1
    
    def save_episode(self, videos: dict, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        print("tasks", tasks, flush=True)
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = [str(video_path)]  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        self._save_episode_table(episode_buffer, episode_index)

        # `meta.save_episode` be executed after encoding the videos
        # add action_config to current episode
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()


def load_local_dataset(
        episode_item
    ):
    frames = []
    videos = {}

    cam_mapping = episode_item["cam_mapping"]
    folder_path = episode_item["path"]

    # 加载所有配置摄像头中的视频数据
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # filename = file.split(".")[0]
            filename = ".".join(file.split(".")[:-1])
            if file.endswith(".mp4") and filename in cam_mapping.keys():
                videos[f"observation.images.{filename}"] = os.path.join(root, file)

    # 创建原始键和模型键的映射
    action_keys_raw2predict = {_ACTION_KEY_FULL_MAPPING[k]: k for k in episode_item["predict_action_keys"]}
    action_keys_raw2obs = {_ACTION_KEY_FULL_MAPPING[k]: k for k in episode_item["obs_action_keys"]}

    # 从文件处理动作数据
    action_data, action_data_obs = process_action(
        folder_path, raw_key2model_key=[action_keys_raw2predict, action_keys_raw2obs], filter_angle_outliers=True
    )

    action = [action_data[key] if action_data[key].ndim == 2 else action_data[key][:, None] for key in episode_item["predict_action_keys"]]
    action = np.concatenate(action, axis=1)

    obs = [action_data_obs[key] if action_data_obs[key].ndim == 2 else action_data_obs[key][:, None] for key in episode_item["obs_action_keys"]]
    obs = np.concatenate(obs, axis=1)

    instruction_info = episode_item["instruction_info"]

    frames = [
        {
            # "timestamp": i / 20,
            "action": action[i+1],
            "observation.state": obs[i],
            "observation.images.faceImg": [videos[f"observation.images.faceImg"]],
            #"observation.images.leftImg": [videos[f"observation.images.leftImg"]],
            "observation.images.rightImg": [videos[f"observation.images.rightImg"]],
        }
        for i in range(action.shape[0]-1)
    ]
    

    return frames, videos

def main(
    src_path: str,
    output_path: str,
    fps: int,
    repo_id: str
):  
    dataset = X2LeRobotDataset.create(
        repo_id = repo_id,
        root = output_path,
        features={"action": {"dtype": "float32", "shape": (7,), "names": None},
                  "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
                  "observation.images.faceImg": {"dtype": "video", "shape": (256, 256, 3), "names": None},
                  "observation.images.rightImg": {"dtype": "video", "shape": (256, 256, 3), "names": None},
                  },
        fps = fps,
    )
    
    # 遍历src_path下的所有文件夹获取episode path
    episode_paths = []
    if src_path and os.path.exists(src_path):
        for item in os.listdir(src_path):
            if item != "record":
                item_path = os.path.join(src_path, item)
                if os.path.isdir(item_path):
                    episode_paths.append(item_path)

        print(f"Found {len(episode_paths)} episode folders in {src_path}")
        for path in episode_paths:
            print(f"  - {path}")
    else:
        # 如果没有提供src_path或路径不存在，使用默认路径
        ValueError("Please provide a valid src_path")

    instructions_file = os.path.join(src_path, "instruction.json")
    with open(instructions_file, "r") as f:
        instruction_dict = json.load(f)
    
    # 处理每个episode
    for episode_path in episode_paths:
        print(f"\nProcessing episode: {episode_path}")
        episode_item = {
            'path': episode_path,
            'cam_mapping': {'faceImg': 'face_view', 'leftImg': 'left_wrist_view', 'rightImg': 'right_wrist_view'},
            'type': 'x2_normal',
            'predict_action_keys': [
                      'master_right_ee_cartesian_pos','master_right_ee_rotation','follow_right_gripper'],
            'obs_action_keys': [
                        'follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper'],
            'instruction_info': instruction_dict.get(os.path.basename(episode_path), "")["instruction"],
        }

        frames, videos = load_local_dataset(episode_item)
        print(f"Loaded {len(frames)} frames and {len(videos)} videos from {episode_path}")

        
        for frame in frames:
            dataset.add_frame(frame, task=episode_item["instruction_info"])


        dataset.save_episode(videos)

        # break


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args),fps=15, repo_id="libero_all")

# python x2robot2lerobot.py --src-path ./20250919_2arms_processed --output-path ./20250919_2arms_processed_lerobot

    