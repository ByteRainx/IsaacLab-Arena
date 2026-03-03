#!/usr/bin/env python3
"""Visualize robot URDF mesh + EE trajectories in interactive 3D (plotly).

Loads the ex001 bimanual robot URDF with STL meshes, renders at home
configuration, and overlays all episode EE trajectories from HDF5 files.

Output: interactive HTML file openable in any browser.

Usage:
    python tools/visualize_ee_trajectories.py
    python tools/visualize_ee_trajectories.py --max_episodes 10
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import trimesh


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _rpy_to_matrix(rpy):
    """Roll-pitch-yaw (XYZ intrinsic) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _quat_to_matrix(quat_wxyz):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _axis_angle_matrix(axis, angle):
    """Rotation matrix from axis-angle (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])


def _make_tf(rot, pos):
    """Build 4x4 homogeneous transform from 3x3 rotation and 3-vector."""
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


# ── URDF Parsing ─────────────────────────────────────────────────────────────

class URDFParser:
    """Lightweight URDF parser for FK and mesh extraction."""

    def __init__(self, urdf_path: str):
        self.urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        self.links = {}
        self.joints = []
        self.children = {}   # parent_link -> [(joint, child_link)]
        self.parent_of = {}  # child_link -> parent_link

        for link_el in root.findall('link'):
            name = link_el.get('name')
            mesh_path = None
            mesh_scale = None
            vis = link_el.find('.//visual/geometry/mesh')
            if vis is not None:
                mesh_path = vis.get('filename')
                scale_str = vis.get('scale')
                if scale_str:
                    mesh_scale = [float(v) for v in scale_str.split()]

            vis_origin = link_el.find('.//visual/origin')
            vis_xyz = [0, 0, 0]
            vis_rpy = [0, 0, 0]
            if vis_origin is not None:
                vis_xyz = [float(v) for v in vis_origin.get('xyz', '0 0 0').split()]
                vis_rpy = [float(v) for v in vis_origin.get('rpy', '0 0 0').split()]

            self.links[name] = {
                'mesh_path': mesh_path,
                'mesh_scale': mesh_scale,
                'vis_xyz': vis_xyz,
                'vis_rpy': vis_rpy,
            }

        seen_joints = set()
        for joint_el in root.findall('joint'):
            jname = joint_el.get('name')
            jtype = joint_el.get('type')
            if jtype is None:
                continue

            parent_el = joint_el.find('parent')
            child_el = joint_el.find('child')
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.get('link')
            child = child_el.get('link')
            if not parent or not child:
                continue

            key = (jname, parent, child)
            if key in seen_joints:
                continue
            seen_joints.add(key)

            origin_el = joint_el.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin_el is not None:
                xyz = [float(v) for v in origin_el.get('xyz', '0 0 0').split()]
                rpy = [float(v) for v in origin_el.get('rpy', '0 0 0').split()]

            axis_el = joint_el.find('axis')
            axis = [0, 0, 1]
            if axis_el is not None:
                axis = [float(v) for v in axis_el.get('xyz', '0 0 1').split()]

            joint_info = {
                'name': jname, 'type': jtype,
                'parent': parent, 'child': child,
                'xyz': xyz, 'rpy': rpy, 'axis': axis,
            }
            self.joints.append(joint_info)
            self.children.setdefault(parent, []).append(joint_info)
            self.parent_of[child] = parent

    def compute_fk(self, joint_values: dict | None = None, base_tf=None):
        """Compute forward kinematics. Returns dict: link_name -> 4x4 transform."""
        if joint_values is None:
            joint_values = {}
        if base_tf is None:
            base_tf = np.eye(4)

        root_links = set(self.links.keys()) - set(self.parent_of.keys())
        link_tfs = {}
        for rl in root_links:
            link_tfs[rl] = base_tf.copy()

        def _walk(link_name):
            if link_name not in self.children:
                return
            for jinfo in self.children[link_name]:
                child = jinfo['child']
                parent_tf = link_tfs[link_name]

                rot_fixed = _rpy_to_matrix(jinfo['rpy'])
                tf_joint_origin = _make_tf(rot_fixed, jinfo['xyz'])

                q = joint_values.get(jinfo['name'], 0.0)
                if jinfo['type'] == 'revolute' or jinfo['type'] == 'continuous':
                    rot_q = _axis_angle_matrix(jinfo['axis'], q)
                    tf_q = _make_tf(rot_q, [0, 0, 0])
                elif jinfo['type'] == 'prismatic':
                    disp = np.array(jinfo['axis']) * q
                    tf_q = _make_tf(np.eye(3), disp)
                else:
                    tf_q = np.eye(4)

                link_tfs[child] = parent_tf @ tf_joint_origin @ tf_q
                _walk(child)

        for rl in root_links:
            _walk(rl)
        return link_tfs

    def load_meshes(self, link_tfs: dict, link_filter=None):
        """Load and transform STL meshes. Returns list of (vertices, faces) tuples."""
        meshes = []
        for link_name, tf in link_tfs.items():
            info = self.links.get(link_name)
            if info is None or info['mesh_path'] is None:
                continue
            if link_filter and not link_filter(link_name):
                continue

            mesh_file = os.path.join(self.urdf_dir, info['mesh_path'])
            if not os.path.exists(mesh_file):
                continue

            try:
                mesh = trimesh.load(mesh_file, force='mesh')
            except Exception:
                continue

            vis_rot = _rpy_to_matrix(info['vis_rpy'])
            vis_tf = _make_tf(vis_rot, info['vis_xyz'])

            verts = np.asarray(mesh.vertices)
            if info['mesh_scale']:
                verts = verts * np.array(info['mesh_scale'])

            ones = np.ones((len(verts), 1))
            verts_h = np.hstack([verts, ones])
            world_tf = tf @ vis_tf
            verts_world = (world_tf @ verts_h.T).T[:, :3]

            meshes.append({
                'name': link_name,
                'vertices': verts_world,
                'faces': np.asarray(mesh.faces),
            })
        return meshes


# ── Data loading ─────────────────────────────────────────────────────────────

def natural_sort_key(path: str) -> list:
    basename = os.path.basename(str(path))
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]


def load_ee_trajectories(data_dir: str, max_episodes: int | None = None, skip_n: int = 2):
    """Load EE trajectories from HDF5 files."""
    pattern = os.path.join(data_dir, '*.hdf5')
    files = sorted(glob.glob(pattern), key=natural_sort_key)
    if max_episodes:
        files = files[:max_episodes]

    left_trajs = []
    right_trajs = []
    first_joint_pos = None

    for fp in files:
        with h5py.File(fp, 'r') as f:
            data_group = f['data']
            demo_key = sorted(data_group.keys())[0]
            obs = data_group[demo_key]['obs']

            left_trajs.append(obs['eef_pos'][skip_n:])
            right_trajs.append(obs['right_eef_pos'][skip_n:])

            if first_joint_pos is None:
                first_joint_pos = obs['joint_pos'][0][:]

    return left_trajs, right_trajs, first_joint_pos


# ── Plotly figure building ───────────────────────────────────────────────────

def _add_robot_meshes(fig, meshes, opacity=0.25):
    """Add robot mesh surfaces to the figure."""
    for m in meshes:
        v = m['vertices']
        f = m['faces']
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color='lightgray', opacity=opacity,
            name=m['name'],
            showlegend=False,
            hoverinfo='name',
        ))


def _add_trajectories(fig, trajs, side: str, colorscale_name: str):
    """Add EE trajectories as lines with start/end markers."""
    n = len(trajs)
    import plotly.express as px
    if colorscale_name == 'warm':
        colors = px.colors.sample_colorscale('YlOrRd', [i / max(n - 1, 1) for i in range(n)])
    else:
        colors = px.colors.sample_colorscale('PuBu', [i / max(n - 1, 1) for i in range(n)])

    starts_x, starts_y, starts_z = [], [], []
    ends_x, ends_y, ends_z = [], [], []
    starts_text, ends_text = [], []

    for i, traj in enumerate(trajs):
        color = colors[i]
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(width=2, color=color),
            name=f'{side} ep{i}',
            legendgroup=side,
            showlegend=(i == 0),
            legendgrouptitle_text=f'{side} arm',
            hovertemplate=f'{side} ep{i}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>',
        ))

        starts_x.append(traj[0, 0])
        starts_y.append(traj[0, 1])
        starts_z.append(traj[0, 2])
        starts_text.append(f'{side} ep{i} start')
        ends_x.append(traj[-1, 0])
        ends_y.append(traj[-1, 1])
        ends_z.append(traj[-1, 2])
        ends_text.append(f'{side} ep{i} end')

    fig.add_trace(go.Scatter3d(
        x=starts_x, y=starts_y, z=starts_z,
        mode='markers', marker=dict(size=4, color='lime', symbol='circle'),
        name=f'{side} starts', legendgroup=side, showlegend=False,
        text=starts_text, hoverinfo='text',
    ))
    fig.add_trace(go.Scatter3d(
        x=ends_x, y=ends_y, z=ends_z,
        mode='markers', marker=dict(size=4, color='red', symbol='diamond'),
        name=f'{side} ends', legendgroup=side, showlegend=False,
        text=ends_text, hoverinfo='text',
    ))


def build_figure(meshes, left_trajs, right_trajs):
    """Build the complete 3D plotly figure."""
    fig = go.Figure()

    _add_robot_meshes(fig, meshes)
    _add_trajectories(fig, left_trajs, 'Left', 'warm')
    _add_trajectories(fig, right_trajs, 'Right', 'cool')

    fig.update_layout(
        title='EE Trajectories + Robot Mesh (Home Pose)',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1400, height=900,
    )
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize EE trajectories with robot mesh.')
    parser.add_argument('--input_dir', type=str,
                        default='data/ex001arm_cvpr_scene_put_blocks_to_color')
    parser.add_argument('--urdf_path', type=str,
                        default=os.path.expanduser(
                            '~/Downloads/ex001_robot_description-v00.02.01/ex001_robot.urdf'))
    parser.add_argument('--output', type=str, default='data/ee_trajectories_3d.html')
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--robot_opacity', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load EE trajectories
    print(f'Loading EE trajectories from: {args.input_dir}')
    left_trajs, right_trajs, home_joint_pos = load_ee_trajectories(
        args.input_dir, max_episodes=args.max_episodes,
    )
    print(f'  Episodes: {len(left_trajs)}')
    print(f'  Home joint_pos (18): {home_joint_pos}')

    # 2. Parse URDF and compute FK
    print(f'Parsing URDF: {args.urdf_path}')
    urdf = URDFParser(args.urdf_path)

    # Map the 18-dim joint_pos to URDF joint names.
    joint_name_order = [
        'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
        'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6',
        'left_arm_gripper', 'left_arm_gripper_left_joint', 'left_arm_gripper_right_joint',
        'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3',
        'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6',
        'right_arm_gripper', 'right_arm_gripper_left_joint', 'right_arm_gripper_right_joint',
    ]

    joint_values = {}
    for i, jname in enumerate(joint_name_order):
        if i < len(home_joint_pos):
            joint_values[jname] = float(home_joint_pos[i])

    # The sim places the robot at init_state.pos. The HDF5 eef_pos is in world
    # frame (target_pos_w - env_origins). Verified: the pure translation offset
    # from URDF FK to HDF5 eef_pos matches init_state.pos exactly for both arms.
    SIM_BASE_POS = np.array([-0.51676, -0.25918, -0.58061])
    base_tf = _make_tf(np.eye(3), SIM_BASE_POS)

    link_tfs = urdf.compute_fk(joint_values, base_tf=base_tf)

    # Only render arm-related links (skip cameras, head, wheels, etc.)
    arm_link_keywords = [
        'arm_base', 'arm_link', 'arm_frame', 'gripper_base',
        'lift_link', 'chassis_base',
    ]

    def link_filter(name):
        return any(kw in name for kw in arm_link_keywords)

    print('Loading STL meshes...')
    meshes = urdf.load_meshes(link_tfs, link_filter=link_filter)
    print(f'  Loaded {len(meshes)} meshes: {[m["name"] for m in meshes]}')

    # 3. Build figure
    print('Building 3D figure...')
    fig = build_figure(meshes, left_trajs, right_trajs)

    # 4. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f'Saved interactive HTML: {output_path.resolve()}')
    print('Open in a browser to explore the trajectories.')


if __name__ == '__main__':
    main()
