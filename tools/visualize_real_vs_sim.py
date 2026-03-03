#!/usr/bin/env python3
"""Real-vs-Sim EE trajectory comparison with proper alignment.

Both real and sim data are 14D EE-space:
  [left_xyz(3), left_rpy(3), left_grip(1),
   right_xyz(3), right_rpy(3), right_grip(1)]

Alignment:
  - Real data is already delta-from-home (XYZ near 0 at start).
  - Sim data is in absolute env-frame; we subtract frame-0 to get delta-from-home.
  - XYZ axes are naturally aligned (verified via teleop code & statistics).
  - RPY is wrapped to [-π, π] after delta computation.
  - Gripper is normalized to [0, 1].

Usage:
    python tools/visualize_real_vs_sim.py
    python tools/visualize_real_vs_sim.py --max_episodes 80
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import trimesh
from plotly.subplots import make_subplots

DIM_LABELS = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Grip']
REAL_GRIP_MAX = 4.8
SIM_GRIP_MAX = 0.086

DEFAULT_URDF = '/home/xr/Downloads/ex001_robot_description-v00.02.01/ex001_robot.urdf'


# ── Rotation / transform helpers ──────────────────────────────────────────────

def _rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_angle_matrix(axis, angle):
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
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


# ── URDF parser (lightweight) ────────────────────────────────────────────────

class URDFParser:
    def __init__(self, urdf_path: str):
        self.urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        self.links = {}
        self.joints = []
        self.children = {}
        self.parent_of = {}

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
                'mesh_path': mesh_path, 'mesh_scale': mesh_scale,
                'vis_xyz': vis_xyz, 'vis_rpy': vis_rpy,
            }

        seen = set()
        for j_el in root.findall('joint'):
            jname = j_el.get('name')
            jtype = j_el.get('type')
            if jtype is None:
                continue
            p_el, c_el = j_el.find('parent'), j_el.find('child')
            if p_el is None or c_el is None:
                continue
            parent, child = p_el.get('link'), c_el.get('link')
            if not parent or not child:
                continue
            key = (jname, parent, child)
            if key in seen:
                continue
            seen.add(key)
            o_el = j_el.find('origin')
            xyz = [float(v) for v in o_el.get('xyz', '0 0 0').split()] if o_el is not None else [0, 0, 0]
            rpy = [float(v) for v in o_el.get('rpy', '0 0 0').split()] if o_el is not None else [0, 0, 0]
            a_el = j_el.find('axis')
            axis = [float(v) for v in a_el.get('xyz', '0 0 1').split()] if a_el is not None else [0, 0, 1]
            ji = {'name': jname, 'type': jtype, 'parent': parent,
                  'child': child, 'xyz': xyz, 'rpy': rpy, 'axis': axis}
            self.joints.append(ji)
            self.children.setdefault(parent, []).append(ji)
            self.parent_of[child] = parent

    def compute_fk(self, joint_values=None, base_tf=None):
        if joint_values is None:
            joint_values = {}
        if base_tf is None:
            base_tf = np.eye(4)
        root_links = set(self.links.keys()) - set(self.parent_of.keys())
        link_tfs = {rl: base_tf.copy() for rl in root_links}

        def _walk(ln):
            if ln not in self.children:
                return
            for ji in self.children[ln]:
                ch = ji['child']
                ptf = link_tfs[ln]
                tf_o = _make_tf(_rpy_to_matrix(ji['rpy']), ji['xyz'])
                q = joint_values.get(ji['name'], 0.0)
                if ji['type'] in ('revolute', 'continuous'):
                    tf_q = _make_tf(_axis_angle_matrix(ji['axis'], q), [0, 0, 0])
                elif ji['type'] == 'prismatic':
                    tf_q = _make_tf(np.eye(3), np.array(ji['axis']) * q)
                else:
                    tf_q = np.eye(4)
                link_tfs[ch] = ptf @ tf_o @ tf_q
                _walk(ch)

        for rl in root_links:
            _walk(rl)
        return link_tfs

    def load_meshes(self, link_tfs, link_filter=None):
        meshes = []
        for ln, tf in link_tfs.items():
            info = self.links.get(ln)
            if info is None or info['mesh_path'] is None:
                continue
            if link_filter and not link_filter(ln):
                continue
            mf = os.path.join(self.urdf_dir, info['mesh_path'])
            if not os.path.exists(mf):
                continue
            try:
                mesh = trimesh.load(mf, force='mesh')
            except Exception:
                continue
            vis_tf = _make_tf(_rpy_to_matrix(info['vis_rpy']), info['vis_xyz'])
            verts = np.asarray(mesh.vertices)
            if info['mesh_scale']:
                verts = verts * np.array(info['mesh_scale'])
            vh = np.hstack([verts, np.ones((len(verts), 1))])
            vw = (tf @ vis_tf @ vh.T).T[:, :3]
            meshes.append({'name': ln, 'vertices': vw, 'faces': np.asarray(mesh.faces)})
        return meshes


def natural_sort_key(path):
    basename = os.path.basename(str(path))
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]


def wrap_angle(a):
    """Wrap angles to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── Data loading & alignment ─────────────────────────────────────────────────

def load_episodes(parquet_dir, max_episodes=None, skip_first=2, key='observation.state'):
    files = sorted(
        glob.glob(os.path.join(parquet_dir, 'data/chunk-*/episode_*.parquet')),
        key=natural_sort_key,
    )
    if max_episodes:
        files = files[:max_episodes]
    episodes = []
    for f in files:
        df = pd.read_parquet(f)
        data = np.stack(df[key].values)[skip_first:]
        episodes.append(data)
    return episodes


def align_episodes(episodes, is_real=True):
    """Align episodes to delta-from-home, wrap RPY, normalize gripper.

    Returns list of 14D aligned episodes.
    """
    aligned = []
    for ep in episodes:
        home = ep[0].copy()
        delta = ep - home

        for rpy_start in [3, 10]:
            delta[:, rpy_start:rpy_start + 3] = wrap_angle(delta[:, rpy_start:rpy_start + 3])

        for g_idx in [6, 13]:
            if is_real:
                delta[:, g_idx] = ep[:, g_idx] / REAL_GRIP_MAX
            else:
                delta[:, g_idx] = ep[:, g_idx] / SIM_GRIP_MAX if SIM_GRIP_MAX > 0 else 0.0

        aligned.append(delta)
    return aligned


# ── Visualization ─────────────────────────────────────────────────────────────

def _add_robot_meshes(fig, meshes, opacity=0.2):
    for m in meshes:
        v, f = m['vertices'], m['faces']
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color='lightgray', opacity=opacity,
            name=m['name'], showlegend=False, hoverinfo='name',
        ))


def _add_traj_set(fig, trajs_xyz, label, colorscale, legendgroup, opacity=0.7):
    n = len(trajs_xyz)
    if n == 0:
        return
    colors = px.colors.sample_colorscale(colorscale, [i / max(n - 1, 1) for i in range(n)])
    for i, traj in enumerate(trajs_xyz):
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(width=2, color=colors[i]),
            opacity=opacity, name=f'{label} ep{i}',
            legendgroup=legendgroup, showlegend=(i == 0),
            legendgrouptitle_text=label,
        ))


def load_robot_meshes(urdf_path, robot_root_pos=None):
    """Load URDF meshes at home configuration with robot root offset."""
    parser = URDFParser(urdf_path)
    if robot_root_pos is not None:
        base_tf = _make_tf(np.eye(3), np.array(robot_root_pos))
    else:
        base_tf = np.eye(4)
    link_tfs = parser.compute_fk(joint_values={}, base_tf=base_tf)
    meshes = parser.load_meshes(link_tfs)
    return meshes


def build_3d_comparison(real_left, real_right, sim_left, sim_right, title,
                        meshes=None):
    fig = go.Figure()
    if meshes:
        _add_robot_meshes(fig, meshes, opacity=0.15)
    _add_traj_set(fig, sim_left, 'Sim Left', 'Greens', 'sim_left', opacity=0.8)
    _add_traj_set(fig, sim_right, 'Sim Right', 'Blues', 'sim_right', opacity=0.8)
    _add_traj_set(fig, real_left, 'Real Left', 'YlOrRd', 'real_left', opacity=0.5)
    _add_traj_set(fig, real_right, 'Real Right', 'Purples', 'real_right', opacity=0.5)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=40, b=0), width=1400, height=900,
    )
    return fig


def build_distribution_comparison(real_all, sim_all, dims, dim_labels, title,
                                  unit='', nbins=100, rows=2, cols=3):
    """Overlaid histogram comparison for selected dimensions."""
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=dim_labels,
                        vertical_spacing=0.12, horizontal_spacing=0.06)
    for idx, d in enumerate(dims):
        r, c = idx // cols + 1, idx % cols + 1
        fig.add_trace(go.Histogram(
            x=real_all[:, d], name='Real', marker_color='rgba(239,85,59,0.5)',
            legendgroup='Real', showlegend=(idx == 0), nbinsx=nbins,
        ), row=r, col=c)
        fig.add_trace(go.Histogram(
            x=sim_all[:, d], name='Sim', marker_color='rgba(99,110,250,0.5)',
            legendgroup='Sim', showlegend=(idx == 0), nbinsx=nbins,
        ), row=r, col=c)
        fig.update_xaxes(title_text=unit, row=r, col=c)
    fig.update_layout(title=title, barmode='overlay',
                      height=280 * rows, width=380 * cols)
    return fig


def build_per_dim_boxplot(real_all, sim_all):
    fig = make_subplots(
        rows=2, cols=7,
        subplot_titles=[f'L_{l}' for l in DIM_LABELS] + [f'R_{l}' for l in DIM_LABELS],
        vertical_spacing=0.12, horizontal_spacing=0.04,
    )
    for d in range(14):
        r, c = d // 7 + 1, d % 7 + 1
        fig.add_trace(go.Box(
            y=real_all[:, d], name='Real', marker_color='#EF553B',
            legendgroup='Real', showlegend=(d == 0), boxmean='sd',
        ), row=r, col=c)
        fig.add_trace(go.Box(
            y=sim_all[:, d], name='Sim', marker_color='#636EFA',
            legendgroup='Sim', showlegend=(d == 0), boxmean='sd',
        ), row=r, col=c)
    fig.update_layout(
        title='Per-Dimension Distribution (aligned delta-from-home, grip normalized)',
        height=700, width=1600,
    )
    return fig


def build_time_series(real_ep, sim_ep, real_fps, sim_fps, ep_idx=0):
    """Side-by-side time series for one episode."""
    fig = make_subplots(
        rows=7, cols=2,
        subplot_titles=[f'Left {l}' for l in DIM_LABELS] + [f'Right {l}' for l in DIM_LABELS],
        vertical_spacing=0.03, horizontal_spacing=0.06,
    )
    real_t = np.arange(len(real_ep)) / real_fps
    sim_t = np.arange(len(sim_ep)) / sim_fps

    for d in range(7):
        for col_off, arm_off in [(1, 0), (2, 7)]:
            fig.add_trace(go.Scatter(
                x=real_t, y=real_ep[:, arm_off + d], mode='lines',
                name='Real', line=dict(color='#EF553B', width=1.5),
                legendgroup='Real', showlegend=(d == 0 and col_off == 1),
            ), row=d + 1, col=col_off)
            fig.add_trace(go.Scatter(
                x=sim_t, y=sim_ep[:, arm_off + d], mode='lines',
                name='Sim', line=dict(color='#636EFA', width=1.5),
                legendgroup='Sim', showlegend=(d == 0 and col_off == 1),
            ), row=d + 1, col=col_off)

    fig.update_layout(
        title=f'Time-series: Episode {ep_idx} (aligned, delta-from-home, grip normalized)',
        height=1400, width=1200,
    )
    for r in range(1, 8):
        for c in [1, 2]:
            fig.update_xaxes(title_text='time (s)' if r == 7 else '', row=r, col=c)
    return fig


def build_workspace_heatmap(real_all, sim_all):
    """2D XY / XZ / YZ workspace heatmaps for both arms."""
    planes = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f'Left {a}-{b}' for a, b, _, _ in planes
        ] + [
            f'Right {a}-{b}' for a, b, _, _ in planes
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for idx, (ax, ay, di, dj) in enumerate(planes):
        for row_off, arm_off, src, name, color in [
            (0, 0, real_all, 'Real', 'rgba(239,85,59,0.3)'),
            (0, 0, sim_all, 'Sim', 'rgba(99,110,250,0.3)'),
            (1, 7, real_all, 'Real', 'rgba(239,85,59,0.3)'),
            (1, 7, sim_all, 'Sim', 'rgba(99,110,250,0.3)'),
        ]:
            xs = src[:, arm_off + di]
            ys = src[:, arm_off + dj]
            sample = np.random.choice(len(xs), min(3000, len(xs)), replace=False)
            fig.add_trace(go.Scatter(
                x=xs[sample], y=ys[sample], mode='markers',
                marker=dict(size=2, color=color),
                name=name, legendgroup=name,
                showlegend=(idx == 0 and row_off == 0 and name == 'Real') or
                           (idx == 0 and row_off == 0 and name == 'Sim'),
            ), row=row_off + 1, col=idx + 1)
        fig.update_xaxes(title_text=f'Δ{ax} (m)', row=1, col=idx + 1)
        fig.update_xaxes(title_text=f'Δ{ax} (m)', row=2, col=idx + 1)
        fig.update_yaxes(title_text=f'Δ{ay} (m)', row=1, col=idx + 1)
        fig.update_yaxes(title_text=f'Δ{ay} (m)', row=2, col=idx + 1)

    fig.update_layout(
        title='Workspace Scatter: Real vs Sim (delta-from-home, sampled 3k pts)',
        height=700, width=1400,
    )
    return fig


# ── Numerical analysis ────────────────────────────────────────────────────────

def print_analysis(real_eps, sim_eps, real_fps, sim_fps):
    sep = '=' * 80
    print(f'\n{sep}')
    print('REAL vs SIM — ALIGNED COMPARISON REPORT')
    print(f'Alignment: delta-from-home | RPY wrapped [-π,π] | Gripper normalized [0,1]')
    print(sep)

    real_lens = [len(e) for e in real_eps]
    sim_lens = [len(e) for e in sim_eps]
    print(f'\n--- Dataset Overview ---')
    print(f'  Real: {len(real_eps)} episodes, {real_fps} FPS, '
          f'mean {np.mean(real_lens)/real_fps:.1f}s '
          f'[{min(real_lens)/real_fps:.1f}s – {max(real_lens)/real_fps:.1f}s]')
    print(f'  Sim:  {len(sim_eps)} episodes, {sim_fps} FPS, '
          f'mean {np.mean(sim_lens)/sim_fps:.1f}s '
          f'[{min(sim_lens)/sim_fps:.1f}s – {max(sim_lens)/sim_fps:.1f}s]')
    print(f'  Real total frames: {sum(real_lens):,}')
    print(f'  Sim  total frames: {sum(sim_lens):,}')

    real_all = np.concatenate(real_eps)
    sim_all = np.concatenate(sim_eps)
    labels_full = [f'L_{l}' for l in DIM_LABELS] + [f'R_{l}' for l in DIM_LABELS]

    print(f'\n--- Per-Dimension Statistics ---')
    print(f'{"Dim":<10} {"Real mean":>10} {"Real std":>10} '
          f'{"Sim mean":>10} {"Sim std":>10} {"Δmean":>10} {"std ratio":>10}')
    for d in range(14):
        rm, rs = real_all[:, d].mean(), real_all[:, d].std()
        sm, ss = sim_all[:, d].mean(), sim_all[:, d].std()
        sr = rs / ss if ss > 1e-6 else float('inf')
        print(f'{labels_full[d]:<10} {rm:>+10.4f} {rs:>10.4f} '
              f'{sm:>+10.4f} {ss:>10.4f} {abs(rm - sm):>10.4f} {sr:>10.2f}')

    print(f'\n--- XYZ Workspace Extent ---')
    for arm, off in [('Left', 0), ('Right', 7)]:
        r_ext = real_all[:, off:off + 3].max(0) - real_all[:, off:off + 3].min(0)
        s_ext = sim_all[:, off:off + 3].max(0) - sim_all[:, off:off + 3].min(0)
        print(f'  {arm}:')
        for i, ax in enumerate(['X', 'Y', 'Z']):
            ratio = r_ext[i] / s_ext[i] if s_ext[i] > 1e-6 else float('inf')
            print(f'    {ax}: Real {r_ext[i]*1000:6.1f} mm | Sim {s_ext[i]*1000:6.1f} mm | ratio {ratio:.2f}')

    print(f'\n--- RPY Range (radians) ---')
    for arm, off in [('Left', 3), ('Right', 10)]:
        r_ext = real_all[:, off:off + 3].max(0) - real_all[:, off:off + 3].min(0)
        s_ext = sim_all[:, off:off + 3].max(0) - sim_all[:, off:off + 3].min(0)
        print(f'  {arm}:')
        for i, ax in enumerate(['Roll', 'Pitch', 'Yaw']):
            ratio = r_ext[i] / s_ext[i] if s_ext[i] > 1e-6 else float('inf')
            print(f'    {ax}: Real {r_ext[i]:.3f} rad | Sim {s_ext[i]:.3f} rad | ratio {ratio:.2f}')

    print(f'\n--- Gripper (normalized [0,1]) ---')
    for arm, idx in [('Left', 6), ('Right', 13)]:
        r_rng = real_all[:, idx].min(), real_all[:, idx].max()
        s_rng = sim_all[:, idx].min(), sim_all[:, idx].max()
        print(f'  {arm}: Real [{r_rng[0]:.3f}, {r_rng[1]:.3f}] | '
              f'Sim [{s_rng[0]:.3f}, {s_rng[1]:.3f}]')

    print(f'\n--- Workspace Overlap (IoU per XYZ axis) ---')
    for arm, off in [('Left', 0), ('Right', 7)]:
        ious = []
        for i, ax in enumerate(['X', 'Y', 'Z']):
            r_lo, r_hi = real_all[:, off + i].min(), real_all[:, off + i].max()
            s_lo, s_hi = sim_all[:, off + i].min(), sim_all[:, off + i].max()
            overlap = max(0, min(r_hi, s_hi) - max(r_lo, s_lo))
            union = max(r_hi, s_hi) - min(r_lo, s_lo)
            iou = overlap / union if union > 0 else 0
            ious.append(iou)
            print(f'  {arm} {ax}: IoU={iou:.3f}  '
                  f'(overlap {overlap*1000:.0f}mm / union {union*1000:.0f}mm)')
        print(f'  {arm} mean IoU: {np.mean(ious):.3f}')

    print(f'\n{sep}\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Real vs Sim EE-space comparison (aligned).')
    p.add_argument('--real_dir', type=str,
                   default='/home/xr/yan/IsaacLab-Arena/put_blocks_to_color')
    p.add_argument('--sim_ee_dir', type=str,
                   default='data/lerobot_ee')
    p.add_argument('--output_dir', type=str, default='data/real_vs_sim')
    p.add_argument('--urdf', type=str, default=DEFAULT_URDF,
                   help='Path to URDF file for robot mesh rendering')
    p.add_argument('--max_real', type=int, default=100,
                   help='Max real episodes to load')
    p.add_argument('--max_sim', type=int, default=50,
                   help='Max sim episodes to load')
    p.add_argument('--real_fps', type=int, default=20)
    p.add_argument('--sim_fps', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print('Loading real data (observation.state)...')
    real_raw = load_episodes(args.real_dir, max_episodes=args.max_real,
                             key='observation.state')
    print(f'  {len(real_raw)} episodes, {sum(len(e) for e in real_raw):,} frames')

    print('Loading sim data (observation.state)...')
    sim_raw = load_episodes(args.sim_ee_dir, max_episodes=args.max_sim,
                            key='observation.state')
    print(f'  {len(sim_raw)} episodes, {sum(len(e) for e in sim_raw):,} frames')

    # ── Align ─────────────────────────────────────────────────────────────
    print('Aligning: delta-from-home + RPY wrap + gripper normalize...')
    real_aligned = align_episodes(real_raw, is_real=True)
    sim_aligned = align_episodes(sim_raw, is_real=False)

    # ── Report ────────────────────────────────────────────────────────────
    print_analysis(real_aligned, sim_aligned, args.real_fps, args.sim_fps)

    # ── Concatenated arrays for histograms / box plots ────────────────────
    real_all = np.concatenate(real_aligned)
    sim_all = np.concatenate(sim_aligned)

    # ── Load robot mesh ──────────────────────────────────────────────────
    # Robot root is at [-0.517, -0.259, -0.221] in sim world frame
    robot_root_pos = [-0.51676, -0.25918, -0.22145]
    meshes = None
    if os.path.exists(args.urdf):
        print(f'Loading URDF mesh: {args.urdf}')
        meshes = load_robot_meshes(args.urdf, robot_root_pos=robot_root_pos)
        print(f'  Loaded {len(meshes)} mesh parts')
    else:
        print(f'  URDF not found: {args.urdf}, skipping robot mesh')

    # ── Compute world-frame trajectories for 3D plot ─────────────────────
    # Sim raw trajectories are already in world frame
    sim_left_world = [e[:, :3] for e in sim_raw]
    sim_right_world = [e[:, 7:10] for e in sim_raw]

    # Real delta trajectories: add sim home offset so they overlap with robot
    # Use the average sim home as the reference point
    sim_home_left = np.mean([e[0, :3] for e in sim_raw], axis=0)
    sim_home_right = np.mean([e[0, 7:10] for e in sim_raw], axis=0)
    real_left_world = [e[:, :3] + sim_home_left for e in real_aligned]
    real_right_world = [e[:, 7:10] + sim_home_right for e in real_aligned]

    # ── Visualizations ────────────────────────────────────────────────────
    print('Building visualizations...')

    # 1. 3D trajectory comparison (world frame with robot mesh)
    fig1 = build_3d_comparison(
        real_left_world, real_right_world,
        sim_left_world, sim_right_world,
        'Real vs Sim — EE Trajectories (world frame, with robot mesh)',
        meshes=meshes,
    )
    p1 = out / '01_3d_trajectory_comparison.html'
    fig1.write_html(str(p1), include_plotlyjs=True)
    print(f'  {p1}')

    # 2. Per-dimension box plot
    fig2 = build_per_dim_boxplot(real_all, sim_all)
    p2 = out / '02_per_dim_boxplot.html'
    fig2.write_html(str(p2), include_plotlyjs=True)
    print(f'  {p2}')

    # 3. XYZ distribution histograms (left + right)
    fig3 = build_distribution_comparison(
        real_all, sim_all,
        dims=[0, 1, 2, 7, 8, 9],
        dim_labels=[f'Left Δ{a}' for a in 'XYZ'] + [f'Right Δ{a}' for a in 'XYZ'],
        title='XYZ Position Distribution (aligned, delta-from-home)',
        unit='m',
    )
    p3 = out / '03_xyz_distribution.html'
    fig3.write_html(str(p3), include_plotlyjs=True)
    print(f'  {p3}')

    # 4. RPY distribution histograms
    fig4 = build_distribution_comparison(
        real_all, sim_all,
        dims=[3, 4, 5, 10, 11, 12],
        dim_labels=[f'Left Δ{a}' for a in ['Roll', 'Pitch', 'Yaw']] +
                   [f'Right Δ{a}' for a in ['Roll', 'Pitch', 'Yaw']],
        title='RPY Orientation Distribution (aligned, wrapped [-π,π])',
        unit='rad',
    )
    p4 = out / '04_rpy_distribution.html'
    fig4.write_html(str(p4), include_plotlyjs=True)
    print(f'  {p4}')

    # 5. Gripper distribution
    fig5 = build_distribution_comparison(
        real_all, sim_all,
        dims=[6, 13],
        dim_labels=['Left Gripper', 'Right Gripper'],
        title='Gripper Distribution (normalized [0,1])',
        unit='normalized', rows=1, cols=2, nbins=60,
    )
    p5 = out / '05_gripper_distribution.html'
    fig5.write_html(str(p5), include_plotlyjs=True)
    print(f'  {p5}')

    # 6. Time series (first few episodes)
    for ep_idx in range(min(3, len(real_aligned), len(sim_aligned))):
        fig6 = build_time_series(real_aligned[ep_idx], sim_aligned[ep_idx],
                                 args.real_fps, args.sim_fps, ep_idx=ep_idx)
        p6 = out / f'06_time_series_ep{ep_idx:03d}.html'
        fig6.write_html(str(p6), include_plotlyjs=True)
        print(f'  {p6}')

    # 7. 2D workspace scatter
    fig7 = build_workspace_heatmap(real_all, sim_all)
    p7 = out / '07_workspace_scatter.html'
    fig7.write_html(str(p7), include_plotlyjs=True)
    print(f'  {p7}')

    print(f'\nAll outputs saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
