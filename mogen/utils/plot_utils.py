"""
This code is borrowed from https://github.com/EricGuo5513/text-to-motion
"""

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20],
                       [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                            [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0],
                            [-1, 0, 0], [0, -1, 0], [0, -1, 0], [1, 0, 0],
                            [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
                            [0, 0, 1]])

t2m_raw_offsets = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                            [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
                            [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
                            [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
                            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
                            [0, -1, 0], [0, -1, 0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                       [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27],
                       [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42],
                        [21, 37, 38, 39], [21, 49, 50, 51]]


def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4, )).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3, )).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    '''Add Y-axis rotation to local joints'''
    rot = qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4, ))
    positions = qrot(rot, positions)
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def plot_3d_motion(save_path,
                   motion_length,
                   kinematic_tree,
                   joints,
                   title,
                   figsize=(10, 10),
                   fps=120,
                   radius=4):
    matplotlib.use('Agg')

    title_list = []
    for idx, t in enumerate(title):
        title_sp = t.split(' ')
        new_t = t
        if len(title_sp) > 20:
            new_t = '\n'.join([
                ' '.join(title_sp[:10]), ' '.join(title_sp[10:20]),
                ' '.join(title_sp[20:])
            ])
        elif len(title_sp) > 10:
            new_t = '\n'.join(
                [' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
        for i in range(motion_length[idx]):
            title_list.append(new_t)

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz],
                 [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        'red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue',
        'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred', 'darkred',
        'darkred', 'darkred'
    ]
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        fig.suptitle(title_list[index], fontsize=20)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                      np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1],
                      linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0],
                      data[index, chain, 1],
                      data[index, chain, 2],
                      linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig,
                        update,
                        frames=frame_number,
                        interval=1000 / fps,
                        repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_siamese_3d_motion(save_path,
                           kinematic_tree,
                           mp_joints,
                           title,
                           figsize=(10, 10),
                           fps=120,
                           radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([
            ' '.join(title_sp[:10]), ' '.join(title_sp[10:20]),
            ' '.join(title_sp[20:])
        ])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz],
                 [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    colors = [
        'red', 'green', 'black', 'red', 'blue', 'darkblue', 'darkblue',
        'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred', 'darkred',
        'darkred', 'darkred'
    ]

    mp_offset = list(range(-len(mp_joints) // 2, len(mp_joints) // 2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i, joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        mp_data.append({
            "joints": data,
            "MINS": MINS,
            "MAXS": MAXS,
            "trajec": trajec,
        })

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid, data in enumerate(mp_data):
            tree_data = zip(kinematic_tree, mp_colors[pid])
            for i, (chain, color) in enumerate(tree_data):
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0],
                          data["joints"][index, chain, 1],
                          data["joints"][index, chain, 2],
                          linewidth=linewidth,
                          color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig,
                        update,
                        frames=frame_number,
                        interval=1000 / fps,
                        repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()
