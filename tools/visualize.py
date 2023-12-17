import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from scipy.ndimage import gaussian_filter

from mogen.models import build_architecture
from mogen.utils.plot_utils import (plot_3d_motion, plot_siamese_3d_motion,
                                    recover_from_ric, t2m_kinematic_chain)


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


def plot_t2m(data, motion_length, result_path, npy_path, caption):
    joints = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joints = motion_temporal_filter(joints, sigma=2.5)
    plot_3d_motion(save_path=result_path,
                   motion_length=motion_length,
                   kinematic_tree=t2m_kinematic_chain,
                   joints=joints,
                   title=caption,
                   fps=20)
    if npy_path is not None:
        np.save(npy_path, joints)


def plot_interhuman(data, result_path, npy_path, caption):
    data = data.reshape(data.shape[0], 2, -1)
    joints1 = data[:, 0, :22 * 3].reshape(-1, 22, 3)
    joints2 = data[:, 1, :22 * 3].reshape(-1, 22, 3)
    joints1 = motion_temporal_filter(joints1, sigma=4.5)
    joints2 = motion_temporal_filter(joints2, sigma=4.5)
    plot_siamese_3d_motion(save_path=result_path,
                           kinematic_tree=t2m_kinematic_chain,
                           mp_joints=[joints1, joints2],
                           title=caption,
                           fps=30)


def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--text', help='motion description', nargs='+')
    parser.add_argument('--motion_length',
                        type=int,
                        help='expected motion length',
                        nargs='+')
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--pose_npy',
                        help='output pose sequence file',
                        default=None)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset_name = cfg.data.test.dataset_name
    assert dataset_name in ["human_ml3d", "inter_human"]
    assert len(args.motion_length) == len(args.text)
    max_length = max(args.motion_length)
    if dataset_name == "human_ml3d":
        input_dim = 263
        assert max_length >= 16 and max_length <= 196
    elif dataset_name == "inter_human":
        input_dim = 524
        assert max_length >= 16 and max_length <= 300
    mean_path = os.path.join("data", "datasets", dataset_name, "mean.npy")
    std_path = os.path.join("data", "datasets", dataset_name, "std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)

    device = args.device
    num_intervals = len(args.text)
    motion = torch.zeros(num_intervals, max_length, input_dim).to(device)
    motion_mask = torch.zeros(num_intervals, max_length).to(device)
    for i in range(num_intervals):
        motion_mask[i, :args.motion_length[i]] = 1
    motion_length = torch.Tensor(args.motion_length).long().to(device)
    model = model.to(device)
    metas = []
    for t in args.text:
        metas.append({'text': t})
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'num_intervals': num_intervals,
        'motion_metas': metas,
    }

    all_pred_motion = []
    with torch.no_grad():
        input['inference_kwargs'] = {}
        output = model(**input)
        for i in range(num_intervals):
            pred_motion = output[i]['pred_motion'][:int(motion_length[i])]
            pred_motion = pred_motion.cpu().detach().numpy()
            pred_motion = pred_motion * std + mean
            all_pred_motion.append(pred_motion)
        pred_motion = np.concatenate(all_pred_motion, axis=0)

    if dataset_name == "human_ml3d":
        plot_t2m(data=pred_motion,
                 motion_length=args.motion_length,
                 result_path=args.out,
                 npy_path=args.pose_npy,
                 caption=args.text)
    elif dataset_name == "inter_human":
        plot_interhuman(data=pred_motion,
                        result_path=args.out,
                        npy_path=args.pose_npy,
                        caption=args.text)


if __name__ == '__main__':
    main()
