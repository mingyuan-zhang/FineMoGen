import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch

from .base_dataset import BaseMotionDataset
from .builder import DATASETS


@DATASETS.register_module()
class TextMotionDataset(BaseMotionDataset):
    """TextMotion dataset.

    Args:
        text_dir (str): Path to the directory containing the text files.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 token_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 siamese_mode: Optional[bool] = False,
                 tcomb_mode: Optional[bool] = False):
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     text_dir)
        if token_dir is not None:
            self.token_dir = os.path.join(data_prefix, 'datasets',
                                          dataset_name, token_dir)
        else:
            self.token_dir = None
        if clip_feat_dir is not None:
            self.clip_feat_dir = os.path.join(data_prefix, 'datasets',
                                              dataset_name, clip_feat_dir)
        else:
            self.clip_feat_dir = None
        self.siamese_mode = siamese_mode
        self.tcomb_mode = tcomb_mode
        super(TextMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_anno(self, name):
        results = {}
        if self.siamese_mode:
            motion_path = os.path.join(self.motion_dir, name + '.npz')
            motion_data = np.load(motion_path)
            results['motion1'] = motion_data['motion1']
            results['motion2'] = motion_data['motion2']
            assert results['motion1'].shape == results['motion2'].shape
        else:
            motion_path = os.path.join(self.motion_dir, name + '.npy')
            motion_data = np.load(motion_path)
            results['motion'] = motion_data
        text_path = os.path.join(self.text_dir, name + '.txt')
        text_data = []
        for line in open(text_path, 'r'):
            text_data.append(line.strip())
        results['text'] = text_data
        if self.token_dir is not None:
            token_path = os.path.join(self.token_dir, name + '.txt')
            token_data = []
            for line in open(token_path, 'r'):
                token_data.append(line.strip())
            results['token'] = token_data
        if self.clip_feat_dir is not None:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            clip_feat = torch.from_numpy(np.load(clip_feat_path))
            results['clip_feat'] = clip_feat
        return results

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]
        if 'clip_feat' in results.keys():
            results['clip_feat'] = results['clip_feat'][idx]
        if 'token' in results.keys():
            results['token'] = results['token'][idx]
        results['dataset_name'] = self.dataset_name
        results = self.pipeline(results)
        return results
