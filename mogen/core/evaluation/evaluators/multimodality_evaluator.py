import numpy as np
import torch

from ..utils import calculate_multimodality
from .base_evaluator import BaseEvaluator


class MultiModalityEvaluator(BaseEvaluator):

    def __init__(self,
                 data_len=0,
                 evaluator_model=None,
                 num_samples=100,
                 num_repeats=30,
                 num_picks=10,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 **kwargs):
        super().__init__(replication_times=replication_times,
                         replication_reduction=replication_reduction,
                         batch_size=batch_size,
                         drop_last=drop_last,
                         eval_begin_idx=data_len,
                         eval_end_idx=data_len + num_samples * num_repeats,
                         evaluator_model=evaluator_model)
        self.num_samples = num_samples
        self.num_repeats = num_repeats
        self.num_picks = num_picks
        self.append_indexes = []
        for i in range(replication_times):
            append_indexes = []
            selected_indexs = np.random.choice(data_len, self.num_samples)
            for index in selected_indexs:
                append_indexes = append_indexes + [index] * self.num_repeats
            self.append_indexes.append(np.array(append_indexes))

    def single_evaluate(self, results):
        results = self.prepare_results(results)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred_motion = results['pred_motion']
        pred_motion_length = results['pred_motion_length']
        pred_motion_mask = results['pred_motion_mask']
        with torch.no_grad():
            pred_motion_emb = self.encode_motion(
                motion=pred_motion,
                motion_length=pred_motion_length,
                motion_mask=pred_motion_mask,
                device=device).cpu().detach().numpy()
        pred_motion_emb = \
            pred_motion_emb.reshape((self.num_samples, self.num_repeats, -1))
        multimodality = calculate_multimodality(pred_motion_emb,
                                                self.num_picks)
        return multimodality

    def parse_values(self, values):
        metrics = {}
        metrics['MultiModality (mean)'] = values[0]
        metrics['MultiModality (conf)'] = values[1]
        return metrics
