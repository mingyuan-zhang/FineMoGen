import numpy as np
import torch

from ..utils import calculate_top_k, euclidean_distance_matrix
from .base_evaluator import BaseEvaluator


class PrecisionEvaluator(BaseEvaluator):

    def __init__(self,
                 data_len=0,
                 evaluator_model=None,
                 top_k=3,
                 batch_size=32,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 **kwargs):
        super().__init__(replication_times=replication_times,
                         replication_reduction=replication_reduction,
                         batch_size=batch_size,
                         drop_last=drop_last,
                         eval_begin_idx=0,
                         eval_end_idx=data_len,
                         evaluator_model=evaluator_model)
        self.append_indexes = None
        self.top_k = top_k

    def single_evaluate(self, results):
        results = self.prepare_results(results)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred_motion = results['pred_motion']
        pred_motion_length = results['pred_motion_length']
        pred_motion_mask = results['pred_motion_mask']
        text = results['text']
        token = results['token']
        with torch.no_grad():
            word_emb = self.encode_text(text=text, token=token,
                                        device=device).cpu().detach().numpy()
            motion_emb = self.encode_motion(
                motion=pred_motion,
                motion_length=pred_motion_length,
                motion_mask=pred_motion_mask,
                device=device).cpu().detach().numpy()
            dist_mat = euclidean_distance_matrix(word_emb, motion_emb)
            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = calculate_top_k(argsmax, top_k=self.top_k)
            top_k_count = top_k_mat.sum(axis=0)
            all_size = word_emb.shape[0]
        return top_k_count, all_size

    def concat_batch_metrics(self, batch_metrics):
        top_k_count = 0
        all_size = 0
        for batch_top_k_count, batch_all_size in batch_metrics:
            top_k_count += batch_top_k_count
            all_size += batch_all_size
        R_precision = top_k_count / all_size
        return R_precision

    def parse_values(self, values):
        metrics = {}
        for top_k in range(self.top_k):
            metric_name_mean = 'R_precision Top %d (mean)' % (top_k + 1)
            metrics[metric_name_mean] = values[0][top_k]
            metric_name_conf = 'R_precision Top %d (conf)' % (top_k + 1)
            metrics[metric_name_conf] = values[1][top_k]
        return metrics
