import torch

from ..utils import calculate_activation_statistics, calculate_frechet_distance
from .base_evaluator import BaseEvaluator


class FIDEvaluator(BaseEvaluator):

    def __init__(self,
                 data_len=0,
                 evaluator_model=None,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 emb_scale=1,
                 replication_reduction='statistics',
                 **kwargs):
        super().__init__(replication_times=replication_times,
                         replication_reduction=replication_reduction,
                         batch_size=batch_size,
                         drop_last=drop_last,
                         eval_begin_idx=0,
                         eval_end_idx=data_len,
                         evaluator_model=evaluator_model)
        self.emb_scale = emb_scale
        self.append_indexes = None

    def single_evaluate(self, results):
        results = self.prepare_results(results)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred_motion = results['pred_motion']
        pred_motion_length = results['pred_motion_length']
        pred_motion_mask = results['pred_motion_mask']
        motion = results['motion']
        motion_length = results['motion_length']
        motion_mask = results['motion_mask']
        with torch.no_grad():
            pred_motion_emb = self.encode_motion(
                motion=pred_motion,
                motion_length=pred_motion_length,
                motion_mask=pred_motion_mask,
                device=device).cpu().detach().numpy()
            gt_motion_emb = self.encode_motion(
                motion=motion,
                motion_length=motion_length,
                motion_mask=motion_mask,
                device=device).cpu().detach().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(
            gt_motion_emb, self.emb_scale)
        pred_mu, pred_cov = calculate_activation_statistics(
            pred_motion_emb, self.emb_scale)
        fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
        return fid

    def parse_values(self, values):
        metrics = {}
        metrics['FID (mean)'] = values[0]
        metrics['FID (conf)'] = values[1]
        return metrics
