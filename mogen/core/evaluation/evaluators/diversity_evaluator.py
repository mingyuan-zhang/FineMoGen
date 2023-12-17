import torch

from ..utils import calculate_diversity
from .base_evaluator import BaseEvaluator


class DiversityEvaluator(BaseEvaluator):

    def __init__(self,
                 data_len=0,
                 evaluator_model=None,
                 num_samples=300,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 emb_scale=1,
                 norm_scale=1,
                 **kwargs):
        super().__init__(replication_times=replication_times,
                         replication_reduction=replication_reduction,
                         batch_size=batch_size,
                         drop_last=drop_last,
                         eval_begin_idx=0,
                         eval_end_idx=data_len,
                         evaluator_model=evaluator_model)
        self.num_samples = num_samples
        self.append_indexes = None
        self.emb_scale = emb_scale
        self.norm_scale = norm_scale

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
            diversity = calculate_diversity(pred_motion_emb, self.num_samples,
                                            self.emb_scale, self.norm_scale)
        return diversity

    def parse_values(self, values):
        metrics = {}
        metrics['Diversity (mean)'] = values[0]
        metrics['Diversity (conf)'] = values[1]
        return metrics
