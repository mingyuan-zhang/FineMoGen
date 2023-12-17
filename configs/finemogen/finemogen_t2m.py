_base_ = ['../_base_/datasets/human_ml3d_bs128.py']

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 263
max_seq_len = 196
latent_dim = 64
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 256
num_heads = 8
dropout = 0
dataset_name = "human_ml3d"

# model settings
model = dict(type='MotionDiffusion',
             model=dict(type='FineMoGenTransformer',
                        input_feats=input_feats,
                        max_seq_len=max_seq_len,
                        latent_dim=latent_dim * 8,
                        time_embed_dim=time_embed_dim,
                        num_layers=4,
                        ca_block_cfg=dict(type='SAMI',
                                          latent_dim=latent_dim,
                                          text_latent_dim=text_latent_dim,
                                          num_heads=8,
                                          num_text_heads=1,
                                          num_experts=16,
                                          topk=2,
                                          gate_type='cosine_top',
                                          gate_noise=1.0,
                                          ffn_dim=ff_size,
                                          time_embed_dim=time_embed_dim,
                                          max_seq_len=max_seq_len,
                                          max_text_seq_len=77,
                                          temporal_comb=False,
                                          dropout=dropout),
                        ffn_cfg=dict(latent_dim=latent_dim,
                                     ffn_dim=ff_size,
                                     dropout=dropout,
                                     time_embed_dim=time_embed_dim),
                        text_encoder=dict(pretrained_model='clip',
                                          latent_dim=text_latent_dim,
                                          num_layers=2,
                                          ff_size=2048,
                                          dropout=dropout,
                                          use_text_proj=False),
                        pose_encoder_cfg=dict(dataset_name=dataset_name,
                                              latent_dim=latent_dim),
                        pose_decoder_cfg=dict(dataset_name=dataset_name,
                                              latent_dim=latent_dim,
                                              output_dim=input_feats),
                        scale_func_cfg=dict(scale=6.5),
                        moe_route_loss_weight=10.0,
                        template_kl_loss_weight=0.0001,
                        use_pos_embedding=False),
             loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
             diffusion_train=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large',
             ),
             diffusion_test=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large',
                 respace='15,15,8,6,6',
             ),
             inference_type='ddim',
             loss_reduction='batch')
data = dict(samples_per_gpu=64,
            train=dict(dataset=dict(ann_file='trainval_wo_mirror.txt')))
