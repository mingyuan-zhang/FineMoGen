_base_ = ['../_base_/datasets/inter_human_bs128.py']
# use_adversarial_train = True

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
lr_config = dict(policy='step', step=[])
runner = dict(type='EpochBasedRunner', max_epochs=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 524
max_seq_len = 300
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 512
ff_size = 1024
num_heads = 8
dropout = 0

# model settings
model = dict(
    type='MotionDiffusion',
    model=dict(type='InterGen',
               input_dim=262,
               latent_dim=1024,
               ff_size=2048,
               num_layers=8,
               num_heads=8,
               dropout=0.1,
               activation="gelu",
               cfg_weight=3.5),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    loss_reduction="batch",
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
        # respace='30,30,16,12,12',
    ),
    inference_type='ddim')
data = dict(samples_per_gpu=64)
