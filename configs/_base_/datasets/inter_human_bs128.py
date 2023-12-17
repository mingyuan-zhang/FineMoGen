# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length', 'clip_feat']
meta_keys = ['text']
train_pipeline = [
    dict(type='SwapSiameseMotion', prob=0.5),
    dict(type='ProcessSiameseMotion',
         feet_threshold=0.001,
         prev_frames=0,
         n_joints=22,
         prob=0.5),
    dict(type='Crop', crop_size=300),
    dict(type='Normalize',
         mean_path='data/datasets/inter_human/mean.npy',
         std_path='data/datasets/inter_human/std.npy'),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='SwapSiameseMotion', prob=0.5),
    dict(type='ProcessSiameseMotion',
         feet_threshold=0.001,
         prev_frames=0,
         n_joints=22,
         prob=0.5),
    dict(type='Crop', crop_size=300),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(type='RepeatDataset',
               dataset=dict(type='TextMotionDataset',
                            dataset_name='inter_human',
                            data_prefix='data',
                            pipeline=train_pipeline,
                            ann_file='train.txt',
                            motion_dir='motions',
                            text_dir='texts',
                            clip_feat_dir='clip_feats',
                            siamese_mode=True),
               times=100),
    test=dict(
        type='TextMotionDataset',
        dataset_name='inter_human',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='test.txt',
        motion_dir='motions',
        text_dir='texts',
        clip_feat_dir='clip_feats',
        siamese_mode=True,
        eval_cfg=dict(
            shuffle_indexes=True,
            replication_times=1,
            replication_reduction='statistics',
            evaluator_model=dict(
                type='InterCLIP',
                input_dim=258,
                latent_dim=1024,
                ff_size=2048,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                activation="gelu",
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='data/evaluators/inter_human/interclip.ckpt')),
            metrics=[
                dict(type='R Precision', batch_size=96, top_k=3),
                dict(type='Matching Score', batch_size=96),
                dict(type='FID', emb_scale=6),
                dict(type='Diversity',
                     num_samples=300,
                     emb_scale=6,
                     norm_scale=0.5),
                dict(type='MultiModality',
                     num_samples=100,
                     num_repeats=30,
                     num_picks=10)
            ]),
        test_mode=True))
