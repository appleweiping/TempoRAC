checkpoint_config = dict(interval=40, by_epoch=True)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
num_queries = 40
clip_len = 512
train_stride_rate = 0.25
test_stride_rate = 0.75
frame_interval_list = [1]
total_epochs = 80
feature_type = 'TSN'
model = dict(
    type='DeTRC',
    input_feat_dim=2048,
    num_class=2,
    feat_dim=512,
    n_head=8,
    encoder_sample_num=4,
    decoder_sample_num=4,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_queries=40,
    clip_len=512,
    stride_rate=0.75,
    test_bg_thershold=0.0,
    coef_l1=5.0,
    coef_iou=2.0,
    coef_ce=1.0,
    coef_aceenc=0.1,
    coef_acedec=1.0,
    coef_quality=1.0,
    coef_iou_decay=0.1,
    coef_contrastive=1.0,
    temperature=100.0,
    use_contrastive=True,
    use_temporal_conv=False,
    use_enc_anchor=True,
    enc_anchor_type='content',
    content_query_type='normal',
    constrastive_type='all_layer',
    use_position_embedding='learnable')
dataset_type = 'RepCountDataset'
data_root_train = './datasets/LLSP/feature-frame-mae/train_rgb.h5'
data_root_val = './datasets/LLSP/feature-frame-mae/valid_rgb.h5'
data_root_test = './datasets/LLSP/feature-frame-mae/test_rgb.h5'
flow_root_train = None
flow_root_val = None
ann_file_train = './datasets/LLSP/annotation/train_new.csv'
ann_file_val = './datasets/LLSP/annotation/valid_new.csv'
ann_file_test = './datasets/LLSP/annotation/test_new.csv'
test_pipeline = [
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox', 'video_gt_box', 'snippet_num'],
        meta_name='video_meta',
        meta_keys=['video_name', 'origin_snippet_num']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='gt_bbox', stack=False, cpu_only=True),
            dict(key='raw_feature', stack=False, cpu_only=True),
            dict(key='video_gt_box', stack=False, cpu_only=True),
            dict(key='snippet_num', stack=True, cpu_only=True)
        ])
]
train_pipeline = [
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox', 'snippet_num'],
        meta_name='video_meta',
        meta_keys=['video_name', 'origin_snippet_num']),
    dict(type='ToTensor', keys=['snippet_num']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='gt_bbox', stack=True, cpu_only=True),
            dict(key='raw_feature', stack=True, cpu_only=True)
        ])
]
val_pipeline = [
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox', 'video_gt_box', 'snippet_num'],
        meta_name='video_meta',
        meta_keys=['video_name', 'origin_snippet_num']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='gt_bbox', stack=False, cpu_only=True),
            dict(key='raw_feature', stack=False, cpu_only=True),
            dict(key='video_gt_box', stack=False, cpu_only=True),
            dict(key='snippet_num', stack=True, cpu_only=True)
        ])
]
data = dict(
    train_dataloader=dict(
        workers_per_gpu=2,
        videos_per_gpu=64,
        drop_last=False,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2),
    val_dataloader=dict(
        workers_per_gpu=2,
        videos_per_gpu=64,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2),
    test_dataloader=dict(
        workers_per_gpu=2,
        videos_per_gpu=64,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2),
    test=dict(
        type='RepCountDataset',
        prop_file=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/candidate/dataset/annotation/test_new.csv',
        ft_path=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/features/rc_test.h5',
        pipeline=[
            dict(
                type='Collect',
                keys=['raw_feature', 'gt_bbox', 'video_gt_box', 'snippet_num'],
                meta_name='video_meta',
                meta_keys=['video_name', 'origin_snippet_num']),
            dict(
                type='ToDataContainer',
                fields=[
                    dict(key='gt_bbox', stack=False, cpu_only=True),
                    dict(key='raw_feature', stack=False, cpu_only=True),
                    dict(key='video_gt_box', stack=False, cpu_only=True),
                    dict(key='snippet_num', stack=True, cpu_only=True)
                ])
        ],
        test_mode=True,
        clip_len=512,
        stride_rate=0.75,
        feature_type='TSN',
        frame_interval_list=[1]),
    val=dict(
        type='RepCountDataset',
        prop_file=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/candidate/dataset/annotation/valid_new.csv',
        ft_path=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/features/rc_valid.h5',
        pipeline=[
            dict(
                type='Collect',
                keys=['raw_feature', 'gt_bbox', 'video_gt_box', 'snippet_num'],
                meta_name='video_meta',
                meta_keys=['video_name', 'origin_snippet_num']),
            dict(
                type='ToDataContainer',
                fields=[
                    dict(key='gt_bbox', stack=False, cpu_only=True),
                    dict(key='raw_feature', stack=False, cpu_only=True),
                    dict(key='video_gt_box', stack=False, cpu_only=True),
                    dict(key='snippet_num', stack=True, cpu_only=True)
                ])
        ],
        test_mode=True,
        clip_len=512,
        stride_rate=0.75,
        feature_type='TSN',
        frame_interval_list=[1]),
    train=dict(
        type='RepCountDataset',
        prop_file=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/annotations/train_corrected.csv',
        ft_path=
        '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/features/rc_train.h5',
        pipeline=[
            dict(
                type='Collect',
                keys=['raw_feature', 'gt_bbox', 'snippet_num'],
                meta_name='video_meta',
                meta_keys=['video_name', 'origin_snippet_num']),
            dict(type='ToTensor', keys=['snippet_num']),
            dict(
                type='ToDataContainer',
                fields=[
                    dict(key='gt_bbox', stack=True, cpu_only=True),
                    dict(key='raw_feature', stack=True, cpu_only=True)
                ])
        ],
        epoch_multiplier=1,
        feature_type='TSN',
        clip_len=512,
        stride_rate=0.25,
        frame_interval_list=[1]))
evaluation = dict(interval=1, save_best='OBO', by_epoch=True, rule='greater')
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0)
optimizer_config = dict()
lr_config = dict(policy='step', step=[15, 60], gamma=0.1, by_epoch=True)
work_dir = '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/runs/full80_seed0'
output_config = dict(
    out=
    '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/runs/full80_seed0/native_results.json'
)
custom_hooks = [dict(type='DeTRCDurableStateHook', priority='LOWEST')]
B = '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites'
