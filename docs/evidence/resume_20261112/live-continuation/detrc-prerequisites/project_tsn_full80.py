_base_ = ['./repcount_tsn_feature_enc_contrastive.py']
B = '/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites'
data = dict(
    train=dict(prop_file=B + '/annotations/train_corrected.csv', ft_path=B + '/features/rc_train.h5'),
    val=dict(prop_file=B + '/candidate/dataset/annotation/valid_new.csv', ft_path=B + '/features/rc_valid.h5'),
    test=dict(prop_file=B + '/candidate/dataset/annotation/test_new.csv', ft_path=B + '/features/rc_test.h5'),
)
work_dir = B + '/runs/full80_seed0'
output_config = dict(out=work_dir + '/native_results.json')
# All scientific recipe fields and native130 validation selection are inherited.
# Full357 label-free inference remains P5; this config alone does not implement it.
