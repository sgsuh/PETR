_base_ = ['./petr_r50dcn_gridmask_p4.py']

data_root = 'data/nuscenes/'

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl'),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl'),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl'))

evaluation = dict(jsonfile_prefix='work_dirs/petr_r50dcn_gridmask_p4_mini/eval')
