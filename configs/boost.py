import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'BOOST_FT'

        self.init_dir()

        self.DATASETS = ['boost']
        self.TEST_DATASET = 'vos_test'
        self.DIR_TEST_ROOT = './datasets/BOOST'
        import pandas as pd
        resample_weight_file = \
            '/home/zh21/code/aot_ft/results/boost_SwinB_AOTL__BOOST_FT/eval/video_df_vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema_vs_BOOST.csv'
        from pathlib import Path
        if Path(resample_weight_file).exists():
            resample_weight_dict = pd.read_csv(resample_weight_file).set_index('name').to_dict()['JF']
            self.RESAMPLE = resample_weight_dict
        else:
            resample_weight_dict = None
            self.RESAMPLE = None

        self.BOOST = dict(
            image_root = os.path.join(self.DIR_DATA, 'BOOST','JPEGImages'),
            label_root = os.path.join(self.DIR_DATA, 'BOOST','Annotations'),
            resample_weight = resample_weight_dict,
            repeat_time=1,
            rand_gap=3,
            dynamic_merge=True,
        )
        self.TRAIN_BATCH_SIZE = 12
        self.USE_LSTT_V2 = True

        # self.MODEL_FREEZE_BACKBONE = True
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = './pretrain_models/AOTv2_85.1_80000.pth'

        self.TRAIN_TOTAL_STEPS = 4000
        self.TRAIN_SAVE_STEP = 4000

        self.TRAIN_LR = 2e-6
        self.TRAIN_LR_MIN = 2e-7

        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1