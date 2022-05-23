from .default import DefaultEngineConfig
import os


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'finetune'

        self.init_dir()

        self.DATASETS = ['finetune']
        self.DIR_FINETUNE = './datasets/finetune'

        self.TRAIN_TOTAL_STEPS = 1000
        self.TRAIN_SAVE_STEP = 1000

        self.TRAIN_LR = 2e-6
        self.TRAIN_LR_MIN = 2e-7

        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1

        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = './pretrain_models/AOTv2_85.1_80000.pth'
