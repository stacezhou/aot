from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'finetune'

        self.init_dir()

        self.DATASETS = ['finetune']
        self.DIR_FINETUNE = './datasets/finetune'

        self.DATA_DYNAMIC_MERGE_PROB = 1.0

        self.TRAIN_LR = 1e-5
        self.TRAIN_LR_MIN = 5e-6
        self.TRAIN_WEIGHT_DECAY = 0.03
        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1
