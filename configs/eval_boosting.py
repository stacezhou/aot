from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'BOOSTING'

        self.init_dir()

        self.DATASETS = ['youtubevos', 'davis2017', '']

        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = './pretrain_models/AOTv2_85.1_80000.pth'
