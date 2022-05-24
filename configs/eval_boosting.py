from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'BOOST_INF'

        self.init_dir()

        self.TEST_DATASET = ['vos_test']
        self.DIR_TEST_ROOT = './datasets/BOOST'

        self.PRETRAIN_FULL = True  # if False, load encoder only
