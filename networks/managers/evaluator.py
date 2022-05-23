import os
import time
import datetime as datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.eval_datasets import YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

class Evaluator(object):
    def __init__(self, cfg, rank=0, seq_queue=None, info_queue=None):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print("Use GPU {} for evaluating.".format(self.gpu))
        torch.cuda.set_device(self.gpu)

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)
        ggn = '/home/zh21/code/Generic-Grouping/configs/mask_rcnn/class_agn_mask_rcnn.py'
        self.ggn = build_detector(Config.fromfile(ggn).model).cuda(self.gpu)
        ggn_ckpt = '/home/zh21/code/ggn_coco.pth'
        load_checkpoint(self.ggn, ggn_ckpt)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg

        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return

        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(
                        map(lambda x: int(x.split('_')[-1].split('.')[0]),
                            ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT,
                                              'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        eval_name = '{}_{}_{}_{}_ckpt_{}'.format(cfg.TEST_DATASET,
                                                 cfg.TEST_DATASET_SPLIT,
                                                 cfg.EXP_NAME, cfg.STAGE_NAME,
                                                 self.ckpt)

        if cfg.TEST_EMA:
            eval_name += '_ema'
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms_' + str(cfg.TEST_MULTISCALE).replace(
                '.', 'dot').replace('[', '').replace(']', '').replace(
                    ', ', '_')

        if 'youtubevos' in cfg.TEST_DATASET:
            year = int(cfg.TEST_DATASET[-4:])
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest

                self.result_root_sparse = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_sparse',
                                                       'Annotations')
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    '{}_sparse.zip'.format(eval_name))
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.dataset = youtubevos_test(root=cfg.DIR_YTB,
                                           year=year,
                                           split=split,
                                           transform=eval_transforms,
                                           result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log('Eval {} on {} {}:'.format(cfg.EXP_NAME,
                                                  cfg.TEST_DATASET,
                                                  cfg.TEST_DATASET_SPLIT))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                          eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                    '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            try:
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Failed to mask dir: {}.'.format(
                    self.result_root))
        self.print_log('Done!')

    @torch.no_grad()
    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0
        processed_video_num = 0
        total_video_num = len(self.dataset)
        start_eval_time = time.time()
        debug_subset = [

        ]

        if self.seq_queue is not None:
            if self.rank == 0:
                for seq_idx in range(total_video_num):
                    self.seq_queue.put(seq_idx)
                for _ in range(self.gpu_num):
                    self.seq_queue.put('END')
            coming_seq_idx = self.seq_queue.get()

        for seq_idx, seq_dataset in enumerate(self.dataset):
            video_num += 1

            if self.seq_queue is not None:
                if coming_seq_idx == 'END':
                    break
                elif coming_seq_idx != seq_idx:
                    continue
                else:
                    coming_seq_idx = self.seq_queue.get()

            processed_video_num += 1

            seq_name = seq_dataset.seq_name
            # if seq_name not in debug_subset:
            #     continue
            print('GPU {} - Processing Seq {} [{}/{}]:'.format(
                self.gpu, seq_name, video_num, total_video_num))
            torch.cuda.empty_cache()

            seq_dataloader = DataLoader(seq_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=cfg.TEST_WORKERS,
                                        pin_memory=True)


            seq_samples = [sample for samples in seq_dataloader for sample in samples]
            seq_imgs = [sample['current_img'] for sample in seq_samples]
            seq_obj_idx = [[int(i) for i in sample['meta']['obj_idx']] for sample in seq_samples]
            seq_imgname = [sample['meta']['current_name'] for sample in seq_samples]
            # seq_ori_h = [sample['meta']['height'] for sample in seq_samples]
            # seq_ori_w = [sample['meta']['width'] for sample in seq_samples]
            seq_masks = [
                F.interpolate(sample['current_label'].float(),img.shape[-2:], mode='nearest') if 'current_label' in sample else None
                for sample,img in zip(seq_samples, seq_imgs)
            ]
            seq_obj_nums = []
            for mask in seq_masks:
                if mask is not None:
                    obj_num = [int(mask.max().item())]
                seq_obj_nums.append(obj_num)
            seq_inputs = list(zip(seq_imgs,seq_masks,seq_obj_nums))

            seq_outputs = []
            memories = None
            saved_memories = []
            for idx,(imgs,ref_masks,obj_nums) in enumerate(seq_inputs):

                ref_masks = ref_masks.cuda() if ref_masks is not None else None
                pred_masks, *memory = self.model(imgs.cuda(), 
                                                memories = memories,
                                                ref_masks = ref_masks, 
                                                obj_nums=obj_nums)
                if memories is None:
                    memories = memory 
                else:
                    memories[1] = memory[1]
                    if idx % 5 == 0:
                        memories[0] = [
                            [ torch.cat([t1,t2], dim=0) for t1,t2 in zip(l1,l2) ]
                            for l1,l2 in zip(memories[0],memory[0])
                        ]
                
                saved_memories.append(memory)
                seq_outputs.append(pred_masks)

            # Save result
            for imgname,obj_idx,mask_result in zip(seq_imgname,seq_obj_idx,seq_outputs):
                save_mask(mask_result.squeeze(0).squeeze(0),
                        os.path.join(self.result_root, seq_name,
                                    imgname[0].split('.')[0] + '.png'),
                            obj_idx)

            max_mem = torch.cuda.max_memory_allocated(
                device=self.gpu) / (1024.**3)
            print(
                "GPU {} - Seq {} - Max Mem: {:.2f}G"
                .format(self.gpu, seq_name, max_mem))

        if self.seq_queue is not None:
            print('Finished the evaluation on GPU {}.'.format(self.gpu))

        if self.rank == 0:
            zip_folder(self.source_folder, self.zip_dir)
            self.print_log('Saving result to {}.'.format(self.zip_dir))
            end_eval_time = time.time()
            total_eval_time = str(
                datetime.timedelta(seconds=int(end_eval_time -
                                               start_eval_time)))
            self.print_log("Total evaluation time: {}".format(total_eval_time))

    def print_log(self, string):
        if self.rank == 0:
            print(string)

    def ggn_mask(self, img):
        img_meta = dict()
        B, C, H, W = img.shape
        img_meta['ori_shape'] = (H, W, C)
        img_meta['pad_shape'] = (H, W, C)
        img_meta['img_shape'] = (H, W, C)
        img_meta['batch_input_shape'] = (H, W)
        img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
        img_meta['flip'] = False
        img_metas = [img_meta]

        # proposal_list = [tensor.Size([N, 5])] x1,y1,x2,y2,p
        proposals = None
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        result = self.roi_head.simple_test(x,
                                           proposal_list,
                                           img_metas,
                                           rescale=False)
        return result