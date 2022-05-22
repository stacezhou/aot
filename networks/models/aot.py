import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
from utils.image import one_hot_mask


class AOT(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(encoder,
                                     frozen_bn=cfg.MODEL_FREEZE_BN,
                                     freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT)
        self.encoder_projector = nn.Conv2d(cfg.MODEL_ENCODER_DIM[-1],
                                           cfg.MODEL_ENCODER_EMBEDDING_DIM,
                                           kernel_size=1)

        self.LSTT = LongShortTermTransformer(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            use_lstt_v2=cfg.USE_LSTT_V2,
            return_intermediate=True)

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * \
            (cfg.MODEL_LSTT_NUM +
             1) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_ENCODER_EMBEDDING_DIM

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)

        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=17,
                stride=16,
                padding=8)
        else:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=16,
                stride=16,
                padding=0)

        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)

        self.pos_generator = PositionEmbeddingSine(
            cfg.MODEL_ENCODER_EMBEDDING_DIM // 2, normalize=True)

        self._init_weight()

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def encode_image(self, img):
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def LSTT_forward(self,
                     curr_embs,
                     long_term_memories,
                     short_term_memories,
                     curr_id_emb=None,
                     pos_emb=None,
                     size_2d=(30, 30)):
        n, c, h, w = curr_embs[-1].size()
        curr_emb = curr_embs[-1].view(n, c, h * w).permute(2, 0, 1)
        lstt_embs, lstt_memories = self.LSTT(curr_emb, long_term_memories,
                                             short_term_memories, curr_id_emb,
                                             pos_emb, size_2d)
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(
            *lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(
                self.cfg.MODEL_ENCODER_EMBEDDING_DIM, -1).permute(0, 1),
            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)

    def forward(self,
                img,
                *,
                memories = None,
                ref_masks=None,
                ori_size=None,
                obj_nums = None,
                pos_emb=None,
                ):
        '''
        给定 context 下预测当前帧的 mask 并返回相关 memories
        Args:
        Return:
        mask, memoris
        '''
        assert (ref_masks is not None ) != (memories is not None and obj_nums is not None)
        
        if not isinstance(img, list):
            img_emb = self.encode_image(img) 
            ori_size = img.shape[-2:]
        else:
            img_emb = img
            assert ori_size is not None

        batch_size, *_, h,w = img_emb[-1].shape
        enc_size_2d = [h,w]
        enc_hw = h*w

        if pos_emb is None:
            pos_emb = self.get_pos_emb(img_emb[-1])
            pos_emb = pos_emb.flatten(start_dim=2).permute(2,0,1).expand(-1,batch_size,-1) # hw,B,C

        if memories is not None:
            out_emb, curr_memory, *_ = self.LSTT_forward(img_emb, *memories,size_2d=enc_size_2d)

            # predict mask of current frame
            pred_id_logits = self.decode_id_logits(out_emb, img_emb)
            for batch_idx, obj_num in enumerate(obj_nums):
                pred_id_logits[batch_idx, (obj_num+1):] = - \
                    1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4
            pred_id_logits = F.interpolate(pred_id_logits, ori_size, mode='bilinear',align_corners=True)
            pred_mask = torch.argmax(pred_id_logits, dim=1)

            # generate long and short memory of current frame
            pred_one_hot_mask = one_hot_mask(pred_mask, self.max_obj_num)
            pred_id_emb = self.get_id_emb(pred_one_hot_mask).view(batch_size, -1,enc_hw).permute(2,0,1) #hw,B,C
            
            long_memory = [
                lstt.make_global_kv(curr_kv, pred_id_emb)
                for curr_kv,lstt in zip(curr_memory, self.LSTT.layers)
            ]
            short_memory = [
                lstt.make_local_kv(global_kv, enc_size_2d)
                for global_kv, lstt in zip(long_memory, self.LSTT.layers)
            ]

            return pred_mask, long_memory, short_memory

        else:
            # generate long and short memory of ref frame
            ref_one_hot_mask = one_hot_mask(ref_masks,self.max_obj_num)
            ref_id_emb = self.get_id_emb(ref_one_hot_mask).view(batch_size,-1,enc_hw).permute(2,0,1) #hw,B,C
            out_emb, _, *memories = self.LSTT_forward(img_emb,None,None,ref_id_emb,pos_emb=pos_emb,size_2d=enc_size_2d)
            return ref_masks, *memories
