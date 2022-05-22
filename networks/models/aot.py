import torch.nn as nn

from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine


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
                obj_nums = None,
                pos_emb=None,
                ref_imgs=None,
                ref_masks=None,
                trans_imgs=None,
                output_all_frame=False,
                output_memories=True,
                ):
        '''
        给定 context 下预测当前帧的 mask 并返回相关memories
        Args:
        Return:
        mask, memoris
        '''
        assert (ref_masks is not None ) != (memories is not None)
        
        # if transition_imgs is not None and transition_imgs.shape[0] > 0:
        #     transition_mask = self(transition_imgs[0],short_memories)
        #     new_memoris = ... # update memories by transition_mask
        #     return self(img,new_memoris,transition_imgs=transition_imgs[1:])


        img_emb = self.encode_image(img) if img.shape[1] == 3 else img
        pos_emb = self.get_pos_emb(img_emb) if pos_emb is None else pos_emb

        if memories is not None:
            import torch
            out_emb, new_memories, *_ = self.LSTT_forward(img_emb, *memories)
            pred_id_logits = self.decode_id_logits(out_emb, img_emb)

            for batch_idx, obj_num in enumerate(self.obj_nums):
                pred_id_logits[batch_idx, (obj_num+1):] = - \
                    1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4
            pred_mask = torch.argmax(pred_id_logits, dim=1)
            memories = update_memories(memories, new_memories)
            return pred_mask, memories

        else:
            from utils.image import one_hot_mask
            ref_img_emb = self.encode_image(ref_imgs)
            ref_one_hot_mask = one_hot_mask(ref_masks,self.max_obj_num)
            obj_nums = ref_masks.max() - 1
            ref_id_emb = self.get_id_emb(ref_one_hot_mask)
            out_emb, _, *memories = self.LSTT_forward(ref_img_emb,None,None,ref_id_emb,pos_emb=pos_emb)
            return self.forward(img_emb,
                                memories=memories,
                                obj_nums=obj_nums,
                                trans_imgs=trans_imgs,
                                pos_emb=pos_emb,
                                output_all_frame=output_all_frame,
                                output_memories=output_memories,
                                )


def update_memories(memories, new_memories):
    pass