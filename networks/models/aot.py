import torch.nn as nn

from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
import torch


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
        if cfg.USE_COO:
            print('use coo')
            self.coo_attention = CoordAttention(cfg.MODEL_ENCODER_DIM[-1], cfg.MODEL_ENCODER_DIM[-1])
            self.with_coo = True
        else:
            self.with_coo = False

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
        if self.with_coo:
            xs[-1] = self.coo_attention(xs[-1])
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

### Coordiate Attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h
