# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import sys
from math import cos, pi
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss
import torch.distributed as dist
from classy_vision.generic.distributed_util import is_distributed_training_run

from models.transformers.transformer_predictor import TransformerPredictor
from utils import init


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='normal'):
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ObjectNeck(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8, 
                 norm_layer=None,
                 mask_type="group",
                 num_proto=64,
                 temp=0.07,
                 **kwargs):
        super(ObjectNeck, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads
        self.mask_type = mask_type
        self.temp = temp
        self.eps = 1e-7

        hid_channels = hid_channels or in_channels
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)

        if mask_type == "attn":
            # self.slot_embed = nn.Embedding(num_proto, out_channels)
            # self.proj_obj = MLP1D(out_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
            self.proj_attn = TransformerPredictor(in_channels=out_channels, hidden_dim=out_channels, num_queries=num_proto,
                                                  nheads=8, dropout=0.1, dim_feedforward=out_channels, enc_layers=0,
                                                  dec_layers=1, pre_norm=False, deep_supervision=False,
                                                  mask_dim=out_channels, enforce_input_project=False,
                                                  mask_classification=False, num_classes=0)
    
    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)
    
    def forward(self, x):
        out = {}

        b, c, h, w = x.shape

        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x = x.flatten(2)    # (bs, c, h*w)
        z = self.proj(torch.cat([x_pool, x], dim=2))    # (bs, d, 1+h*w)
        z_g, z_feat = torch.split(z, [1, x.shape[2]], dim=2)  # (bs, d, 1), (bs, d, h*w)

        z_feat = z_feat.contiguous()

        if self.mask_type == "attn":
            z_feat = z_feat.view(b, -1, h, w)
            x = x.view(b, c, h, w)
            attn_out = self.proj_attn(z_feat, None)
            mask_embed = attn_out["mask_embed"]  # (bs, q, c)
            out["mask_embed"] = mask_embed

            dots = torch.einsum('bqc,bchw->bqhw', F.normalize(mask_embed, dim=2), F.normalize(z_feat, dim=1))
            obj_attn = (dots / self.temp).softmax(dim=1) + self.eps
            # obj_attn = (dots / 1.0).softmax(dim=1) + self.eps
            slots = torch.einsum('bchw,bqhw->bqc', x, obj_attn / obj_attn.sum(dim=(2, 3), keepdim=True))
            # slots = torch.einsum('bchw,bqhw->bqc', z_feat, obj_attn / obj_attn.sum(dim=(2, 3), keepdim=True))
            obj_attn = obj_attn.view(b, -1, h * w)
            out["dots"] = dots
        else:
            # do attention according to obj attention map
            obj_attn = F.normalize(z_feat, dim=1) if self.l2_norm else z_feat
            obj_attn /= self.scale
            obj_attn = obj_attn.view(b, self.num_heads, -1, h * w)  # (bs, h, d/h, h*w)
        obj_attn_raw = F.softmax(obj_attn, dim=-1)

        if self.mask_type == "group":
            obj_attn = F.softmax(obj_attn, dim=-1)
            x = x.view(b, self.num_heads, -1, h*w)  # (bs, h, c/h, h*w)
            obj_val = torch.matmul(x, obj_attn.transpose(3, 2))    # (bs, h, c//h, d/h)
            obj_val = obj_val.view(b, c, obj_attn.shape[-2])    # (bs, c, d/h)
        elif self.mask_type == "max":
            obj_attn, _ = torch.max(obj_attn, dim=1)  # (bs, d/h, h*w)
            # obj_attn = torch.mean(obj_attn, dim=1)
            obj_attn = F.softmax(obj_attn, dim=-1)
            obj_val = torch.matmul(x, obj_attn.transpose(2, 1))  # (bs, c, d/h)
        elif self.mask_type == "attn":
            obj_val = slots.transpose(2, 1)  # (bs, c, q)

        # projection
        obj_val = self.proj_obj(obj_val)    # (bs, d, d/h)

        out["obj_attn"] = obj_attn
        out["obj_attn_raw"] = obj_attn_raw

        return z_g, obj_val, out  # (bs, d, 1), (bs, d, d//h), where the second dim is channel
    
    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)


class EncoderObj(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, mask_type="group", num_proto=64, temp=0.07):
        super(EncoderObj, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer, with_avgpool=False)
        in_dim = self.backbone.out_channels
        self.neck = ObjectNeck(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim,
                               norm_layer=norm_layer, num_layers=num_mlp,
                               scale=scale, l2_norm=l2_norm, num_heads=num_heads, mask_type=mask_type,
                               num_proto=num_proto, temp=temp)
        # self.neck.init_weights(init_linear='kaiming')

    def forward(self, im):
        out = self.backbone(im)
        out = self.neck(out)
        return out


class LEWELB_EMAN(nn.Module):
    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, loss_weight=0.5, mask_type="group", num_proto=64,
                 teacher_temp=0.07, student_temp=0.1, loss_w_cluster=0.5, **kwargs):
        super().__init__()

        self.base_m = m
        self.curr_m = m
        self.loss_weight = loss_weight
        self.loss_w_cluster = loss_w_cluster
        self.mask_type = mask_type
        assert mask_type in ["group", "max", "attn"]
        self.num_proto = num_proto
        self.student_temp = student_temp  # 0.1
        self.teacher_temp = teacher_temp  # 0.07

        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads, mask_type=mask_type,
                                     num_proto=num_proto, temp=self.teacher_temp)

        # checkpoint = torch.load("./checkpoints/lewel_b_400ep.pth", map_location="cpu")
        # msg = self.online_net.backbone.load_state_dict(checkpoint)
        # assert set(msg.missing_keys) == set()
        # state_dict = checkpoint['state_dict']
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("online_net.backbone.", ""): v for k, v in state_dict.items()}
        # self.online_net.backbone.load_state_dict(state_dict)

        self.target_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads, mask_type=mask_type,
                                     num_proto=num_proto, temp=self.teacher_temp)
        self.predictor = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        # self.predictor.init_weights()
        self.predictor_obj = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        # self.predictor_obj.init_weights()
        self.encoder_q = self.online_net.backbone

        # copy params from online model to target model
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)  # initialize
            param_tgt.requires_grad = False  # not update by gradient

        self.center_momentum = 0.9
        self.register_buffer("center", torch.zeros(1, self.num_proto))
    
    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return loss

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def loss_func(self, online, target):
        z_o, obj_o, res_o = online
        z_t, obj_t, res_t = target
        # instance-level loss
        z_o_pred = self.predictor(z_o).squeeze(-1)
        z_t = z_t.squeeze(-1)
        loss_inst = self.mse_loss(z_o_pred, z_t)
        # object-level loss
        b, c, n = obj_o.shape
        obj_o_pred = self.predictor_obj(obj_o).transpose(2, 1).reshape(b*n, c)
        obj_t = obj_t.transpose(2, 1).reshape(b*n, c)
        loss_obj = self.mse_loss(obj_o_pred, obj_t)

        # score_q = torch.einsum('bnc,bc->bn', F.normalize(obj_o_pred, dim=2), F.normalize(z_o_pred, dim=1))
        # score_k = torch.einsum('bnc,bc->bn', F.normalize(obj_t, dim=2), F.normalize(z_t, dim=1))
        # score_q = torch.einsum('bnc,bc->bn', F.normalize(obj_o.transpose(2, 1), dim=2), F.normalize(z_o.squeeze(-1), dim=1))
        # # score_q = torch.einsum('bnc,bc->bn', F.normalize(obj_t, dim=2), F.normalize(z_o.squeeze(-1), dim=1))
        # score_k = torch.einsum('bnc,bc->bn', F.normalize(obj_t, dim=2), F.normalize(z_t, dim=1))

        # score_q = torch.einsum('bnc,bc->bn', F.normalize(res_o["mask_embed"], dim=2), F.normalize(z_o.squeeze(-1), dim=1))
        # score_q = torch.einsum('bnc,bc->bn', F.normalize(res_o["mask_embed"], dim=2), F.normalize(z_t, dim=1))
        # score_q = torch.einsum('bnc,bc->bn', F.normalize(res_t["mask_embed"], dim=2), F.normalize(z_o.squeeze(-1), dim=1))
        # score_k = torch.einsum('bnc,bc->bn', F.normalize(res_t["mask_embed"], dim=2), F.normalize(z_t, dim=1))
        # loss_relation = self.self_distill(score_q, score_k)

        # score_q_1 = torch.einsum('bnc,bc->bn', F.normalize(res_o["mask_embed"], dim=2), F.normalize(z_t, dim=1))
        # score_q_2 = torch.einsum('bnc,bc->bn', F.normalize(res_t["mask_embed"], dim=2), F.normalize(z_o.squeeze(-1), dim=1))
        # score_k = torch.einsum('bnc,bc->bn', F.normalize(res_t["mask_embed"], dim=2), F.normalize(z_t, dim=1))
        # loss_relation = 0.5 * (self.self_distill(score_q_1, score_k) + self.self_distill(score_q_2, score_k))

        loss_base = loss_inst * self.loss_weight + loss_obj * (1 - self.loss_weight)

        # sum
        return loss_base, loss_inst, loss_obj
    
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.online_net.state_dict()
        state_dict_tgt = self.target_net.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if is_distributed_training_run():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def get_heatmap(self, x):
        _, _, out = self.online_net(x)
        return out

    def ctr_loss(self, online_1, online_2, target_1, target_2):
        z_o_1, obj_o_1, res_o_1 = online_1
        z_o_2, obj_o_2, res_o_2 = online_2
        z_t_1, obj_t_1, res_t_1 = target_1
        z_t_2, obj_t_2, res_t_2 = target_2

        # corre_o = torch.matmul(F.normalize(res_o_1["mask_embed"], dim=2),
        #                        F.normalize(res_o_2["mask_embed"], dim=2).transpose(2, 1))  # b, q, c
        # corre_t = torch.matmul(F.normalize(res_t_1["mask_embed"], dim=2),
        #                        F.normalize(res_t_2["mask_embed"], dim=2).transpose(2, 1))  # b, q, c
        # loss = self.self_distill(corre_o.flatten(0, 1), corre_t.flatten(0, 1))
        # score = corre_t.flatten(0, 1)

        loss = 0.5 * (self.self_distill(res_o_1["dots"].permute(0, 2, 3, 1).flatten(0, 2),
                                        res_t_1["dots"].permute(0, 2, 3, 1).flatten(0, 2))
                      + self.self_distill(res_o_2["dots"].permute(0, 2, 3, 1).flatten(0, 2),
                                          res_t_2["dots"].permute(0, 2, 3, 1).flatten(0, 2)))
        score_k1 = res_t_1["dots"]
        score_k2 = res_t_2["dots"]
        score = torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2)

        return loss, score

    def forward(self, im_v1, im_v2=None, **kwargs):
        """
        Input:
            im_v1: a batch of view1 images
            im_v2: a batch of view2 images
        Output:
            loss
        """
        # for inference, online_net.backbone model only
        if im_v2 is None:
            feats = self.online_net.backbone(im_v1)
            return F.adaptive_avg_pool2d(feats, 1).flatten(1)

        # compute online_net features
        proj_online_v1 = self.online_net(im_v1)
        proj_online_v2 = self.online_net(im_v2)

        # compute target_net features
        with torch.no_grad():  # no gradient to keys
            proj_target_v1 = [x.clone().detach() if isinstance(x, torch.Tensor) else x for x in self.target_net(im_v1)]
            proj_target_v2 = [x.clone().detach() if isinstance(x, torch.Tensor) else x for x in self.target_net(im_v2)]

        # loss. NOTE: the predction is moved to loss_func
        loss_base1, loss_inst1, loss_obj1 = self.loss_func(proj_online_v1, proj_target_v2)
        loss_base2, loss_inst2, loss_obj2 = self.loss_func(proj_online_v2, proj_target_v1)
        loss_base = loss_base1 + loss_base2

        loss_relation, score = self.ctr_loss(proj_online_v1, proj_online_v2, proj_target_v1, proj_target_v2)
        loss = loss_base + loss_relation * self.loss_w_cluster

        loss_pack = {}
        loss_pack["base"] = loss_base
        loss_pack["inst"] = (loss_inst1 + loss_inst2) * self.loss_weight
        loss_pack["obj"] = (loss_obj1 + loss_obj2) * (1 - self.loss_weight)
        loss_pack["relation"] = loss_relation

        self.update_center(score)

        return loss, loss_pack

    def extra_repr(self) -> str:
        parts = []
        for name in ["loss_weight", "mask_type", "num_proto", "teacher_temp", "loss_w_cluster"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)


class LEWELB(LEWELB_EMAN):
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data = param_tgt.data * momentum + param_ol.data * (1. - momentum)


if __name__ == '__main__':
    from models import get_model
    import backbone as backbone_models

    model_func = get_model("LEWELB_EMAN")
    norm_layer = None
    model = model_func(
        backbone_models.__dict__["resnet50_encoder"],
        dim=256,
        m=0.996,
        hid_dim=4096,
        norm_layer=norm_layer,
        num_neck_mlp=2,
        scale=1.,
        l2_norm=True,
        num_heads=4,
        loss_weight=0.5,
        mask_type="attn"
    )
    print(model)

    x1 = torch.randn(16, 3, 224, 224)
    x2 = torch.randn(16, 3, 224, 224)
    out = model(x1, x2)
