# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import sys
import math
from math import cos, pi
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss
import torch.distributed as dist
from classy_vision.generic.distributed_util import is_distributed_training_run

from models.transformers.transformer_predictor import TransformerPredictor
from utils import init



@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True, epsilon=0.05):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    # Q = torch.exp(Q / epsilon).t()
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T


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
        self.proj_pixel = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)

        if mask_type == "attn":
            self.proj_attn = TransformerPredictor(in_channels=in_channels, hidden_dim=out_channels, num_queries=num_proto,
                                                  nheads=8, dropout=0.1, dim_feedforward=out_channels, enc_layers=0,
                                                  dec_layers=2, pre_norm=False, deep_supervision=False,
                                                  mask_dim=out_channels, enforce_input_project=False,
                                                  mask_classification=False, num_classes=0)

            self.proto_momentum = 0.9
            self.register_buffer("proto", torch.randn(num_proto, out_channels))
            # self.proto = nn.Embedding(num_proto, out_channels)
    
    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_pixel.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)

    @torch.no_grad()
    def update_proto(self, mask_embed):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(mask_embed, dim=0)
        if is_distributed_training_run():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.proto = self.proto * self.proto_momentum + batch_center * (1 - self.proto_momentum)

    def forward(self, x, isTrain=True):
        out = {}

        b, c, h, w = x.shape

        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x = x.flatten(2)    # (bs, c, h*w)
        z_g = self.proj(x_pool)
        z_feat = self.proj_pixel(x)

        if self.mask_type == "attn":
            z_feat = z_feat.view(b, -1, h, w)
            x = x.view(b, c, h, w)
            # attn_out = self.proj_attn(z_feat, None)
            attn_out = self.proj_attn(x, None)
            mask_embed = attn_out["mask_embed"]  # (bs, q, c)

            if isTrain:
                # mask_embed = AllReduce.apply(torch.mean(mask_embed, dim=0, keepdim=True))
                mask_embed_avg = torch.mean(mask_embed, dim=0, keepdim=True)
                if is_distributed_training_run():
                    dist.all_reduce(mask_embed_avg)
                    mask_embed_avg = mask_embed_avg / dist.get_world_size()
                mask_embed_avg = mask_embed_avg.repeat(x.size(0), 1, 1)
                if z_feat.requires_grad:
                    assert mask_embed_avg.requires_grad

                dots = torch.einsum('bqc,bchw->bqhw', F.normalize(mask_embed_avg, dim=2), F.normalize(z_feat, dim=1))
            else:
                dots = torch.einsum('qc,bchw->bqhw', F.normalize(self.proto, dim=1), F.normalize(z_feat, dim=1))

            obj_attn = (dots / self.scale).softmax(dim=1) + self.eps

            slots = torch.einsum('bchw,bqhw->bqc', x, obj_attn / obj_attn.sum(dim=(2, 3), keepdim=True))

            out["dots"] = dots
            out["feat"] = z_feat
            out["obj_attn"] = obj_attn
        else:
            # do attention according to obj attention map
            obj_attn = F.normalize(z_feat, dim=1) if self.l2_norm else z_feat
            obj_attn /= self.scale
            obj_attn = obj_attn.view(b, self.num_heads, -1, h * w)  # (bs, h, d/h, h*w)

        if self.mask_type == "group":
            obj_attn = F.softmax(obj_attn, dim=-1)
            x = x.view(b, self.num_heads, -1, h*w)  # (bs, h, c/h, h*w)
            obj_val = torch.matmul(x, obj_attn.transpose(3, 2))    # (bs, h, c//h, d/h)
            obj_val = obj_val.view(b, c, obj_attn.shape[-2])    # (bs, c, d/h)
        elif self.mask_type == "max":
            obj_attn, _ = torch.max(obj_attn, dim=1)  # (bs, d/h, h*w)
            # obj_attn = torch.mean(obj_attn, dim=1)
            out["obj_attn"] = obj_attn
            obj_attn = F.softmax(obj_attn, dim=-1)
            obj_val = torch.matmul(x, obj_attn.transpose(2, 1))  # (bs, c, d/h)
        elif self.mask_type == "attn":
            obj_val = slots.transpose(2, 1)  # (bs, c, q)

        # projection
        obj_val = self.proj_obj(obj_val)  # (bs, d, q)

        if isTrain:
            self.update_proto(mask_embed)

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
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im, isTrain=True):
        out = self.backbone(im)
        out = self.neck(out, isTrain)
        return out


class FRAB_EMAN(nn.Module):
    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, loss_weight=0.5, mask_type="group", num_proto=8,
                 teacher_temp=0.04, student_temp=0.1, loss_w_cluster=0.1, **kwargs):
        super().__init__()

        self.base_m = m
        self.curr_m = m
        self.loss_weight = loss_weight
        self.loss_w_cluster = loss_w_cluster
        self.loss_w_obj = 0.02
        self.mask_type = mask_type
        assert mask_type in ["group", "max", "attn"]
        self.num_proto = num_proto
        self.student_temp = student_temp  # 0.1
        self.teacher_temp = teacher_temp  # 0.04

        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads, mask_type=mask_type,
                                     num_proto=num_proto, temp=self.teacher_temp)

        self.target_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads, mask_type=mask_type,
                                     num_proto=num_proto, temp=self.teacher_temp)
        self.predictor = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()
        self.predictor_obj = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor_obj.init_weights()
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

    def self_distill(self, q, k, use_sinkhorn=True, me_max=True):
        q_probs = F.log_softmax(q / self.student_temp, dim=-1)
        k_probs = F.softmax((k - self.center) / self.teacher_temp, dim=-1)

        if use_sinkhorn:
            k_probs = distributed_sinkhorn(k_probs)

        ce_loss = torch.sum(-k_probs * q_probs, dim=-1).mean()

        rloss = 0.
        if me_max:
            probs = F.softmax(q / self.student_temp, dim=-1)

            avg_probs = torch.mean(probs, dim=0)
            if is_distributed_training_run():
                dist.all_reduce(avg_probs)
                avg_probs = avg_probs / dist.get_world_size()
            # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

        loss = ce_loss + 1.0 * rloss

        return loss

    def assign_loss(self, online_1, online_2, target_1, target_2):
        z_o1, obj_o1, res_o1 = online_1
        z_o2, obj_o2, res_o2 = online_2
        z_t1, obj_t1, res_t1 = target_1
        z_t2, obj_t2, res_t2 = target_2

        loss = 0.5 * (self.self_distill(res_o1["dots"].permute(0, 2, 3, 1).flatten(0, 2),
                                        res_t1["dots"].permute(0, 2, 3, 1).flatten(0, 2))
                      + self.self_distill(res_o2["dots"].permute(0, 2, 3, 1).flatten(0, 2),
                                          res_t2["dots"].permute(0, 2, 3, 1).flatten(0, 2)))
        score_k1 = res_t1["dots"]
        score_k2 = res_t2["dots"]
        score = torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2)

        return loss, score

    def compute_unigrad_loss(self, pred, target, idxs=None):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        if idxs is not None:
            pos_term = self.mse_loss(dense_pred[idxs], dense_target[idxs])
        else:
            pos_term = self.mse_loss(dense_pred, dense_target)

        # compute neg term
        mask = torch.eye(pred.shape[1], device=pred.device).unsqueeze(0).repeat(pred.size(0), 1, 1)
        correlation = torch.matmul(pred, target.transpose(2, 1))  # b,c,c
        correlation = correlation * (1.0 - mask)
        neg_term = ((correlation**2).sum(-1) / target.shape[1]).reshape(-1)

        if idxs is not None:
            neg_term = torch.mean(neg_term[idxs])
        else:
            neg_term = torch.mean(neg_term)

        # # correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
        # correlation = torch.matmul(target.transpose(2, 1), target) / target.shape[1]  # b,c,c
        # # if is_distributed_training_run():
        # #     torch.distributed.all_reduce(correlation)
        # #     correlation = correlation / torch.distributed.get_world_size()
        #
        # # neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()
        # neg_term = torch.matmul(torch.matmul(pred, correlation), pred.transpose(2, 1))
        # neg_term = torch.diagonal(neg_term, dim1=-2, dim2=-1).mean()

        loss = pos_term + self.loss_w_obj * neg_term

        return loss

    def loss_func(self, online, target):
        z_o, obj_o, res_o = online
        z_t, obj_t, res_t = target

        # instance-level loss
        z_o_pred = self.predictor(z_o).squeeze(-1)
        z_t = z_t.squeeze(-1)
        loss_inst = self.mse_loss(z_o_pred, z_t)

        # object-level loss
        b, c, n = obj_o.shape
        obj_o_pred = self.predictor_obj(obj_o).transpose(2, 1)
        obj_t = obj_t.transpose(2, 1)

        score_q = res_o["dots"]
        score_k = res_t["dots"]
        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(
            -1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(
            -1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        # loss_obj = self.mse_loss(obj_o_pred.reshape(b*n, c)[idxs_q], obj_t.reshape(b*n, c)[idxs_q])
        # loss_obj = self.compute_unigrad_loss(obj_o_pred, obj_t, idxs_q)
        loss_obj = self.compute_unigrad_loss(obj_o_pred, obj_t)

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

        loss_cluster, score = self.assign_loss(proj_online_v1, proj_online_v2, proj_target_v1, proj_target_v2)
        loss = loss_base + loss_cluster * self.loss_w_cluster

        loss_pack = {}
        loss_pack["base"] = loss_base
        loss_pack["inst"] = (loss_inst1 + loss_inst2) * self.loss_weight
        loss_pack["obj"] = (loss_obj1 + loss_obj2) * (1 - self.loss_weight)
        loss_pack["clu"] = loss_cluster

        # self.update_center(score)

        return loss, loss_pack

    def extra_repr(self) -> str:
        parts = []
        for name in ["loss_weight", "mask_type", "num_proto", "teacher_temp", "loss_w_obj", "loss_w_cluster"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)


class FRAB(FRAB_EMAN):
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

    checkpoint = torch.load("./checkpoints/flr_r50_vgg_face.pth", map_location="cpu")
    state_dict = checkpoint['state_dict'] if "state_dict" in checkpoint else checkpoint

    model_func = get_model("FRAB")
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
        mask_type="attn",
        num_proto=8,
        teacher_temp=0.04,
    )
    print(model)

    x1 = torch.randn(16, 3, 224, 224)
    x2 = torch.randn(16, 3, 224, 224)
    out = model(x1, x2)
