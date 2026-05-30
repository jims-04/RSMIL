import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class TokenMixer(nn.Module):
    def __init__(
        self,
        in_channels,
        mem_dim=None,
        att_head_dim=1,
        if_att=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.att_head_dim = att_head_dim
        self.if_att = if_att
        self.gc = mem_dim
        self.att_head_num = int(self.gc / att_head_dim)
        self.split_indexes = (in_channels - 3 * self.gc, self.gc, self.gc, self.gc)

        if self.if_att:
            self.dwconv1 = nn.Conv2d(self.gc, self.gc, kernel_size=3, padding=1, groups=self.gc)
            self.dwconv2 = nn.Conv2d(self.gc, self.gc, kernel_size=3, padding=1, groups=self.gc)
            self.qk = nn.Conv2d(self.gc, 2 * self.gc, kernel_size=1)
            self.elu = nn.ELU()
        else:
            self.dwconvatt_1 = nn.Conv2d(self.gc, self.gc, kernel_size=(1, 3), padding=(0, 1), groups=self.gc)
            self.dwconvatt_2 = nn.Conv2d(self.gc, self.gc, kernel_size=(3, 1), padding=(1, 0), groups=self.gc)
            self.dwconv1 = nn.Conv2d(self.gc, self.gc, kernel_size=1, padding=0, groups=self.gc)
            self.dwconv2 = nn.Conv2d(self.gc, self.gc, kernel_size=3, padding=1, groups=self.gc)

    def grasp_memory(self, x):
        mem, feat = torch.split(x, [self.split_indexes[0], sum(self.split_indexes[1:])], dim=1)
        return mem, feat

    def asymmetric_decouple(self, x):
        feat_att, feat_conv1, feat_conv2 = torch.split(x, self.split_indexes[1:], dim=1)
        return feat_att, feat_conv1, feat_conv2

    def forward(self, x):
        bsz, _, height, width = x.shape
        mem, feats = self.grasp_memory(x)
        feat_att, feat_conv1, feat_conv2 = self.asymmetric_decouple(feats)

        feat_conv1 = self.dwconv1(feat_conv1)
        feat_conv2 = self.dwconv2(feat_conv2)

        if self.if_att:
            x_qk = self.qk(feat_att).reshape(bsz, -1, height * width)
            (q, k), v = x_qk.split(self.gc, dim=1), feat_att.reshape(bsz, -1, height * width)
            q = q.reshape(bsz, self.att_head_num, self.att_head_dim, height * width).transpose(-2, -1)
            k = k.reshape(bsz, self.att_head_num, self.att_head_dim, height * width).transpose(-2, -1)
            v = v.reshape(bsz, self.att_head_num, self.att_head_dim, height * width).transpose(-2, -1)
            q = self.elu(q) + 1.0
            k = self.elu(k) + 1.0
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) @ v) / (height * width)
            feat_att = (q @ kv * z).transpose(-2, -1).reshape(bsz, self.gc, height, width)
        else:
            feat_att = self.dwconvatt_1(feat_att) + self.dwconvatt_2(feat_att)

        feats = torch.cat([feat_att, feat_conv1, feat_conv2], dim=1)
        return mem, feats


class InterModule(nn.Module):
    def __init__(
        self,
        in_features,
        ratio,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        bias=True,
        drop=0.0,
        mem_dim=None,
        last=False,
    ):
        super().__init__()
        bias = to_2tuple(bias)
        self.last = last
        ratio1, ratio2 = ratio

        dim1 = mem_dim * 3
        self.inter1_norm = norm_layer(dim1) if norm_layer else nn.Identity()
        self.inter1_fc1 = nn.Conv2d(dim1, ratio1 * dim1, kernel_size=1, bias=bias[0])
        self.inter1_act = act_layer()
        self.inter1_conv = nn.Conv2d(ratio1 * dim1, ratio1 * dim1, kernel_size=3, padding=1, groups=ratio1 * dim1)
        self.inter1_drop = nn.Dropout(drop)
        self.inter1_fc2 = nn.Conv2d(ratio1 * dim1, dim1, kernel_size=1, bias=bias[1])

        dim2 = int(in_features)
        self.inter2_norm = norm_layer(dim2) if norm_layer else nn.Identity()
        self.inter2_fc1 = nn.Conv2d(dim2, ratio2 * dim2, kernel_size=1, bias=bias[0])
        self.inter2_act = act_layer()
        self.inter2_conv = nn.Conv2d(ratio2 * dim2, ratio2 * dim2, kernel_size=3, padding=1, groups=ratio2 * dim2)
        self.inter2_drop = nn.Dropout(drop)
        if not self.last:
            self.inter2_fc2 = nn.Conv2d(ratio2 * dim2, dim2, kernel_size=1, bias=bias[1])

    def forward(self, x):
        mem, feats = x
        res = feats
        feats = self.inter1_norm(feats)
        feats = self.inter1_fc1(feats)
        feats = self.inter1_act(feats)
        feats = self.inter1_conv(feats)
        feats = self.inter1_drop(feats)
        feats = self.inter1_fc2(feats) + res

        cat = torch.cat([mem, feats], dim=1)
        res = cat
        cat = self.inter2_norm(cat)
        cat = self.inter2_fc1(cat)
        cat = self.inter2_act(cat)
        cat = self.inter2_conv(cat)
        cat = self.inter2_drop(cat)
        if not self.last:
            cat = self.inter2_fc2(cat) + res
        return cat


class MixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        mem_dim=None,
        token_mixer=TokenMixer,
        norm_layer=nn.BatchNorm2d,
        inter_layer=InterModule,
        act_layer=nn.GELU,
        ratio=4,
        ls_init_value=1e-6,
        drop_path=0.0,
        att_head_dim=1,
        last=False,
        if_att=False,
    ):
        super().__init__()
        self.last = last
        self.token_mixer = token_mixer(
            in_channels=dim,
            mem_dim=mem_dim,
            att_head_dim=att_head_dim,
            if_att=if_att,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.inter_layer = inter_layer(
            dim,
            mem_dim=mem_dim,
            ratio=ratio,
            act_layer=act_layer,
            last=last,
        )
        if not self.last:
            self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.inter_layer(x)
        if not self.last:
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))
            x = self.drop_path(x) + shortcut
        else:
            x = self.drop_path(x)
        return x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class RSMILMixerMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        n_classes,
        dropout,
        act,
        survival=False,
        embed_dim=256,
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
    ):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for RSMIL mixer blocks.")

        self._fc1 = [nn.Linear(in_dim, embed_dim)]
        if act.lower() == "relu":
            self._fc1 += [nn.ReLU()]
        elif act.lower() == "gelu":
            self._fc1 += [nn.GELU()]
        else:
            raise ValueError(f"Unsupported activation: {act}")
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)

        mem_dim = embed_dim // 4
        total_blocks = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.layers = nn.ModuleList()

        block_idx = 0
        for stage_idx, depth in enumerate(depths):
            if_att = stage_idx >= 2
            ratio = (2, 4) if stage_idx < 2 else (4, 4)
            att_head_dim = 1 if not if_att else max(1, mem_dim // 8)
            for _ in range(depth):
                self.layers.append(
                    MixerBlock(
                        dim=embed_dim,
                        mem_dim=mem_dim,
                        ratio=ratio,
                        drop_path=drop_paths[block_idx],
                        if_att=if_att,
                        att_head_dim=att_head_dim,
                        last=False,
                    )
                )
                block_idx += 1

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.n_classes = n_classes
        self.survival = survival

        self.apply(initialize_weights)

    def _to_grid(self, h):
        num_instances = h.shape[1]
        side = int(np.ceil(np.sqrt(num_instances)))
        add_length = side * side - num_instances
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
        return h, side, side

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        h = x.float()
        h = self._fc1(h)
        h, grid_h, grid_w = self._to_grid(h)

        # Turn the bag into a padded 2D token map so CARE blocks can mix neighborhoods.
        h = h.transpose(1, 2).reshape(h.shape[0], h.shape[2], grid_h, grid_w)

        for layer in self.layers:
            h = layer(h)

        h = self.pool(h).flatten(1)
        h = self.norm(h)

        logits = self.classifier(h)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)
        self.pool = self.pool.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)




