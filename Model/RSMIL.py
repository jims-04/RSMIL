import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .RSMILMixer import MixerBlock, initialize_weights
    from .RSMILComponents import GatedAttentionHead, LinearClassifierHead, FeatureReducer
except ImportError:
    from Model.RSMILMixer import MixerBlock, initialize_weights
    from Model.RSMILComponents import GatedAttentionHead, LinearClassifierHead, FeatureReducer


RSMIL_MODEL_TYPES = (
    "rsmil_local",
    "rsmil_region",
    "rsmil_stage2",
    "rsmil_stage1",
    "rsmil_dual",
)


def build_rsmil_model(model_type, **kwargs):
    builders = {
        "rsmil_local": RSMILLocalMixer,
        "rsmil_region": RSMILRegionMixer,
        "rsmil_stage2": RSMILReplaceStage2,
        "rsmil_stage1": RSMILReplaceStage1,
        "rsmil_dual": RSMILDual,
    }
    if model_type not in builders:
        raise ValueError(f"Unsupported RSMIL model type: {model_type}")
    return builders[model_type](**kwargs)


def _build_projection(in_dim, out_dim, act, dropout):
    layers = [nn.Linear(in_dim, out_dim)]
    act_name = act.lower()
    if act_name == "relu":
        layers.append(nn.ReLU())
    elif act_name == "gelu":
        layers.append(nn.GELU())
    else:
        raise ValueError(f"Unsupported activation: {act}")
    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _cosine_token_scores(tokens, pooled):
    pooled = F.normalize(pooled, dim=-1).unsqueeze(1)
    tokens = F.normalize(tokens, dim=-1)
    return torch.sum(tokens * pooled, dim=-1) * math.sqrt(tokens.shape[-1])


class RSMILSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        embed_dim=256,
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        act="relu",
        dropout=0.25,
    ):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError("rsmil_mixer_embed_dim must be divisible by 4.")

        self.in_proj = _build_projection(in_dim, embed_dim, act, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Identity() if out_dim == embed_dim else nn.Linear(embed_dim, out_dim)

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

        self.apply(initialize_weights)

    def _to_grid(self, tokens):
        num_tokens = tokens.shape[1]
        side = int(math.ceil(math.sqrt(num_tokens)))
        add_length = side * side - num_tokens
        if add_length > 0:
            tokens = torch.cat([tokens, tokens[:, :add_length, :]], dim=1)
        return tokens, num_tokens, side

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        tokens = self.in_proj(x.float())
        tokens, num_tokens, side = self._to_grid(tokens)
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

        for layer in self.layers:
            tokens = layer(tokens)

        seq_tokens = tokens.flatten(2).transpose(1, 2)[:, :num_tokens, :]
        pooled = self.pool(tokens).flatten(1)
        seq_tokens = self.norm(seq_tokens)
        pooled = self.norm(pooled)

        seq_tokens = self.out_proj(seq_tokens)
        pooled = self.out_proj(pooled)
        return seq_tokens, pooled


class BaseRSMIL(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        n_classes=2,
        inner_dim=128,
        attn_dim=128,
        dropout=0.25,
        n_token_1=1,
        n_token_2=1,
        n_masked_patch_2=20,
        mask_drop=0.4,
        dim_reduction=False,
        rsmil_mixer_embed_dim=256,
        rsmil_mixer_depths=(2, 2, 2, 2),
        rsmil_mixer_drop_path_rate=0.1,
        rsmil_mixer_act="relu",
        rsmil_mixer_dropout=0.25,
        survival=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.n_token_1 = n_token_1
        self.n_token_2 = n_token_2
        self.n_masked_patch_2 = n_masked_patch_2
        self.mask_drop = mask_drop
        self.use_dim_reduction = dim_reduction
        self.stage2_feature_dim = inner_dim if dim_reduction else in_dim
        self.survival = survival

        self.dimreduction_1 = FeatureReducer(in_dim, inner_dim)
        self.dimreduction_2 = FeatureReducer(in_dim, inner_dim)
        self.attention_1 = GatedAttentionHead(self.stage2_feature_dim, attn_dim, n_token_1)
        self.attention_2 = GatedAttentionHead(self.stage2_feature_dim, attn_dim, n_token_2)
        self.sub_classifiers = nn.ModuleList(
            [LinearClassifierHead(self.stage2_feature_dim, n_classes, dropout) for _ in range(n_token_2)]
        )
        self.slide_classifier = LinearClassifierHead(self.stage2_feature_dim, n_classes, dropout)

        self.mixer_kwargs = {
            "embed_dim": rsmil_mixer_embed_dim,
            "depths": tuple(rsmil_mixer_depths),
            "drop_path_rate": rsmil_mixer_drop_path_rate,
            "act": rsmil_mixer_act,
            "dropout": rsmil_mixer_dropout,
        }

        self.apply(initialize_weights)

    def _unwrap_slide_tensor(self, x):
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise RuntimeError(f"RSMIL expects batch size 1, but received {x.shape[0]}.")
            return x[0]
        if x.ndim == 3:
            return x
        raise RuntimeError(
            "RSMIL expects hierarchical hr features with shape [N_lr, K_hr, D]. "
            f"Current input shape: {tuple(x.shape)}"
        )

    def _unwrap_region_tensor(self, x):
        if x.ndim == 3 and x.shape[0] == 1:
            return x[0]
        if x.ndim in (2, 3):
            return x
        raise RuntimeError(f"Unsupported region feature shape: {tuple(x.shape)}")

    def _apply_region_mask(self, attn):
        if not self.training or self.n_masked_patch_2 <= 0:
            return attn

        n_tokens, n_items = attn.shape
        n_masked = min(self.n_masked_patch_2, n_items)
        n_random = int(n_masked * self.mask_drop)
        if n_random <= 0:
            return attn

        _, indices = torch.topk(attn, n_masked, dim=-1)
        rand_selected = torch.argsort(torch.rand(*indices.shape, device=attn.device), dim=-1)[:, :n_random]
        masked_indices = indices.gather(-1, rand_selected)
        random_mask = torch.ones_like(attn)
        random_mask.scatter_(-1, masked_indices, 0)
        return attn.masked_fill(random_mask == 0, -1e9)

    def _compute_stage1_attention(self, local_features):
        attention_features = self.dimreduction_1(local_features) if self.use_dim_reduction else local_features
        attn_raw = self.attention_1(attention_features)
        attn = F.softmax(attn_raw, dim=-1)

        bag_attn = attn.mean(dim=1, keepdim=True)
        region_features = torch.bmm(bag_attn, local_features).squeeze(1)
        return region_features, attn, attn_raw

    def _classify_with_stage2_attention(self, region_features, stage2_tokens=None, value_features=None):
        if stage2_tokens is None:
            stage2_tokens = self.dimreduction_2(region_features) if self.use_dim_reduction else region_features
        if value_features is None:
            value_features = stage2_tokens

        attn_raw = self.attention_2(stage2_tokens)
        attn_raw = self._apply_region_mask(attn_raw)
        attn = F.softmax(attn_raw, dim=1)

        head_features = torch.mm(attn, value_features)
        sub_logits = torch.stack(
            [head(head_features[idx]).squeeze(0) for idx, head in enumerate(self.sub_classifiers)],
            dim=0,
        )

        bag_attn = attn.mean(dim=0, keepdim=True)
        bag_feat = torch.mm(bag_attn, value_features)
        logits = self.slide_classifier(bag_feat)
        return sub_logits, logits, attn, attn_raw, bag_feat

    def _classify_with_mixer_stage2(self, region_features, region_encoder):
        tokens, pooled = region_encoder(region_features)
        tokens = tokens.squeeze(0)
        pooled = pooled.squeeze(0)

        attn_raw = _cosine_token_scores(tokens.unsqueeze(0), pooled.unsqueeze(0))
        attn_raw = self._apply_region_mask(attn_raw)
        attn = F.softmax(attn_raw, dim=1)
        bag_feat = torch.mm(attn, tokens)

        sub_logits = torch.stack([head(bag_feat[0]).squeeze(0) for head in self.sub_classifiers], dim=0)
        logits = self.slide_classifier(bag_feat)
        return sub_logits, logits, attn, attn_raw, bag_feat

    def _aggregate_regions_with_mixer(self, local_features, local_encoder):
        tokens, pooled = local_encoder(local_features)
        attn_raw = _cosine_token_scores(tokens, pooled).unsqueeze(1)
        attn = F.softmax(attn_raw, dim=-1)

        region_features = torch.bmm(attn.mean(dim=1, keepdim=True), tokens).squeeze(1)
        return region_features, attn, attn_raw

    def _pack_outputs(self, logits, sub_logits, attn_1, attn_2, region_features, attn_raw):
        if self.survival:
            y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            surv = torch.cumprod(1 - hazards, dim=1)
            return hazards, surv, y_hat, attn_raw, {
                "sub_preds": sub_logits,
                "attn_1": attn_1,
                "attn_2": attn_2,
                "features": region_features,
                "attn_raw": attn_raw,
            }

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = {
            "sub_preds": sub_logits,
            "attn_1": attn_1,
            "attn_2": attn_2,
            "features": region_features,
            "attn_raw": attn_raw,
        }
        return logits, y_prob, y_hat, attn_raw, results_dict

    def forward(self, x):
        raise NotImplementedError

    def classify(self, x, average_block2_weights=False):
        raise NotImplementedError

    def get_hr_fa(self, x):
        raise NotImplementedError


class RSMILLocalMixer(BaseRSMIL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_mixer = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            **self.mixer_kwargs,
        )

    def _mix_locals(self, local_features):
        mixed_features, _ = self.local_mixer(local_features)
        return mixed_features

    def forward(self, x):
        feat = self._unwrap_slide_tensor(x)
        mixed_feat = self._mix_locals(feat)
        region_features, attn_1, _ = self._compute_stage1_attention(mixed_feat)
        sub_logits, logits, attn_2, attn_raw, _ = self._classify_with_stage2_attention(region_features)
        return self._pack_outputs(logits, sub_logits, attn_1, attn_2, region_features, attn_raw)

    def classify(self, x, average_block2_weights=False):
        region_features = self._unwrap_region_tensor(x)
        _, logits, attn_2, _, _ = self._classify_with_stage2_attention(region_features)
        if average_block2_weights:
            return attn_2.mean(dim=0, keepdim=True)
        return logits, attn_2

    def get_hr_fa(self, x):
        feat = self._unwrap_slide_tensor(x)
        mixed_feat = self._mix_locals(feat)
        region_features, _, _ = self._compute_stage1_attention(mixed_feat)
        return region_features


class RSMILRegionMixer(BaseRSMIL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_mixer = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.stage2_feature_dim,
            **self.mixer_kwargs,
        )

    def _mix_regions(self, region_features):
        mixed_features, _ = self.region_mixer(region_features)
        return mixed_features.squeeze(0)

    def forward(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, attn_1, _ = self._compute_stage1_attention(feat)
        mixed_features = self._mix_regions(region_features.unsqueeze(0))
        sub_logits, logits, attn_2, attn_raw, _ = self._classify_with_stage2_attention(
            region_features,
            stage2_tokens=mixed_features,
            value_features=mixed_features,
        )
        return self._pack_outputs(logits, sub_logits, attn_1, attn_2, region_features, attn_raw)

    def classify(self, x, average_block2_weights=False):
        region_features = self._unwrap_region_tensor(x)
        mixed_features = self._mix_regions(region_features.unsqueeze(0))
        _, logits, attn_2, _, _ = self._classify_with_stage2_attention(
            region_features,
            stage2_tokens=mixed_features,
            value_features=mixed_features,
        )
        if average_block2_weights:
            return attn_2.mean(dim=0, keepdim=True)
        return logits, attn_2

    def get_hr_fa(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, _, _ = self._compute_stage1_attention(feat)
        return region_features


class RSMILReplaceStage2(BaseRSMIL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_encoder = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.stage2_feature_dim,
            **self.mixer_kwargs,
        )

    def forward(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, attn_1, _ = self._compute_stage1_attention(feat)
        sub_logits, logits, attn_2, attn_raw, _ = self._classify_with_mixer_stage2(
            region_features,
            self.region_encoder,
        )
        return self._pack_outputs(logits, sub_logits, attn_1, attn_2, region_features, attn_raw)

    def classify(self, x, average_block2_weights=False):
        region_features = self._unwrap_region_tensor(x)
        _, logits, attn_2, _, _ = self._classify_with_mixer_stage2(region_features, self.region_encoder)
        if average_block2_weights:
            return attn_2.mean(dim=0, keepdim=True)
        return logits, attn_2

    def get_hr_fa(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, _, _ = self._compute_stage1_attention(feat)
        return region_features


class RSMILReplaceStage1(BaseRSMIL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_encoder = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            **self.mixer_kwargs,
        )

    def forward(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, attn_1, _ = self._aggregate_regions_with_mixer(feat, self.local_encoder)
        sub_logits, logits, attn_2, attn_raw, _ = self._classify_with_stage2_attention(region_features)
        return self._pack_outputs(logits, sub_logits, attn_1, attn_2, region_features, attn_raw)

    def classify(self, x, average_block2_weights=False):
        region_features = self._unwrap_region_tensor(x)
        _, logits, attn_2, _, _ = self._classify_with_stage2_attention(region_features)
        if average_block2_weights:
            return attn_2.mean(dim=0, keepdim=True)
        return logits, attn_2

    def get_hr_fa(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, _, _ = self._aggregate_regions_with_mixer(feat, self.local_encoder)
        return region_features


class RSMILDual(BaseRSMIL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_encoder = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            **self.mixer_kwargs,
        )
        self.region_encoder = RSMILSequenceEncoder(
            in_dim=self.in_dim,
            out_dim=self.stage2_feature_dim,
            **self.mixer_kwargs,
        )

    def forward(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, attn_1, _ = self._aggregate_regions_with_mixer(feat, self.local_encoder)
        sub_logits, logits, attn_2, attn_raw, _ = self._classify_with_mixer_stage2(
            region_features,
            self.region_encoder,
        )
        return self._pack_outputs(logits, sub_logits, attn_1, attn_2, region_features, attn_raw)

    def classify(self, x, average_block2_weights=False):
        region_features = self._unwrap_region_tensor(x)
        _, logits, attn_2, _, _ = self._classify_with_mixer_stage2(region_features, self.region_encoder)
        if average_block2_weights:
            return attn_2.mean(dim=0, keepdim=True)
        return logits, attn_2

    def get_hr_fa(self, x):
        feat = self._unwrap_slide_tensor(x)
        region_features, _, _ = self._aggregate_regions_with_mixer(feat, self.local_encoder)
        return region_features
