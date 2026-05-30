# ===== NEW CODE START: RSMIL components =====
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc(self.dropout(x))


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class FeatureReducer(nn.Module):
    def __init__(self, in_dim, out_dim=512, num_res_layers=0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(num_res_layers)])

    def forward(self, x):
        x = self.relu(self.fc(x))
        return self.res_blocks(x)


class GatedAttentionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_token=1):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, n_token)

    def forward(self, x):
        attn_v = self.attention_v(x)
        attn_u = self.attention_u(x)
        attn = self.attention_weights(attn_v * attn_u)
        return torch.transpose(attn, -1, -2)


class RSMILBaselineMIL(nn.Module):
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
        self.survival = survival

        self.dimreduction_1 = FeatureReducer(in_dim, inner_dim)
        self.dimreduction_2 = FeatureReducer(in_dim, inner_dim)

        stage_dim = inner_dim if dim_reduction else in_dim
        self.attention_1 = GatedAttentionHead(stage_dim, attn_dim, n_token_1)
        self.attention_2 = GatedAttentionHead(stage_dim, attn_dim, n_token_2)
        self.sub_classifiers = nn.ModuleList(
            [LinearClassifierHead(stage_dim, n_classes, dropout) for _ in range(n_token_2)]
        )
        self.slide_classifier = LinearClassifierHead(stage_dim, n_classes, dropout)

        self.apply(initialize_weights)

    def _mask_stage2(self, attn):
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

    def forward(self, x):
        if x.dim() == 4:
            if x.shape[0] != 1:
                raise RuntimeError(f'RSMIL baseline expects batch size 1, but received {x.shape[0]}.')
            x = x[0]
        if x.dim() != 3:
            raise RuntimeError(
                'RSMIL baseline expects hierarchical hr features with shape [N_lr, K_hr, D]. '
                f'Current input shape: {tuple(x.shape)}'
            )

        feat = x.float()
        stage1_input = self.dimreduction_1(feat) if self.use_dim_reduction else feat

        attn_1_raw = self.attention_1(stage1_input)
        attn_1 = F.softmax(attn_1_raw, dim=-1)

        bag_attn_1 = attn_1.mean(dim=1, keepdim=True)
        aggregated_stage1 = torch.bmm(bag_attn_1, feat).squeeze(1)

        stage2_input = self.dimreduction_2(aggregated_stage1) if self.use_dim_reduction else aggregated_stage1
        attn_2_raw = self.attention_2(stage2_input)
        attn_2_raw = self._mask_stage2(attn_2_raw)
        attn_2 = F.softmax(attn_2_raw, dim=1)

        pooled_stage2 = torch.mm(attn_2, stage2_input)
        sub_logits = torch.stack(
            [head(pooled_stage2[idx]).squeeze(0) for idx, head in enumerate(self.sub_classifiers)],
            dim=0
        )

        bag_attn_2 = attn_2.mean(dim=0, keepdim=True)
        bag_feat = torch.mm(bag_attn_2, stage2_input)
        logits = self.slide_classifier(bag_feat)

        if self.survival:
            y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            surv = torch.cumprod(1 - hazards, dim=1)
            return hazards, surv, y_hat, attn_2_raw, {'sub_preds': sub_logits, 'attn_1': attn_1, 'attn_2': attn_2}

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = {
            'sub_preds': sub_logits,
            'attn_1': attn_1,
            'attn_2': attn_2,
            'features': aggregated_stage1,
            'attn_raw': attn_2_raw,
        }
        return logits, y_prob, y_hat, attn_2_raw, results_dict

    def relocate(self):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimreduction_1 = self.dimreduction_1.to(target_device)
        self.dimreduction_2 = self.dimreduction_2.to(target_device)
        self.attention_1 = self.attention_1.to(target_device)
        self.attention_2 = self.attention_2.to(target_device)
        self.sub_classifiers = self.sub_classifiers.to(target_device)
        self.slide_classifier = self.slide_classifier.to(target_device)


RSMILBaseline = RSMILBaselineMIL
# ===== NEW CODE END: RSMIL components =====
