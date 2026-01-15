import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class AutoEncoder(nn.Module):
    """将输入编码到高维 feature 空间，支持可选的 VIB 动态瓶颈。"""
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        dims: List[int],
        dynamic_ib: bool = False
    ):
        super().__init__()
        self.dynamic_ib = dynamic_ib
        self.feature_dim = feature_dim

        layers = []
        prev = input_dim
        for h in dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2),
                # ResidualBlock(h),
            ]
            prev = h

        if not dynamic_ib:
            layers += [nn.Linear(prev, feature_dim), nn.ReLU()]
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = nn.Sequential(*layers)
            self.mu_layer     = nn.Linear(prev, feature_dim)
            self.logvar_layer = nn.Linear(prev, feature_dim)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        if self.dynamic_ib:
            mu     = self.mu_layer(x)
            logvar = self.logvar_layer(x)
            std    = torch.exp(0.5 * logvar)
            eps    = torch.randn_like(std)
            z      = mu + eps * std
            return z, mu, logvar
        else:
            return x


class AutoDecoder(nn.Module):
    """将高维 feature 重构回原始输入空间。"""
    def __init__(self, input_dim: int, feature_dim: int, dims: List[int]):
        super().__init__()
        self.decoder = nn.Sequential()
        rev = list(reversed(dims))
        prev = feature_dim
        for i, h in enumerate(rev):
            self.decoder.add_module(f"Linear{i}", nn.Linear(prev, h))
            self.decoder.add_module(f"ReLU{i}", nn.ReLU())
            self.decoder.add_module(f"BN{i}", nn.BatchNorm1d(h))
            self.decoder.add_module(f"Drop{i}", nn.Dropout(0.5))
            self.decoder.add_module(f"Res{i}", ResidualBlock(h))
            prev = h
        self.decoder.add_module("Linear_out", nn.Linear(prev, input_dim))
        self.decoder.add_module("ReLU_out", nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class AdaptiveFusion(nn.Module):
    """自适应融合层，支持 static/dynamic/attention 三种模式。"""
    def __init__(self, num_views: int, feature_dim: int, mode: str = 'static'):
        super().__init__()
        self.num_views = num_views
        self.mode = mode

        if mode == 'static':
            # 静态可训练权重
            self.weights = nn.Parameter(torch.ones(num_views) / num_views)
            print(f"Initialized static fusion weights: {self.weights.data}")

        elif mode == 'dynamic':
            # 动态权重网络：对每个视图 feature -> 一个标量
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1, bias=True),
            )
            # 用单独的向量 bias_vec 控制每个视图的初始偏好（length=num_views）
            self.bias_vec = nn.Parameter(torch.zeros(num_views))
            # 初始化 bias_vec 为平等小负值
            with torch.no_grad():
                self.bias_vec.fill_(-0.5)

        elif mode == 'attention':
            # QKV 自注意力融合多视图 feature
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.attention_scale = feature_dim ** -0.5

        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def adjust_bias_for_view(self, preferred_view_idx: int) -> bool:
        """dynamic 模式下微调 bias_vec，使其偏好指定视图。"""
        if self.mode != 'dynamic' or not hasattr(self, 'bias_vec'):
            print("Can only adjust bias for dynamic fusion mode")
            return False

        if not (0 <= preferred_view_idx < self.num_views):
            print(f"Invalid view idx: {preferred_view_idx}")
            return False

        with torch.no_grad():
            # 先重置为轻微负偏置
            self.bias_vec.fill_(-0.5)
            # 再给目标视图设置正偏置
            self.bias_vec[preferred_view_idx] = 0.5
            print(f"Adjusted bias_vec to prefer view {preferred_view_idx}")
        return True

    def forward(
        self,
        logits_list: List[Tensor],
        features:   List[Tensor] = None
    ) -> Tensor:
        """
        Args:
          logits_list: List[Tensor], 每个 Tensor 形状 [B, C]
          features:    List[Tensor], 每个 Tensor 形状 [B, D] (dynamic/attention 模式需要)
        Returns:
          static/dynamic -> fused_prob [B, C]
          attention    -> fused_feature [B, D]
        """
        # ---- static 分支 ----
        if self.mode == 'static':
            weights = F.softmax(self.weights, dim=0)  # [V]
            probs   = [F.softmax(l, dim=1) for l in logits_list]
            fused   = sum(w * p for w, p in zip(weights, probs))
            return fused

        # ---- dynamic 分支 ----
        if self.mode == 'dynamic':
            assert features is not None, "dynamic 模式下必须传入 features"
            # 1) 先按视图计算 raw score [B,1]
            w_logits = [self.weight_net(f) for f in features]      # list of [B,1]
            w_logits = torch.cat(w_logits, dim=1)                  # [B, V]
            # 2) 加上视图偏置 vec
            w_logits = w_logits + self.bias_vec.unsqueeze(0)       # [B, V]
            # 3) softmax 得到样本级 attention
            attn = F.softmax(w_logits, dim=1)                       # [B, V]
            # 4) 按样本加权各视图概率
            probs = [F.softmax(l, dim=1) for l in logits_list]
            fused = sum(attn[:, i].unsqueeze(1) * probs[i]
                        for i in range(self.num_views))
            return fused

        # ---- attention 分支 ----
        if self.mode == 'attention':
            assert features is not None, "attention 模式下必须传入 features"
            # 堆叠多视图 feature -> [B, V, D]
            stacked = torch.stack(features, dim=1)
            Q = self.query_proj(stacked)   # [B, V, D]
            K = self.key_proj(stacked)     # [B, V, D]
            Vv = self.value_proj(stacked)  # [B, V, D]
            # 自注意力分数 [B, V, V]
            attn = torch.bmm(Q, K.transpose(1,2)) * self.attention_scale
            attn = F.softmax(attn, dim=2)
            # 加权 V -> [B, V, D]
            out = torch.bmm(attn, Vv)
            # 在视图维度取平均 -> [B, D]
            h_fused = out.mean(dim=1)
            return h_fused

        raise ValueError(f"Unknown fusion mode: {self.mode}")


class AD_MVC(nn.Module):

    def __init__(
        self,
        num_views: int,
        input_sizes: List[int],
        dims: List[int],
        dim_high_feature: int,
        dim_low_feature: int,
        num_clusters: int,
        teacher_index: int = 0,
        fusion_mode: str = 'static'
    ):
        super().__init__()
        self.num_views     = num_views
        self.teacher_index = teacher_index

        # --- 各视图自编码器 & 解码器 ---
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for v in range(num_views):
            dynamic_ib = (v != teacher_index)
            self.encoders.append(
                AutoEncoder(input_sizes[v], dim_high_feature, dims, dynamic_ib)
            )
            self.decoders.append(
                AutoDecoder(input_sizes[v], dim_high_feature, dims)
            )

        # --- 单视图标签学习 head: feature -> num_clusters 概率 ---
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.ReLU(),
            nn.BatchNorm1d(dim_low_feature),
            nn.Dropout(0.5),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1),
        )

        # --- 每视图 raw logits 层，用于 static/dynamic 融合 ---
        self.logit_layer_v = nn.ModuleList([
            nn.Linear(dim_high_feature, num_clusters)
            for _ in range(num_views)
        ])

        # --- 自适应融合层 ---
        self.fusion_layer = AdaptiveFusion(
            num_views=num_views,
            feature_dim=dim_high_feature,
            mode=fusion_mode
        )

    @torch.no_grad()
    def set_teacher(self, new_teacher_idx: int):
        """动态切换教师视图，控制 VIB 瓶颈 on/off。"""
        if new_teacher_idx == self.teacher_index:
            return
        self.teacher_index = new_teacher_idx
        device = next(self.parameters()).device

        for v, enc in enumerate(self.encoders):
            if v == new_teacher_idx and enc.dynamic_ib:
                # 关闭瓶颈
                enc.dynamic_ib = False
                del enc.mu_layer
                del enc.logvar_layer
                # 补充 Linear->ReLU 到 feature_dim
                last_dim = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_dim = m.out_features
                        break
                if last_dim != enc.feature_dim:
                    enc.encoder.add_module("to_feat",
                                           nn.Linear(last_dim, enc.feature_dim).to(device))
                    enc.encoder.add_module("relu_feat", nn.ReLU())
            elif v != new_teacher_idx and not enc.dynamic_ib:
                # 打开瓶颈
                enc.dynamic_ib = True
                # 回退末尾到隐藏层
                while isinstance(enc.encoder[-1], (nn.ReLU, nn.Linear)):
                    enc.encoder = enc.encoder[:-1]
                last_dim = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_dim = m.out_features
                        break
                enc.mu_layer     = nn.Linear(last_dim, enc.feature_dim).to(device)
                enc.logvar_layer = nn.Linear(last_dim, enc.feature_dim).to(device)

    def forward(self, data_views: List[Tensor]):
        """
        Args:
          data_views: List[Tensor], 每个 Tensor 形状 [B, input_dim_v]
        Returns:
          label_probs:   List[Tensor] 每视图 [B, C]
          recons:        List[Tensor] 每视图 [B, input_dim_v]
          features:      List[Tensor] 每视图 [B, D]
          kl_loss_total: float        VIB KL loss 之和
          fused_prob:    Tensor [B, C] 最终融合概率 (static/dynamic/attention)
        """
        recons, features = [], []
        kl_loss_total = 0.0

        # --- 各视图前向: 编码 -> VIB KL -> 重构 -> 收集 feature ---
        for v, x in enumerate(data_views):
            out = self.encoders[v](x)
            if isinstance(out, tuple):
                z, mu, logvar = out
                kl_loss_total += -0.5 * torch.mean(
                    (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                )
            else:
                z = out
            recons.append(self.decoders[v](z))
            features.append(z)

        # --- 单视图 raw logits & probs ---
        assert len(features) == self.num_views
        logits_list = [
            self.logit_layer_v[v](features[v])
            for v in range(self.num_views)
        ]
        label_probs = [F.softmax(l, dim=1) for l in logits_list]

        # --- 自适应融合 (static/dynamic/attention) ---
        mode = self.fusion_layer.mode
        if mode == 'static':
            fused_prob = self.fusion_layer(logits_list)
        elif mode == 'dynamic':
            fused_prob = self.fusion_layer(logits_list, features)
        else:  # 'attention'
            h_fused = self.fusion_layer(logits_list, features)
            fused_prob = self.label_learning_module(h_fused)

        return label_probs, recons, features, kl_loss_total, fused_prob
