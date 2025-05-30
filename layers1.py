import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims, dynamic_ib=False):
        super().__init__()
        self.dynamic_ib = dynamic_ib
        self.feature_dim = feature_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                # ResidualBlock(hidden_dim)
            ])
            prev_dim = hidden_dim

        if not dynamic_ib:
            encoder_layers.extend([
                nn.Linear(prev_dim, feature_dim),
                nn.ReLU()
            ])
            self.encoder = nn.Sequential(*encoder_layers)
        else:
            self.encoder = nn.Sequential(*encoder_layers)
            self.mu_layer = nn.Linear(prev_dim, feature_dim)
            self.logvar_layer = nn.Linear(prev_dim, feature_dim)

    def forward(self, x):
        x = self.encoder(x)
        if self.dynamic_ib:
            mu = self.mu_layer(x)
            logvar = self.logvar_layer(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return x


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super().__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        prev_dim = feature_dim
        for i, hidden_dim in enumerate(dims):
            self.decoder.add_module(f"Linear{i}", nn.Linear(prev_dim, hidden_dim))
            self.decoder.add_module(f"relu{i}", nn.ReLU())
            self.decoder.add_module(f"bn{i}", nn.BatchNorm1d(hidden_dim))
            self.decoder.add_module(f"drop{i}", nn.Dropout(0.5))
            self.decoder.add_module(f"res{i}", ResidualBlock(hidden_dim))
            prev_dim = hidden_dim
        self.decoder.add_module("Linear_out", nn.Linear(prev_dim, input_dim))
        self.decoder.add_module("relu_out", nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class AdaptiveFusion(nn.Module):
    """自适应融合层，基于注意力机制整合多个视图输出"""

    def __init__(self, num_views, feature_dim, mode='static'):
        """
        参数:
            num_views: 视图数量
            feature_dim: 特征维度
            mode: 'static', 'dynamic', 'attention'
        """
        super().__init__()
        self.num_views = num_views
        self.mode = mode

        if mode == 'static':
            # 静态可训练权重
            self.weights = nn.Parameter(torch.ones(num_views) / num_views)
            print(f"Initialized static fusion weights: {self.weights.data}")

        elif mode == 'dynamic':
            # 动态权重网络
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1, bias=True)  # 确保有偏置
            )
            # 初始化最后一层的偏置为0
            nn.init.zeros_(self.weight_net[-1].bias)

        elif mode == 'attention':
            # 简化的自注意力机制
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.attention_scale = feature_dim ** -0.5

    def adjust_bias_for_view(self, preferred_view_idx):
        """调整动态权重网络的偏置，使其偏好特定视图"""
        if self.mode != 'dynamic' or not hasattr(self, 'weight_net'):
            print("Can only adjust bias for dynamic fusion mode")
            return False

        # 获取最后一层
        last_layer = self.weight_net[-1]
        if not isinstance(last_layer, nn.Linear) or last_layer.bias is None:
            print("Last layer doesn't have bias to adjust")
            return False

        with torch.no_grad():
            # 重置所有偏置为轻微负值
            last_layer.bias.fill_(-0.5)
            # 设置偏好视图的偏置为正值
            if preferred_view_idx < self.num_views:
                print(f"Adjusting bias to prefer view {preferred_view_idx}")
                return True
            else:
                print(f"Invalid view index: {preferred_view_idx}")
                return False

    def forward(self, view_features):
        if self.mode == 'static':
            # 静态加权平均
            weights = F.softmax(self.weights, dim=0)
            return sum(w * feat for w, feat in zip(weights, view_features))

        elif self.mode == 'dynamic':
            # 动态权重计算
            weights = []
            for feat in view_features:
                w = self.weight_net(feat)  # [batch_size, 1]
                weights.append(w)

            # [batch_size, num_views]
            weights = torch.cat(weights, dim=1)
            weights = F.softmax(weights, dim=1)

            # 加权汇总
            output = torch.zeros_like(view_features[0])
            for i, feat in enumerate(view_features):
                output += feat * weights[:, i].unsqueeze(1)

            return output

        elif self.mode == 'attention':
            # 堆叠特征: [batch_size, num_views, feature_dim]
            stacked = torch.stack(view_features, dim=1)
            batch_size = stacked.shape[0]

            # 自注意力计算
            q = self.query_proj(stacked)  # [B, V, D]
            k = self.key_proj(stacked)  # [B, V, D]
            v = self.value_proj(stacked)  # [B, V, D]

            # 注意力分数
            attn = torch.bmm(q, k.transpose(1, 2)) * self.attention_scale  # [B, V, V]
            attn = F.softmax(attn, dim=2)

            # 加权汇总
            output = torch.bmm(attn, v)  # [B, V, D]

            # 取平均
            output = output.mean(dim=1)  # [B, D]

            return output


class EnhancedCVCLNetwork(nn.Module):
    """增强型CVCL网络，集成自适应融合"""

    def __init__(self,
                 num_views,
                 input_sizes,
                 dims,
                 dim_high_feature,
                 dim_low_feature,
                 num_clusters,
                 teacher_index=0,
                 fusion_mode='static'):
        super().__init__()
        self.teacher_index = teacher_index
        self.num_views = num_views
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # 编码器和解码器
        for idx in range(num_views):
            if idx == teacher_index:
                encoder = AutoEncoder(input_sizes[idx],
                                      dim_high_feature,
                                      dims,
                                      dynamic_ib=False)
            else:
                encoder = AutoEncoder(input_sizes[idx],
                                      dim_high_feature,
                                      dims,
                                      dynamic_ib=True)
            self.encoders.append(encoder)
            self.decoders.append(AutoDecoder(input_sizes[idx],
                                             dim_high_feature,
                                             dims))

        # 标签学习模块
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.ReLU(),
            nn.BatchNorm1d(dim_low_feature),
            nn.Dropout(0.5),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

        # 自适应融合层
        self.fusion_layer = AdaptiveFusion(
            num_views=num_views,
            feature_dim=dim_high_feature,
            mode=fusion_mode
        )

    @torch.no_grad()
    def set_teacher(self, new_teacher_idx: int):
        """动态切换教师视图"""
        if new_teacher_idx == self.teacher_index:
            return
        self.teacher_index = new_teacher_idx
        # 获取模型当前设备
        device = next(self.parameters()).device

        for idx, enc in enumerate(self.encoders):
            if idx == new_teacher_idx and enc.dynamic_ib:
                enc.dynamic_ib = False
                if hasattr(enc, 'mu_layer'):
                    del enc.mu_layer
                    del enc.logvar_layer
                last_linear_out = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_linear_out = m.out_features
                        break
                if last_linear_out != enc.feature_dim:
                    # enc.encoder.append(nn.Linear(last_linear_out, enc.feature_dim))
                    enc.encoder.append(nn.Linear(last_linear_out, enc.feature_dim).to(device))
                    enc.encoder.append(nn.ReLU())

            elif idx != new_teacher_idx and not enc.dynamic_ib:
                enc.dynamic_ib = True
                while isinstance(enc.encoder[-1], (nn.ReLU, nn.Linear)):
                    enc.encoder = enc.encoder[:-1]
                    if isinstance(enc.encoder[-1], nn.Linear):
                        break
                last_linear_out = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_linear_out = m.out_features
                        break

                enc.mu_layer = nn.Linear(last_linear_out, enc.feature_dim).to(device)
                enc.logvar_layer = nn.Linear(last_linear_out, enc.feature_dim).to(device)

    def forward(self, data_views):
        """前向传播，使用自适应融合"""
        label_probs, recons, features = [], [], []
        kl_loss_total = 0.0

        # 各视图前向传播
        for idx, view in enumerate(data_views):
            enc_out = self.encoders[idx](view)
            if isinstance(enc_out, tuple):
                z, mu, logvar = enc_out
                kl = -0.5 * torch.mean(torch.sum(
                    1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1))
                kl_loss_total += kl
            else:
                z = enc_out

            label_probs.append(self.label_learning_module(z))
            recons.append(self.decoders[idx](z))
            features.append(z)

        # 使用自适应层融合特征
        fused_feature = self.fusion_layer(features)

        # 计算融合标签概率
        fused_prob = self.label_learning_module(fused_feature)

        return label_probs, recons, features, kl_loss_total, fused_prob
