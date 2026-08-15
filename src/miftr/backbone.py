import torch
import torch.nn as nn
from einops import rearrange
from kornia.utils import create_meshgrid
from src.convnextv2.convnextv2 import convnextv2_nano


class CNN_backbone(nn.Module):
    def __init__(self, model_type='full'):
        super().__init__()
        if model_type.lower() == 'light':
            self.backbone = backbone_light()
        else:
            self.backbone = backbone_full()

    def forward(self, data):
        self.backbone(data)


class backbone_light(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = convnextv2_nano()
        # 去掉 ConvNeXt 的 norm、head 及后两级 downsample/stages
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        self.lin_4 = nn.Conv2d(80, 128, 1)
        self.lin_8 = nn.Conv2d(160, 256, 1)

    def forward(self, data):
        B, _, H, W = data['imagec_0'].shape
        x = torch.cat([data['imagec_0'], data['imagec_1']], 0)
        feature_pyramid = self.cnn.forward_features_8(x)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)  # nano
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)

        # 网格坐标
        scale = 8
        h_8, w_8 = H//scale, W//scale
        grid = [rearrange((create_meshgrid(h_8, w_8, False, device=x.device, dtype=x.dtype) * scale).squeeze(0), 'h w t->(h w) t')] * B  # kpt_xy
        grid_8 = torch.stack(grid, 0)

        # 更新 data
        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })


class backbone_full(nn.Module):
    def __init__(self):
        super().__init__()
        num_path = 1  # 共享的 CNN 骨干
        # num_path = 4  # 独立的 CNN 骨干（每个支路一个）,(不合理且实验证明没有意义)
        self.cnn = nn.ModuleList([convnextv2_nano() for _ in range(num_path)])
        dim4, dim8 = 80, 160
        # 去掉 ConvNeXt 的 norm、head 及后两级 downsample/stages
        for cnn in self.cnn:
            cnn.norm = None
            cnn.head = None
            cnn.downsample_layers[2] = None
            cnn.downsample_layers[3] = None
            cnn.stages[2] = None
            cnn.stages[3] = None

        # 可学习融合：concat 后再降维
        self.fuse_4 = nn.Conv2d(dim4 * 4, 128, 1)
        self.fuse_8 = nn.Conv2d(dim8 * 4, 256, 1)

        # Rot-Attention
        self.rotatt_4 = RotAttention(dim4 * 4, 4)
        self.rotatt_8 = RotAttention(dim8 * 4, 4)

    def forward(self, data):
        x0 = data['imagec_0']  # (B, C, H, W)
        x1 = data['imagec_1']
        B, _, H, W = x0.shape

        # 初始化最终融合后的特征（直接存储结果，不保留中间特征）
        feat_4_0 = feat_4_1 = feat_8_0 = feat_8_1 = None

        # 独立处理两个样本
        for i, x in enumerate([x0, x1]):
            # 初始化当前样本的多支路特征容器
            feats4, feats8 = [], []
            # 四个旋转支路
            for k in range(4):
                lyr = k if len(self.cnn) == 4 else 0
                # 旋转输入 + 特征提取
                x_rot = torch.rot90(x, k=k, dims=(2, 3))
                pyramid = self.cnn[lyr].forward_features_8(x_rot)
                # 逆旋转对齐
                f8 = torch.rot90(pyramid[8], k=(4 - k) % 4, dims=(2, 3))
                f4 = torch.rot90(pyramid[4], k=(4 - k) % 4, dims=(2, 3))
                feats8.append(f8)
                feats4.append(f4)

            # 多通道特征融合
            feats8 = self.rotatt_8(torch.cat(feats8, dim=1))  # (B, dim8*4, H/8, W/8)
            feats4 = self.rotatt_4(torch.cat(feats4, dim=1))  # (B, dim4*4, H/4, W/4)

            feats8 = self.fuse_8(feats8)  # (B, 256, H/8, W/8)
            feats4 = self.fuse_4(feats4)  # (B, 128, H/4, W/4)
            # 根据样本索引存储结果
            if i == 0:
                feat_8_0, feat_4_0 = feats8, feats4
            else:
                feat_8_1, feat_4_1 = feats8, feats4

        # 网格坐标
        scale = 8
        h8, w8 = H // scale, W // scale
        mesh = create_meshgrid(h8, w8, normalized_coordinates=False, device=x0.device, dtype=feat_8_0.dtype) * scale
        grid = rearrange(mesh.squeeze(0), 'h w t->(h w) t')
        grid_8 = grid.unsqueeze(0).repeat(B, 1, 1)

        # 更新 data
        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h8,
            'w_8': w8,
            'hw_8': h8 * w8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })
        return data


class RotAttention(nn.Module):
    def __init__(self, total_channels=640, num_groups=4):
        super().__init__()
        assert total_channels % num_groups == 0, "总通道数必须能被分组大小整除"
        self.num_groups = num_groups
        self.group_size = total_channels // num_groups
        self._norm = self.group_size ** 0.5

        # 共享的QKV投影层（每组独立处理）
        self.qkv_proj = nn.Conv2d(total_channels, total_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(total_channels, total_channels, kernel_size=1)

    def forward(self, x):
        """
        输入: (batch_size, total_channels, H, W)
        输出: (batch_size, total_channels, H, W)
        """
        B, C, H, W = x.shape
        num_groups = self.num_groups

        # 生成QKV [B, 3*640, H, W] -> 拆分为3张量
        qkv = self.qkv_proj(x)  # [B, 1920, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 每个[B, 640, H, W]

        # 分组处理（关键步骤）
        q = q.view(B, num_groups, self.group_size, H, W)  # [B, 4, 160, H, W]
        k = k.view(B, num_groups, self.group_size, H, W)
        v = v.view(B, num_groups, self.group_size, H, W)

        # 计算注意力分数（空间位置间独立计算）
        attn_scores = torch.einsum('bqcxy,bkcxy->bqkxy', q, k)  # [B, 4, 4, H, W]
        attn_scores = attn_scores / self._norm
        attn_weights = torch.softmax(attn_scores, dim=2)  # 沿group维度softmax

        # 加权求和
        out = torch.einsum('bqkxy,bkcxy->bqcxy', attn_weights, v)  # [B, 4, 160, H, W]
        out = out.reshape(B, C, H, W)  # 合并分组 [B, 640, H, W]

        # 输出投影
        return self.out_proj(out)


class MemoryEfficientRotAttention(nn.Module):
    def __init__(self, total_channels=640, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = total_channels // num_groups
        self._norm = self.group_size ** 0.5

        # 共享的QKV投影（输出通道减少为3*group_size）
        self.qkv_proj = nn.Conv2d(total_channels, num_groups * 3 * self.group_size, kernel_size=1)
        self.out_proj = nn.Conv2d(total_channels, total_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        output = torch.zeros_like(x)

        # 逐位置处理
        for h in range(H):
            for w in range(W):
                # 获取当前位置的640维特征 [B, C]
                x_pixel = x[:, :, h, w]  # [B, 640]

                # 生成当前像素的QKV [B, 3*640] -> [B, 3, num_groups, group_size]
                qkv = self.qkv_proj(x_pixel.unsqueeze(-1).unsqueeze(-1)).squeeze()  # [B, 3*640]
                qkv = qkv.view(B, 3, self.num_groups, self.group_size)  # [B, 3, 4, 160]
                q, k, v = qkv.unbind(1)  # 每个 [B, 4, 160]

                # 计算分组注意力 [B, 4, 160]
                attn = torch.einsum('bgc,bgc->bg', q, k) / self._norm  # [B, 4]
                attn = torch.softmax(attn, dim=-1)  # [B, 4]
                out_pixel = torch.einsum('bg,bgc->bgc', attn, v)  # [B, 4, 160]

                # 重组输出 [B, 640]
                output[:, :, h, w] = out_pixel.reshape(B, C)

        return self.out_proj(output)
