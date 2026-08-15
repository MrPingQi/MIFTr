import torch
import torch.nn as nn
from einops.einops import rearrange


class KeypointEncoder_wo_score(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


class TransLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.transpose(1,2)).transpose(1,2)


class TransLN_2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        _, _, h, _ = x.shape
        x = rearrange(x, 'b d h w->b (h w) d')
        x = self.ln(x)
        return rearrange(x, 'b (h w) d->b d h w', h=h)


class up_conv4(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(up_conv4, self).__init__()
        self.lin = nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1)
        self.inter = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transconv = nn.ConvTranspose2d(dim_in, dim_mid, kernel_size=2, stride=2)
        self.cbr = nn.Sequential(
            nn.Conv2d(dim_mid, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_inter = self.inter(self.lin(x))
        x_conv = self.transconv(x)
        x = self.cbr(x_inter+x_conv)
        return x


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(TransLN(channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class GLU(nn.Module):
    def __init__(self, dim, mid_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(dim, mid_dim, bias=False)
        self.V = nn.Linear(dim, mid_dim, bias=False)
        self.W2 = nn.Linear(mid_dim, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, feat):
        feat_act = self.act(self.W(feat))
        feat_linear = self.V(feat)
        feat = feat_act * feat_linear
        feat = self.W2(feat)
        return feat


class GLU_3(nn.Module):
    def __init__(self, dim, mid_dim):
        super(GLU_3, self).__init__()
        self.W = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.V = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(mid_dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(True)
        self.act = nn.GELU()

    def forward(self, feat):
        feat_act = self.act(self.W(feat))
        feat_linear = self.V(feat)
        feat = feat_act * feat_linear
        feat = self.W2(feat)
        return feat
        # return self.V(feat)


class conv_3(nn.Module):
    def __init__(self, dim, ks):
        super(conv_3, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks//2, bias=False)

    def forward(self, feat):
        feat = feat + self.conv(feat)
        return feat


class poolformer(nn.Module):
    def __init__(self, ks = 3):
        super(poolformer, self).__init__()
        self.pool = nn.AvgPool2d(ks, stride=1, padding=ks//2, count_include_pad=False)

    def forward(self, feat):
        feat = feat + self.pool(feat)
        return feat


class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, dim1, dim2, factor=1):
        super(MLPMixerEncoderLayer, self).__init__()

        self.mlp1 = nn.Sequential(nn.Linear(dim1, dim1*factor),
                                  nn.GELU(),
                                  nn.Linear(dim1*factor, dim1))
        self.mlp2 = nn.Sequential(nn.Linear(dim2, dim2*factor),
                                  nn.LayerNorm(dim2*factor),
                                  nn.GELU(),
                                  nn.Linear(dim2*factor, dim2))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            x_mask (torch.Tensor): [N, L] (optional)
        """
        x = x + self.mlp1(x)
        x = x.transpose(1, 2)
        x = x + self.mlp2(x)
        return x.transpose(1, 2)
