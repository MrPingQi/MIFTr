import math
import torch
from torch import nn
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None
from src.utils.profiler import PassThroughProfiler
from src.miftr.utils.utils import GLU_3
from src.mamba_module.mamba.mamba_v1 import Mamba
from src.mamba_module.mamba.mamba_v2 import Mamba2
from src.mamba_module.mamba_vision.mamba_vision import MambaVisionMixer as MambaV


# ———————————————————————————————————————————————————— Mamba Blocks ————————————————————————————————————————————————————
class Block(nn.Module):
    def __init__(self,
                 dim,
                 mixer_cls,
                 norm_cls=nn.LayerNorm,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 drop_path=0.,):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, desc, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            desc = desc.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return desc + hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_devide_out=False,
    init_layer_scale=None,
    mamba_type="vision",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if mamba_type.lower() == "vision":
        mixer_cls = partial(MambaV, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif mamba_type.lower() == "v2":
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba,  layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ————————————————————————————————————————————————— CrossVM Operations —————————————————————————————————————————————————
def scan_crossVM(desc0, desc1, step_size, sym_flag=False):
    B, C, org_h, org_w = desc0.shape
    H = org_h // step_size
    W = org_w // step_size * 2

    xs = torch.empty((B, 4, C, H * W), dtype=desc0.dtype, device=desc0.device)

    desc_cat = torch.cat([desc0, desc1], 3)  # desc_2w
    xs[:, 0] = desc_cat[:, :, 0::step_size, 0::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 2] = desc_cat[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [h/2, 2w/2]

    if sym_flag:
        desc1 = torch.rot90(desc1, k=2, dims=(2, 3))

    desc_cat = torch.cat([desc0, desc1], 2)  # desc_2h
    xs[:, 1] = desc_cat.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 0::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]
    xs[:, 3] = desc_cat.transpose(dim0=2, dim1=3)[:, :, 0::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_crossVM(ys, ori_h: int, ori_w: int, step_size=2, sym_flag=False):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2 * new_w), device=ys.device,
                       dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2w[:, :, 0::step_size, 0::step_size] = ys[:, 0].reshape(B, C, H, 2 * W)
    y_2w[:, :, 1::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, H, 2 * W)

    y_2h = torch.zeros((B, C, 2 * new_h, new_w), device=ys.device,
                        dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))
    y_2h[:, :, 0::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2 * H).transpose(dim0=2, dim1=3)
    y_2h[:, :, 1::step_size, 0::step_size] = ys[:, 3].flip([2]).reshape(B, C, W, 2 * H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()

    desc0_w, desc1_w = torch.chunk(y_2w, 2, dim=3)
    desc0_h, desc1_h = torch.chunk(y_2h, 2, dim=2)
    if sym_flag:
        desc1_h = torch.rot90(desc1_h, k=2, dims=(2, 3))

    return desc0_w + desc0_h, desc1_w + desc1_h


# —————————————————————————————————————————————————— CrossVM Encoder ——————————————————————————————————————————————————
class CrossVM(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 depth=4,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 profiler=None,
                 mamba_type="v1",
                 model_type="full",
                 ):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(create_block(
                    feature_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    mamba_type=mamba_type,
                ))
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.jointing = GLU_3(feature_dim, feature_dim)
        self.sym_flag = True if model_type == "full" else False

    def forward(self, data):
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        desc0, desc1 = (desc0.view(data['bs'], -1, data['h_8'], data['w_8']),
                        desc1.view(data['bs'], -1, data['h_8'], data['w_8']))
        x, ori_h, ori_w = scan_crossVM(desc0, desc1, 2, sym_flag=self.sym_flag)
        y0, y1, y2, y3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        for i in range(len(self.layers) // 4):
            y0 = self.layers[i * 4 + 0](y0)
            y1 = self.layers[i * 4 + 1](y1)
            y2 = self.layers[i * 4 + 2](y2)
            y3 = self.layers[i * 4 + 3](y3)
        y = torch.stack([y0, y1, y2, y3], 1).transpose(2, 3).to(dtype=x.dtype)
        desc0, desc1 = merge_crossVM(y, ori_h, ori_w, 2, sym_flag=self.sym_flag)
        desc = self.jointing(torch.cat([desc0, desc1], 0))
        desc0, desc1 = torch.chunk(desc, 2, dim=0)
        if self.sym_flag:
            desc0, desc1 = desc0.flatten(2, 3), desc1.flatten(2, 3)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })


# ——————————————————————————————————————————————————— CrossVM Light ———————————————————————————————————————————————————
class AG_CrossVM(nn.Module):
    def __init__(self,
                 d_model,
                 agg_size=2,
                 depth=4,
                 mamba_type='v1',
                 rms_norm=True,
                 residual_in_fp32=True,
                 fused_add_norm=True,
                 profiler=None,
                 ):
        super(AG_CrossVM, self).__init__()
        self.profiler = profiler or PassThroughProfiler()

        # aggregation
        self.agg_size = agg_size
        self.aggregate0 = nn.Conv2d(d_model, d_model,
                                    kernel_size=agg_size,
                                    padding=0,
                                    stride=agg_size,
                                    bias=False,
                                    groups=d_model) if agg_size != 1 else nn.Identity()
        self.aggregate1 = nn.Conv2d(d_model, d_model,
                                    kernel_size=agg_size,
                                    padding=0,
                                    stride=agg_size,
                                    bias=False,
                                    groups=d_model) if agg_size != 1 else nn.Identity()

        # sampling smoothing
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.crossVM = CrossVM(d_model, depth=depth, model_type="light", mamba_type=mamba_type, rms_norm=rms_norm,
                               residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, profiler=profiler)

    def forward(self, data):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """

        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        desc0, desc1 = (desc0.view(data['bs'], -1, data['h_8'], data['w_8']),
                        desc1.view(data['bs'], -1, data['h_8'], data['w_8']))
        # 要双聚合，或者说两个操作一样，因为不区分主次
        desc0_ag = self.norm1(self.aggregate0(desc0).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        desc1_ag = self.norm1(self.aggregate1(desc1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        # desc1_ag = self.norm1(self._max_pool(desc1))
        data.update({
            'feat_8_0': desc0_ag,
            'feat_8_1': desc1_ag,
            'h_8': data['h_8'] // self.agg_size,
            'w_8': data['w_8'] // self.agg_size,
        })

        self.crossVM(data)
        desc0_ag, desc1_ag = data['feat_8_0'], data['feat_8_1']

        # Upsample feature
        if self.agg_size != 1:
            desc0_ag = torch.nn.functional.interpolate(desc0_ag, scale_factor=self.agg_size,
                                                       mode='bilinear', align_corners=False)  # [N, C, H0, W0]
            desc1_ag = torch.nn.functional.interpolate(desc1_ag, scale_factor=self.agg_size,
                                                       mode='bilinear', align_corners=False)  # [N, C, H0, W0]
            desc0_ag = self.smooth_conv(desc0_ag)
            desc1_ag = self.smooth_conv(desc1_ag)

        # feed-forward network
        desc0_ag = self.mlp(torch.cat([desc0, desc0_ag], dim=1).permute(0, 2, 3, 1))  # [N, H0, W0, C]
        desc1_ag = self.mlp(torch.cat([desc1, desc1_ag], dim=1).permute(0, 2, 3, 1))  # [N, H0, W0, C]
        desc0_ag = self.norm2(desc0_ag).permute(0, 3, 1, 2).contiguous()  # [N, C, H0, W0]
        desc1_ag = self.norm2(desc1_ag).permute(0, 3, 1, 2).contiguous()  # [N, C, H0, W0]

        desc0, desc1 = desc0 + desc0_ag, desc1 + desc1_ag
        desc0, desc1 = desc0.flatten(2, 3), desc1.flatten(2, 3)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
            'h_8': data['h_8'] * self.agg_size,
            'w_8': data['w_8'] * self.agg_size,
        })
