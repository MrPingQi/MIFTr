import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
INF = 1e9


########################################  ↓  Coarse-Level matching  ↓  ########################################
class CoarseMatching(nn.Module):
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        # general config
        d_model = 256
        self.thr = config['thr']
        self.use_sm = config['use_sm']
        self.inference = config['inference']
        self.border_rm = config['border_rm']

        self.final_proj = nn.Linear(d_model, d_model, bias=True)

        self.temperature = config['dsmax_temperature']
        self.profiler = profiler

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                  feat_c1) / self.temperature

        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                -INF)

        if self.inference:
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
            data.update({'conf_matrix_0_to_1': conf_matrix_0_to_1,
                         'conf_matrix_1_to_0': conf_matrix_1_to_0})
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_training(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 2) if self.use_sm else sim_matrix

        # confidence thresholding and nearest neighbour for 0 to 1
        mask = (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=2, keepdim=True)[0])

        # unlike training, reuse the same conf martix to decrease the vram consumption
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 1) if self.use_sm else sim_matrix

        # update mask {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(mask,
                                (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = sim_matrix[b_ids, i_ids, j_ids]

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current coarse level predictions
        coarse_matches.update({
            'mconf': mconf,
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_training(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix_0_to_1.device

        # confidence thresholding
        # {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > self.thr) * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > self.thr) * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids])

        # random sampling of training samples for fine-level XoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # the sampling is performed across all pairs in a batch without manually balancing
            # samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.config['train_coarse_percent'])
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config['train_pad_num_gt_min']
            assert train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred,
                     train_pad_num_gt_min),),
                device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mkpts0_c_train': mkpts0_c,
            'mkpts1_c_train': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


########################################  ↓  Fine/Sub-Level matching  ↓  ########################################
class FineSubMatching(nn.Module):
    """Fine-level and Sub-pixel matching"""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.inference = config['fine']['inference']
        self.fine_thr = config['fine']['thr']
        dim_f = 64
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Linear(2 * dim_f, 4, bias=False))
        self.fine_spv_max = None  # saving memory
        self.profiler = profiler

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        """
        Args:
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'mconf_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0_unfold.shape

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            device = feat_f0_unfold.device
            W_f2 = self.W_f * self.W_f
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=device),
                'mkpts0_f_train': data['mkpts0_c_train'],
                'mkpts1_f_train': data['mkpts1_c_train'],
                'conf_matrix_fine': torch.zeros(1, W_f2, W_f2, device=device),
                'b_ids_fine': torch.zeros(0, device=device),
                'i_ids_fine': torch.zeros(0, device=device),
                'j_ids_fine': torch.zeros(0, device=device),
            })
            return

        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_f0, feat_f1])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0, feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        with torch.no_grad():
            W_f = self.W_f

            # 1. confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            # 2. match only the highest confidence
            mask = mask * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

            # 3. find all valid fine matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # 4. update with matches in original image resolution

            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2)
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1],
                 torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c[b_ids]

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # fine-level discrete matches on window coordiantes
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')], dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')], dim=1)

        # sub-pixel refinement
        sub_ref = self.subpixel_mlp(torch.cat([feat_f0_unfold[b_ids, i_ids],
                                               feat_f1_unfold[b_ids, j_ids]], dim=-1))
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        sub_ref0, sub_ref1 = sub_ref0.squeeze(1), sub_ref1.squeeze(1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad) * scale0  # + sub_ref0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad) * scale1  # + sub_ref1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0  # + sub_ref0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1  # + sub_ref1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches are the current prediction (for visualization)
        mask = mconf != 0  # mconf == 0 => gt matches
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],
            'mkpts0_f1': mkpts0_f1[mconf != 0],
            'mkpts1_f1': mkpts1_f1[mconf != 0],
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }
        # sub_pixel_matches = {
        #     'm_bids': b_ids_c[b_ids[mask]],
        #     'mkpts0_f1': mkpts0_f1[mask],
        #     'mkpts1_f1': mkpts1_f1[mask],
        #     'mkpts0_f': mkpts0_f[mask],
        #     'mkpts1_f': mkpts1_f[mask],
        #     'mconf_f': mconf[mask]
        # }

        # These matches are used for training
        if not self.inference:
            sub_pixel_matches.update({
                'mkpts0_f_train': mkpts0_f_train[mconf != 0],
                'mkpts1_f_train': mkpts1_f_train[mconf != 0],
            })
            if self.fine_spv_max is None or self.fine_spv_max > len(data['b_ids']):
                sub_pixel_matches.update({
                    'b_ids_fine': data['b_ids'],
                    'i_ids_fine': data['i_ids'],
                    'j_ids_fine': data['j_ids'],
                    'conf_matrix_fine': conf_matrix_fine
                })
            else:
                train_mask = generate_random_mask(len(data['b_ids']), self.fine_spv_max)
                sub_pixel_matches.update({
                    'b_ids_fine': data['b_ids'][train_mask],
                    'i_ids_fine': data['i_ids'][train_mask],
                    'j_ids_fine': data['j_ids'][train_mask],
                    'conf_matrix_fine': conf_matrix_fine[train_mask]
                })

        return sub_pixel_matches


def generate_random_mask(n, num_true):
    # 创建全 False 的掩码
    mask = torch.zeros(n, dtype=torch.bool)

    # 随机选择 num_true 个位置并设置为 True
    indices = torch.randperm(n)[:num_true]
    mask[indices] = True

    return mask
