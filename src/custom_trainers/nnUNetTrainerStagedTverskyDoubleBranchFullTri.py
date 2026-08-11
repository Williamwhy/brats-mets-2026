"""
nnUNetTrainerStagedTverskyDoubleBranch
======================================
Multi-mode encoder architecture for nnU-Net with integrated features:
- Clinical Prior attention map
- Modality dropout (single: per-channel; dual/triple: max 1 per group)
- Small-lesion oversampling with CC-Dice boost mask and MIL bags
- Configurable RC activation threshold
- Optional attention-based fusion

Encoder modes:
  SINGLE_ENCODER: Standard 4-channel ResidualEncoderUNet
  DUAL_ENCODER:   Two encoders (T1/T1ce + T2/FLAIR) + fusion
  TRIPLE_ENCODER: Three encoders (+ spectral features) + fusion
"""

import importlib
import inspect
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

# --------------------------------------------------------------------------- #
# Import base trainer AND dataloader
# --------------------------------------------------------------------------- #
try:
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerStagedTversky import (
        nnUNetTrainerStagedTversky,
        nnUNetTrainerStagedTversky_Stage1Calibration,
    )
except ImportError:
    from nnUNetTrainerStagedTversky import (
        nnUNetTrainerStagedTversky,
        nnUNetTrainerStagedTversky_Stage1Calibration,
    )

# CRITICAL IMPORT: nnUNetDataLoader base class for custom dataloader
try:
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
except ImportError:
    # Fallback if import path differs in your nnU-Net install
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataLoader


# =============================================================================
# CUSTOM DATALOADER WITH SMALL-LESION OVERSAMPLING
# =============================================================================

class nnUNetDataLoaderWithBBox(nnUNetDataLoader):
    """
    Extends nnUNetDataLoader with:
    - bbox_lbs exposed in every batch dict
    - Small-lesion oversampling: SMALL_OVERSAMPLE_RATE of patches centred
      on precomputed small-lesion voxel coords
    - Lesion-wise continuous inverse-size weight maps
    - MIL bag generation for small lesions
    """
    SMALL_OVERSAMPLE_RATE = 0.50
    DEBUG_PRINT = False

    def __init__(self, data, batch_size, patch_size, final_patch_size, label_manager, *args,
                 small_lesion_locs=None, build_small_mask=False, small_lesion_voxel_threshold=50,
                 cc_connectivity=26, region_definitions=None, region_order=None, **kwargs):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, *args, **kwargs)
        self._label_manager = label_manager
        self._small_lesion_locs = small_lesion_locs if small_lesion_locs is not None else {}
        self._build_small_mask = build_small_mask
        self._small_lesion_voxel_threshold = small_lesion_voxel_threshold
        self._cc_structure = np.ones((3, 3, 3)) if cc_connectivity == 26 else None
        self._region_definitions = region_definitions or {}
        self._region_order = region_order or []

    def generate_train_batch(self):
        import random as _rnd
        from threadpoolctl import threadpool_limits
        from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
        from scipy.ndimage import label as _cc_label

        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None
        all_bbox_lbs = []
        small_mask_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]

                    # Small-lesion oversampling
                    class_locs = properties['class_locations']
                    sl_locs = self._small_lesion_locs.get(i, {})
                    took_small_branch = bool(sl_locs) and _rnd.random() < self.SMALL_OVERSAMPLE_RATE
                    if took_small_branch:
                        bbox_lbs, bbox_ubs = self.get_bbox(shape, True, sl_locs)
                    else:
                        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locs)

                    all_bbox_lbs.append(list(bbox_lbs))
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1)).to(torch.int16)

                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        t = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = t['image']
                        seg_sample = t['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    data_all[j] = data_sample
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                       for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                        full_res_seg = seg_sample[0]
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape),
                                                  dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
                        full_res_seg = seg_sample

                    # Lesion-wise continuous inverse-size weight & MIL bag generation
                    if self._build_small_mask:
                        n_regions = len(self._region_order)
                        if small_mask_all is None:
                            small_mask_all = torch.ones(
                                (self.batch_size, n_regions, *full_res_seg.shape[1:]),
                                dtype=torch.float32)
                        if 'mil_bags_all' not in locals():
                            mil_bags_all = torch.zeros(
                                (self.batch_size, n_regions, *full_res_seg.shape[1:]),
                                dtype=torch.float32)
                        
                        seg_np = full_res_seg[0].numpy()
                        for region in self._region_order:
                            region_idx = self._region_order.index(region)
                            region_mask = np.isin(seg_np, self._region_definitions[region])
                            if not region_mask.any():
                                continue
                            
                            lbl, n = _cc_label(region_mask, structure=self._cc_structure)
                            if n == 0:
                                continue
                            
                            counts = np.bincount(lbl.ravel())
                            small_bag_counter = 0
                            
                            for cid in range(1, n + 1):
                                cnt = counts[cid]
                                inverse_weight = 1.0 / float(cnt)
                                small_mask_all[j, region_idx][lbl == cid] = inverse_weight
                                
                                if cnt < self._small_lesion_voxel_threshold:
                                    small_bag_counter += 1
                                    mil_bags_all[j, region_idx][lbl == cid] = small_bag_counter
                                
                        for r_idx in range(n_regions):
                            fg_mask = small_mask_all[j, r_idx] != 1.0
                            if fg_mask.any():
                                mean_val = small_mask_all[j, r_idx][fg_mask].mean()
                                small_mask_all[j, r_idx][fg_mask] /= (mean_val + 1e-5)

        batch = {'data': data_all, 'target': seg_all,
                 'keys': selected_keys, 'bbox_lbs': all_bbox_lbs}
        if small_mask_all is not None:
            batch['small_mask'] = small_mask_all
        if 'mil_bags_all' in locals():
            batch['mil_bags'] = mil_bags_all
        return batch


# =============================================================================
# FUSION MODULES
# =============================================================================

class Conv1x1FusionModule(nn.Module):
    """Lightweight 1x1 conv fusion — DEFAULT."""
    def __init__(self, channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__()
        self.fuse = conv_op(2 * channels, channels, kernel_size=1, bias=conv_bias)
        self.norm = norm_op(channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class Conv1x1Fusion3WayModule(nn.Module):
    """1x1 conv fusion for 3 branches."""
    def __init__(self, channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__()
        self.fuse = conv_op(3 * channels, channels, kernel_size=1, bias=conv_bias)
        self.norm = norm_op(channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, s1, s2, s3):
        x = torch.cat([s1, s2, s3], dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class AttentionFusionModule(nn.Module):
    """Per-voxel attention fusion for 2 branches."""
    def __init__(self, channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__()
        mid = max(channels // 4, 8)
        self.conv1 = conv_op(2 * channels, mid, kernel_size=3, padding=1, bias=conv_bias)
        self.norm1 = norm_op(mid, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(mid, 2, kernel_size=3, padding=1, bias=True)
        
    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        logits = self.conv2(x)
        weights = F.softmax(logits, dim=1)
        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        return w1 * s1 + w2 * s2


class AttentionFusion3WayModule(nn.Module):
    """Per-voxel attention fusion for 3 branches."""
    def __init__(self, channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__()
        mid = max(channels // 4, 8)
        self.conv1 = conv_op(3 * channels, mid, kernel_size=3, padding=1, bias=conv_bias)
        self.norm1 = norm_op(mid, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(mid, 3, kernel_size=3, padding=1, bias=True)
        
    def forward(self, s1, s2, s3):
        x = torch.cat([s1, s2, s3], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        logits = self.conv2(x)
        weights = F.softmax(logits, dim=1)
        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]
        return w1 * s1 + w2 * s2 + w3 * s3


# =============================================================================
# SPECTRAL FEATURE EXTRACTOR
# =============================================================================

class SpectralFeatureExtractor(nn.Module):
    """
    Extracts spectral/frequency-domain features from T1ce and FLAIR.
    Produces 2 channels: local high-frequency energy and low-frequency dominance.
    """
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.conv_low = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv_mid = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv_high = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, groups=in_channels)
        self.mix = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        low = self.conv_low(x)
        mid = self.conv_mid(x)
        high = self.conv_high(x)
        fused = torch.cat([low, mid, high], dim=1)
        out = self.mix(fused)
        out = self.bn(out)
        out = self.relu(out)
        return out


# =============================================================================
# MULTI-MODE NETWORK ARCHITECTURE
# =============================================================================

class MultiModeResidualEncoderUNet(nn.Module):
    """
    Three modes:
      SINGLE_ENCODER: Standard 4-channel ResidualEncoderUNet
      DUAL_ENCODER:   Two encoders (T1/T1ce + T2/FLAIR) + fusion
      TRIPLE_ENCODER: Three encoders (+ spectral features) + fusion
    """
    def __init__(self, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides, n_blocks_per_stage, num_classes,
                 n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                 deep_supervision=True, nonlin_first=False,
                 encoder_mode="dual",
                 use_attention_fusion=False,
                 rc_threshold=0.5,
                 rc_channel_index=3,
                 **extra_kwargs):
        super().__init__()

        self.encoder_mode = encoder_mode.lower()
        self.use_attention_fusion = use_attention_fusion
        self.rc_threshold = rc_threshold
        self.rc_channel_index = rc_channel_index

        all_kwargs = dict(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )
        all_kwargs.update(extra_kwargs)

        sig = inspect.signature(ResidualEncoderUNet.__init__)
        valid_keys = set(sig.parameters.keys()) - {'self'}
        kwargs = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        if self.encoder_mode == "single":
            kwargs_single = dict(kwargs)
            kwargs_single['input_channels'] = 4
            self.network = ResidualEncoderUNet(**kwargs_single)
            self.encoder = self.network.encoder
            self.decoder = self.network.decoder
            self.fusion_modules = nn.ModuleList()
            
        elif self.encoder_mode == "dual":
            kwargs_dual = dict(kwargs)
            kwargs_dual['input_channels'] = 2
            ref_net_t1 = ResidualEncoderUNet(**kwargs_dual)
            ref_net_t2 = ResidualEncoderUNet(**kwargs_dual)
            self.encoder_t1 = ref_net_t1.encoder
            self.encoder_t2 = ref_net_t2.encoder
            self.decoder = ref_net_t1.decoder
            fusion_cls = AttentionFusionModule if use_attention_fusion else Conv1x1FusionModule
            self.fusion_modules = nn.ModuleList([
                fusion_cls(f, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
                for f in features_per_stage
            ])
            
        elif self.encoder_mode == "triple":
            kwargs_dual = dict(kwargs)
            kwargs_dual['input_channels'] = 2
            ref_net_t1 = ResidualEncoderUNet(**kwargs_dual)
            ref_net_t2 = ResidualEncoderUNet(**kwargs_dual)
            ref_net_spectral = ResidualEncoderUNet(**kwargs_dual)
            self.encoder_t1 = ref_net_t1.encoder
            self.encoder_t2 = ref_net_t2.encoder
            self.encoder_spectral = ref_net_spectral.encoder
            self.decoder = ref_net_t1.decoder
            self.spectral_extractor = SpectralFeatureExtractor(in_channels=2, out_channels=2)
            fusion_cls = AttentionFusion3WayModule if use_attention_fusion else Conv1x1Fusion3WayModule
            self.fusion_modules = nn.ModuleList([
                fusion_cls(f, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
                for f in features_per_stage
            ])
        else:
            raise ValueError(f"encoder_mode must be 'single', 'dual', or 'triple', got '{encoder_mode}'")

    def forward(self, x):
        if self.encoder_mode == "single":
            out = self.network(x)
        elif self.encoder_mode == "dual":
            x_t1 = x[:, [0, 1]]
            x_t2 = x[:, [2, 3]]
            skips_t1 = self.encoder_t1(x_t1)
            skips_t2 = self.encoder_t2(x_t2)
            fused_skips = []
            for i, (s1, s2) in enumerate(zip(skips_t1, skips_t2)):
                fused = self.fusion_modules[i](s1, s2)
                fused_skips.append(fused)
            out = self.decoder(fused_skips)
        elif self.encoder_mode == "triple":
            x_t1 = x[:, [0, 1]]
            x_t2 = x[:, [2, 3]]
            x_spectral_input = x[:, [1, 3]]
            x_spectral = self.spectral_extractor(x_spectral_input)
            skips_t1 = self.encoder_t1(x_t1)
            skips_t2 = self.encoder_t2(x_t2)
            skips_spectral = self.encoder_spectral(x_spectral)
            fused_skips = []
            for i, (s1, s2, s3) in enumerate(zip(skips_t1, skips_t2, skips_spectral)):
                fused = self.fusion_modules[i](s1, s2, s3)
                fused_skips.append(fused)
            out = self.decoder(fused_skips)
        
        if isinstance(out, (list, tuple)) and not self.training:
            out = out[0]
        
        if not self.training and self.encoder_mode != "single":
            out = self._apply_rc_threshold(out)
        
        return out

    def _apply_rc_threshold(self, out):
        o = out.clone()
        prob = torch.sigmoid(o)
        rc_prob = prob[:, self.rc_channel_index:self.rc_channel_index+1]
        rc_thresholded = torch.where(
            rc_prob >= self.rc_threshold,
            torch.full_like(rc_prob, 5.0),
            torch.full_like(rc_prob, -5.0)
        )
        o[:, self.rc_channel_index:self.rc_channel_index+1] = rc_thresholded
        return o


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class nnUNetTrainerStagedTverskyDoubleBranchFullTri(nnUNetTrainerStagedTversky):
    """
    Multi-mode trainer with:
    - Clinical prior attention
    - Modality dropout (single: per-channel; dual/triple: max 1 per group)
    - Small-lesion oversampling with CC-Dice boost and MIL bags
    - Configurable RC threshold
    - Optional attention fusion
    """
    # --- Encoder / Fusion Config ---
    ENCODER_MODE: str = "triple"  # Options: "single", "dual", "triple"
    USE_ATTENTION_FUSION: bool = False
    RC_THRESHOLD: float = 0.5
    
    # --- Clinical Prior Config ---
    ENABLE_CLINICAL_PRIOR: bool = True
    PRIOR_T1C_CHANNEL: int = 1
    
    # --- Modality Dropout Config ---
    ENABLE_MODALITY_DROPOUT: bool = True
    P_PER_CHANNEL: float = 0.25
    P_PER_ENCODER_GROUP: float = 0.25
    
    # --- Small Lesion Config ---
    ENABLE_SMALL_LESION_OVERSAMPLING: bool = True
    SMALL_LESION_VOXEL_THRESHOLD: int = 29
    CC_CONNECTIVITY: int = 26
    SMALL_OVERSAMPLE_RATE: float = 0.50
    SMALL_BOOST_FACTOR: float = 10.0
    
    # --- Region Config ---
    REGION_ORDER = ['WT', 'TC', 'ET', 'RC']
    REGION_DEFINITIONS = {
        'WT': (1, 2, 3),
        'TC': (1, 3),
        'ET': (3,),
        'RC': (4,),
    }
    REGION_PAINT_LABEL = {'WT': 2, 'TC': 1, 'ET': 3, 'RC': 4}
    N_RAW_FG_CLASSES = 4
    REGION_LOSS_WEIGHTS = {'WT': 1.0, 'TC': 1.0, 'ET': 1.5, 'RC': 3.0}

    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._val_small_masks = {}
        self._val_large_masks = {}
        self._val_small_locs = {}
        self._train_small_locs = {}
        self._small_f1_acc = []
        self._large_f1_acc = []
        self._precomputed = False
        self._current_small_mask = None
        self._current_mil_bags = None

    # -------------------------------------------------------------------------
    # CLINICAL PRIOR
    # -------------------------------------------------------------------------
    
    def generate_clinical_prior_map(self, data_tensor):
        B, C, X, Y, Z = data_tensor.shape
        device = data_tensor.device
        gx = torch.linspace(0, 1, X, device=device).view(1, X, 1, 1).expand(B, X, Y, Z)
        gy = torch.linspace(0, 1, Y, device=device).view(1, 1, Y, 1).expand(B, X, Y, Z)
        gz = torch.linspace(0, 1, Z, device=device).view(1, 1, 1, Z).expand(B, X, Y, Z)
        t1c = data_tensor[:, self.PRIOR_T1C_CHANNEL]
        grad_x = torch.abs(t1c[:, 1:, :, :] - t1c[:, :-1, :, :])
        grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
        w_gwmi = grad_x / (torch.max(grad_x) + 1e-5)
        sv = 0.04
        w_vasc = torch.max(torch.exp(-((gx - 0.33) ** 2) / (2 * sv ** 2)),
                           torch.exp(-((gx - 0.66) ** 2) / (2 * sv ** 2)))
        w_rol = torch.exp(-((gx - 0.5) ** 2 / (2 * 0.15 ** 2)
                            + (gy - 0.5) ** 2 / (2 * 0.15 ** 2)
                            + (gz - 0.85) ** 2 / (2 * 0.10 ** 2)))
        prior = w_gwmi + 0.5 * w_vasc + 0.5 * w_rol
        return (prior / (torch.max(prior) + 1e-5)).unsqueeze(1)

    # -------------------------------------------------------------------------
    # MODALITY DROPOUT
    # -------------------------------------------------------------------------
    
    @staticmethod
    def apply_modality_dropout_single(data, p_per_channel=0.25):
        B, C = data.shape[:2]
        device = data.device
        keep = torch.rand(B, C, device=device) >= p_per_channel
        all_dropped = ~keep.any(dim=1, keepdim=True)
        rescue_idx = torch.randint(C, (B, 1), device=device)
        rescue_mask = torch.zeros(B, C, dtype=torch.bool, device=device)
        rescue_mask.scatter_(1, rescue_idx, True)
        keep = keep | (all_dropped & rescue_mask)
        return data * keep.float().view(B, C, 1, 1, 1)

    @staticmethod
    def apply_modality_dropout_dual(data, p_per_group=0.25):
        B, C = data.shape[:2]
        device = data.device
        out = data.clone()
        
        keep_g0 = torch.rand(B, 2, device=device) >= p_per_group
        all_dropped_g0 = ~keep_g0.any(dim=1)
        rescue_g0 = torch.randint(2, (B,), device=device)
        keep_g0[all_dropped_g0, rescue_g0[all_dropped_g0]] = True
        out[:, 0] *= keep_g0[:, 0].float().view(B, 1, 1, 1)
        out[:, 1] *= keep_g0[:, 1].float().view(B, 1, 1, 1)
        
        keep_g1 = torch.rand(B, 2, device=device) >= p_per_group
        all_dropped_g1 = ~keep_g1.any(dim=1)
        rescue_g1 = torch.randint(2, (B,), device=device)
        keep_g1[all_dropped_g1, rescue_g1[all_dropped_g1]] = True
        out[:, 2] *= keep_g1[:, 0].float().view(B, 1, 1, 1)
        out[:, 3] *= keep_g1[:, 1].float().view(B, 1, 1, 1)
        
        return out

    def apply_modality_dropout(self, data):
        if not self.ENABLE_MODALITY_DROPOUT:
            return data
        if self.ENCODER_MODE == "single":
            return self.apply_modality_dropout_single(data, self.P_PER_CHANNEL)
        elif self.ENCODER_MODE in ("dual", "triple"):
            return self.apply_modality_dropout_dual(data, self.P_PER_ENCODER_GROUP)
        return data

    # -------------------------------------------------------------------------
    # SMALL-LESION PRECOMPUTATION
    # -------------------------------------------------------------------------
    
    def initialize(self):
        super().initialize()
        if getattr(self, '_precomputed', False):
            self.print_to_log_file("[SmallLesion] Preserving precomputed small-lesion dicts.")
            self._current_small_mask = None
            return
        self._val_small_masks = {}
        self._val_large_masks = {}
        self._val_small_locs = {}
        self._train_small_locs = {}
        self._small_f1_acc = []
        self._large_f1_acc = []
        self._precomputed = False
        self._current_small_mask = None
        self._current_mil_bags = None

    def on_train_start(self):
        if not self._precomputed and self.ENABLE_SMALL_LESION_OVERSAMPLING:
            self._precompute_train_small_lesion_locs()
            self._precompute_val_small_lesion_masks()
            self._precomputed = True
        super().on_train_start()

    def _precompute_train_small_lesion_locs(self):
        import cc3d
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.utilities.file_and_folder_operations import load_json

        self.print_to_log_file("[SmallLesion] Precomputing small-lesion locs for TRAINING...")
        t0 = time.time()

        splits_file = os.path.join(
            os.path.dirname(self.preprocessed_dataset_folder), 'splits_final.json')
        train_keys = load_json(splits_file)[self.fold]['train']
        ds_class = infer_dataset_class(self.preprocessed_dataset_folder)
        dataset = ds_class(self.preprocessed_dataset_folder, identifiers=train_keys)

        n_with = 0
        for key in train_keys:
            _, seg, _, _ = dataset.load_case(key)
            seg_np = np.array(seg[0]).astype(np.int32)
            locs = {}
            for region in self.REGION_ORDER:
                mask = np.isin(seg_np, self.REGION_DEFINITIONS[region]).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                cc_map, n = cc3d.connected_components(mask, connectivity=self.CC_CONNECTIVITY, return_N=True)
                if n == 0:
                    continue
                stats = cc3d.statistics(cc_map)
                coords = [np.argwhere(cc_map == cid)
                          for cid in range(1, n + 1)
                          if stats['voxel_counts'][cid] < self.SMALL_LESION_VOXEL_THRESHOLD]
                if coords:
                    coords_arr = np.concatenate(coords, axis=0)
                    locs[region] = np.concatenate(
                        [np.zeros((coords_arr.shape[0], 1), dtype=coords_arr.dtype), coords_arr], axis=1)
            if locs:
                self._train_small_locs[key] = locs
                n_with += 1

        self.print_to_log_file(
            f"[SmallLesion] Training: {n_with}/{len(train_keys)} cases have small lesions "
            f"— done in {time.time()-t0:.1f}s"
        )

    def _precompute_val_small_lesion_masks(self):
        import cc3d
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.utilities.file_and_folder_operations import load_json

        self.print_to_log_file("[SmallLesion] Precomputing small-lesion masks for VAL...")
        t0 = time.time()

        splits_file = os.path.join(
            os.path.dirname(self.preprocessed_dataset_folder), 'splits_final.json')
        val_keys = load_json(splits_file)[self.fold]['val']
        ds_class = infer_dataset_class(self.preprocessed_dataset_folder)
        dataset = ds_class(self.preprocessed_dataset_folder, identifiers=val_keys)

        for key in val_keys:
            _, seg, _, _ = dataset.load_case(key)
            seg_np = np.array(seg[0]).astype(np.int32)
            masks = {}
            large_masks = {}
            locs = {}
            for region in self.REGION_ORDER:
                region_mask = np.isin(seg_np, self.REGION_DEFINITIONS[region]).astype(np.uint8)
                if region_mask.sum() == 0:
                    masks[region] = np.zeros(seg_np.shape, dtype=bool)
                    large_masks[region] = np.zeros(seg_np.shape, dtype=bool)
                    continue
                cc_map, n = cc3d.connected_components(region_mask, connectivity=self.CC_CONNECTIVITY, return_N=True)
                small = np.zeros(seg_np.shape, dtype=bool)
                if n > 0:
                    stats = cc3d.statistics(cc_map)
                    coords = []
                    for cid in range(1, n + 1):
                        if stats['voxel_counts'][cid] < self.SMALL_LESION_VOXEL_THRESHOLD:
                            small[cc_map == cid] = True
                            coords.append(np.argwhere(cc_map == cid))
                    if coords:
                        coords_arr = np.concatenate(coords, axis=0)
                        locs[region] = np.concatenate(
                            [np.zeros((coords_arr.shape[0], 1), dtype=coords_arr.dtype), coords_arr], axis=1)
                masks[region] = small
                large_masks[region] = region_mask.astype(bool) & ~small
            self._val_small_masks[key] = masks
            self._val_large_masks[key] = large_masks
            if locs:
                self._val_small_locs[key] = locs

        self.print_to_log_file(f"[SmallLesion] Val: {len(val_keys)} cases in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # DATALOADERS
    # -------------------------------------------------------------------------
    
    def get_dataloaders(self):
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _base_mod

        if self.dataset_class is None:
            self.dataset_class = _base_mod.infer_dataset_class(self.preprocessed_dataset_folder)

        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes) = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            self.configuration_manager.patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if self.ENABLE_SMALL_LESION_OVERSAMPLING:
            dl_tr = nnUNetDataLoaderWithBBox(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
                small_lesion_locs=self._train_small_locs, build_small_mask=True,
                small_lesion_voxel_threshold=self.SMALL_LESION_VOXEL_THRESHOLD,
                cc_connectivity=self.CC_CONNECTIVITY,
                region_definitions=self.REGION_DEFINITIONS,
                region_order=self.REGION_ORDER,
            )
            dl_val = nnUNetDataLoaderWithBBox(
                dataset_val, self.batch_size, self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
                small_lesion_locs=self._val_small_locs,
                small_lesion_voxel_threshold=self.SMALL_LESION_VOXEL_THRESHOLD,
                cc_connectivity=self.CC_CONNECTIVITY,
                region_definitions=self.REGION_DEFINITIONS,
                region_order=self.REGION_ORDER,
            )
        else:
            dl_tr = nnUNetDataLoader(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataLoader(
                dataset_val, self.batch_size, self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = _base_mod.get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = _base_mod.SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = _base_mod.SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = _base_mod.NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = _base_mod.NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # -------------------------------------------------------------------------
    # BATCH HOOKS
    # -------------------------------------------------------------------------
    
    def on_train_batch_start(self, batch):
        batch = super().on_train_batch_start(batch)
        batch['data'] = batch['data'].to(self.device, non_blocking=True)
        
        if self.ENABLE_CLINICAL_PRIOR:
            prior = self.generate_clinical_prior_map(batch['data'])
            batch['data'] = batch['data'] * (1.0 + prior)
        
        batch['data'] = self.apply_modality_dropout(batch['data'])
        
        if 'small_mask' in batch:
            self._current_small_mask = batch['small_mask'].to(self.device, non_blocking=True)
        else:
            self._current_small_mask = None
            
        if 'mil_bags' in batch:
            self._current_mil_bags = batch['mil_bags'].to(self.device, non_blocking=True)
        else:
            self._current_mil_bags = None
            
        return batch

    def on_val_batch_start(self, batch):
        batch = super().on_val_batch_start(batch)
        batch['data'] = batch['data'].to(self.device, non_blocking=True)
        
        if self.ENABLE_CLINICAL_PRIOR:
            prior = self.generate_clinical_prior_map(batch['data'])
            batch['data'] = batch['data'] * (1.0 + prior)
        
        self._current_small_mask = None
        self._current_mil_bags = None
        return batch

    # -------------------------------------------------------------------------
    # VALIDATION STEP (with small/large lesion F1)
    # -------------------------------------------------------------------------
    
    def validation_step(self, batch):
        batch = self.on_val_batch_start(batch)
        data = batch['data']
        target = batch['target']

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                if isinstance(target, (list, tuple)):
                    target_device = [t.to(self.device, non_blocking=True) for t in target]
                else:
                    target_device = target.to(self.device, non_blocking=True)
                loss = self.loss(output, target_device)
            full_res = output[0] if isinstance(output, (list, tuple)) else output

        target_full_res = target_device[0] if isinstance(target_device, (list, tuple)) else target_device

        prob = torch.sigmoid(full_res)
        pred_regions = prob > 0.5
        target_regions = target_full_res.bool()

        n_regions = pred_regions.shape[1]
        tp_h = np.zeros(n_regions, dtype=np.float32)
        fp_h = np.zeros(n_regions, dtype=np.float32)
        fn_h = np.zeros(n_regions, dtype=np.float32)
        for i in range(n_regions):
            pred_c = pred_regions[:, i]
            target_c = target_regions[:, i]
            tp_h[i] = (pred_c & target_c).sum().item()
            fp_h[i] = (pred_c & ~target_c).sum().item()
            fn_h[i] = (~pred_c & target_c).sum().item()

        bbox_lbs = batch.get('bbox_lbs', [None] * pred_regions.shape[0])
        for b in range(pred_regions.shape[0]):
            key = batch['keys'][b]
            lbs = bbox_lbs[b] if bbox_lbs[b] is not None else [0, 0, 0]
            self._small_f1_acc.append(
                self._compute_f1_from_masks(pred_regions[b], key, lbs, self._val_small_masks))
            self._large_f1_acc.append(
                self._compute_f1_from_masks(pred_regions[b], key, lbs, self._val_large_masks))

        return {'loss': loss.detach().cpu().numpy(),
                'tp_hard': tp_h, 'fp_hard': fp_h, 'fn_hard': fn_h}

    def _compute_f1_from_masks(self, pred_regions_b, key, bbox_lbs, mask_store):
        if key not in mask_store:
            return np.full(len(self.REGION_ORDER), np.nan)
        full_masks = mask_store[key]
        f1 = np.full(len(self.REGION_ORDER), np.nan)
        H, W, D = pred_regions_b.shape[1:]
        lbs = bbox_lbs
        for i, region in enumerate(self.REGION_ORDER):
            if region not in full_masks:
                continue
            full_target = full_masks[region]
            fH, fW, fD = full_target.shape
            crop_x0 = max(0, -lbs[0]); full_x0 = max(0, lbs[0])
            crop_y0 = max(0, -lbs[1]); full_y0 = max(0, lbs[1])
            crop_z0 = max(0, -lbs[2]); full_z0 = max(0, lbs[2])
            crop_x1 = min(H, fH - lbs[0]); full_x1 = min(fH, lbs[0] + H)
            crop_y1 = min(W, fW - lbs[1]); full_y1 = min(fW, lbs[1] + W)
            crop_z1 = min(D, fD - lbs[2]); full_z1 = min(fD, lbs[2] + D)
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0 or crop_z1 <= crop_z0:
                continue
            target_crop = full_target[full_x0:full_x1, full_y0:full_y1, full_z0:full_z1]
            n_target = target_crop.sum()
            if n_target == 0:
                continue
            target_t = torch.from_numpy(target_crop).to(pred_regions_b.device)
            pred_region = pred_regions_b[i, crop_x0:crop_x1, crop_y0:crop_y1, crop_z0:crop_z1]
            tp = (target_t & pred_region).sum().item()
            fn = n_target - tp
            fp = (pred_region & ~target_t).sum().item()
            denom = 2 * tp + fp + fn
            f1[i] = (2 * tp / denom) if denom > 0 else np.nan
        return f1

    def on_validation_epoch_end(self, val_outputs):
        def _log(acc, label):
            if not acc:
                return
            all_f1 = np.array(acc)
            names = self.REGION_ORDER[:all_f1.shape[1]]
            with np.errstate(invalid='ignore'):
                mf1 = np.nanmean(all_f1, axis=0)
                nv = np.sum(~np.isnan(all_f1), axis=0)
            s = ', '.join(
                f"{n}={'n/a' if np.isnan(v) else f'{v:.3f}'} (n={int(k)})"
                for n, v, k in zip(names, mf1, nv)
            )
            self.print_to_log_file(f"[{label} - {len(all_f1)} cases] {s}")

        _log(self._small_f1_acc, "SmallLesionF1")
        _log(self._large_f1_acc, "LargeLesionF1")
        self._small_f1_acc = []
        self._large_f1_acc = []
        super().on_validation_epoch_end(val_outputs)

    # -------------------------------------------------------------------------
    # NETWORK ARCHITECTURE
    # -------------------------------------------------------------------------
    
    @classmethod
    def build_network_architecture(cls, *args, **kwargs):
        is_training_instance = (
            len(args) > 0 and 
            not isinstance(args[0], str) and
            hasattr(args[0], 'configuration_manager') and 
            hasattr(args[0], 'label_manager')
        )

        if is_training_instance:
            self = args[0]
            if len(args) > 1 and isinstance(args[1], dict):
                arch_kwargs = dict(args[1])
            else:
                arch_kwargs = dict(self.configuration_manager.network_arch_init_kwargs)
            if len(args) > 2 and isinstance(args[2], list):
                req_import = args[2]
            else:
                req_import = self.configuration_manager.network_arch_init_kwargs_req_import
            num_input_channels = args[3] if len(args) > 3 else 4
            num_output_channels = args[4] if len(args) > 4 else self.label_manager.num_segmentation_heads
            log_fn = self.print_to_log_file
        else:
            if len(args) > 1 and isinstance(args[1], dict):
                arch_kwargs = dict(args[1])
            else:
                arch_kwargs = {}
            if len(args) > 2 and isinstance(args[2], list):
                req_import = args[2]
            else:
                req_import = []
            num_input_channels = args[3] if len(args) > 3 else 4
            num_output_channels = args[4] if len(args) > 4 else 4
            log_fn = print

        for key in req_import:
            if key not in arch_kwargs or arch_kwargs[key] is None:
                continue
            module_path = arch_kwargs[key]
            if not isinstance(module_path, str):
                continue
            parts = module_path.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = importlib.import_module(module_name)
            arch_kwargs[key] = getattr(module, class_name)

        all_kwargs = dict(arch_kwargs)
        all_kwargs['input_channels'] = num_input_channels
        all_kwargs['num_classes'] = num_output_channels
        all_kwargs['encoder_mode'] = cls.ENCODER_MODE
        all_kwargs['use_attention_fusion'] = cls.USE_ATTENTION_FUSION
        all_kwargs['rc_threshold'] = cls.RC_THRESHOLD

        network = MultiModeResidualEncoderUNet(**all_kwargs)

        if is_training_instance:
            total = sum(p.numel() for p in network.parameters())
            dec = sum(p.numel() for p in network.decoder.parameters())
            if cls.ENCODER_MODE == "single":
                enc = sum(p.numel() for p in network.encoder.parameters())
                log_msg = (
                    f"[MultiMode] Mode=SINGLE | ClinicalPrior={cls.ENABLE_CLINICAL_PRIOR} | "
                    f"ModalityDropout={'single' if cls.ENABLE_MODALITY_DROPOUT else 'OFF'} | "
                    f"SmallLesion={cls.ENABLE_SMALL_LESION_OVERSAMPLING} | "
                    f"Built: {total:,} total params | encoder={enc:,} decoder={dec:,}"
                )
            elif cls.ENCODER_MODE == "dual":
                enc1 = sum(p.numel() for p in network.encoder_t1.parameters())
                enc2 = sum(p.numel() for p in network.encoder_t2.parameters())
                fusion = sum(p.numel() for p in network.fusion_modules.parameters())
                fusion_name = "ATTENTION" if cls.USE_ATTENTION_FUSION else "1x1conv"
                log_msg = (
                    f"[MultiMode] Mode=DUAL | Fusion={fusion_name} | RC_threshold={cls.RC_THRESHOLD} | "
                    f"ClinicalPrior={cls.ENABLE_CLINICAL_PRIOR} | "
                    f"ModalityDropout={'grouped' if cls.ENABLE_MODALITY_DROPOUT else 'OFF'} | "
                    f"SmallLesion={cls.ENABLE_SMALL_LESION_OVERSAMPLING} | "
                    f"Built: {total:,} total params | enc_T1/T1ce={enc1:,} enc_T2/FLAIR={enc2:,} "
                    f"fusion={fusion:,} decoder={dec:,}"
                )
            else:
                enc1 = sum(p.numel() for p in network.encoder_t1.parameters())
                enc2 = sum(p.numel() for p in network.encoder_t2.parameters())
                enc3 = sum(p.numel() for p in network.encoder_spectral.parameters())
                spectral = sum(p.numel() for p in network.spectral_extractor.parameters())
                fusion = sum(p.numel() for p in network.fusion_modules.parameters())
                fusion_name = "ATTENTION" if cls.USE_ATTENTION_FUSION else "1x1conv"
                log_msg = (
                    f"[MultiMode] Mode=TRIPLE | Fusion={fusion_name} | RC_threshold={cls.RC_THRESHOLD} | "
                    f"ClinicalPrior={cls.ENABLE_CLINICAL_PRIOR} | "
                    f"ModalityDropout={'grouped' if cls.ENABLE_MODALITY_DROPOUT else 'OFF'} | "
                    f"SmallLesion={cls.ENABLE_SMALL_LESION_OVERSAMPLING} | "
                    f"Built: {total:,} total params | enc_T1/T1ce={enc1:,} enc_T2/FLAIR={enc2:,} "
                    f"enc_spectral={enc3:,} spectral_extractor={spectral:,} fusion={fusion:,} decoder={dec:,}"
                )
            log_fn(log_msg)

        return network

    def validation_step(self, batch: dict) -> dict:
        if self.network.encoder_mode == "single":
            return super().validation_step(batch)
        
        original_threshold = self.network.rc_threshold
        self.network.rc_threshold = 0.5
        try:
            class _ListWrapper(torch.nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self._net = net
                def forward(self, x):
                    out = self._net(x)
                    if isinstance(out, torch.Tensor):
                        return [out]
                    return out
                def __getattr__(self, name):
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self._net, name)
            
            original_network = self.network
            self.network = _ListWrapper(original_network)
            try:
                result = super().validation_step(batch)
            finally:
                self.network = original_network
        finally:
            self.network.rc_threshold = original_threshold
        return result





if __name__ == "__main__":
    print(
        "nnUNetTrainerStagedTverskyDoubleBranch (MultiMode) ready.\n\n"
        "Encoder modes:\n"
        "  single:  Standard 4-ch nnU-Net\n"
        "  dual:    Two encoders (T1/T1ce + T2/FLAIR) + fusion [DEFAULT]\n"
        "  triple:  Three encoders (+ spectral features) + fusion\n\n"
        "Integrated features (all toggleable):\n"
        "  - Clinical Prior attention map\n"
        "  - Modality dropout (single: per-channel | dual/triple: max 1 per group)\n"
        "  - Small-lesion oversampling with CC-Dice boost + MIL bags\n"
        "  - Configurable RC activation threshold\n\n"
        "Pre-configured variants:\n"
        "  nnUNetTrainerStagedTverskySingle           (single encoder)\n"
        "  nnUNetTrainerStagedTverskyDoubleBranch       (dual, 1x1 fusion) [default]\n"
        "  nnUNetTrainerStagedTverskyDualAttention     (dual, attention)\n"
        "  nnUNetTrainerStagedTverskyTriple            (triple, 1x1 fusion)\n"
        "  nnUNetTrainerStagedTverskyTripleAttention   (triple, attention)\n"
        "  nnUNetTrainerStagedTverskyRCConservative    (RC threshold 0.7)\n"
        "  nnUNetTrainerStagedTverskyRCAggressive      (RC threshold 0.3)\n"
        "  nnUNetTrainerStagedTverskyNoDropout         (no modality dropout)\n"
        "  nnUNetTrainerStagedTverskyNoPrior           (no clinical prior)\n"
        "  nnUNetTrainerStagedTverskyNoSmallLesion     (no small-lesion oversampling)"
    )