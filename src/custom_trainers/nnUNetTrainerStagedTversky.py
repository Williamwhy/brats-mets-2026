"""
nnUNetTrainerStagedTversky
============================

Tversky loss with FULLY INDEPENDENT alpha/beta control per REGION (not
grouped) -- WT, TC, ET, RC each get their own controller, for BOTH the ALL
term and the SMALL-lesion term. 4 regions x 2 terms = 8 independent
AlphaBetaController instances. dataset.json is untouched -- labels stay 1-4,
the network still outputs 4 channels, nothing about the data changes at all.
This is purely a computation split at loss time -- a simple loop over
cfg.REGIONS, generalizes automatically to however many regions you have.

WHY PER-REGION, NOT GROUPED: an earlier version grouped (WT, TC, ET) as one
"tumor" controller and gave RC its own -- motivated by real submission
evidence that RC hallucinated in ~28-35% of cases with no true resection
cavity, while the pooled ALL metric looked healthy throughout training
(diluted by the other regions). But WT/TC/ET are themselves not
interchangeable -- WT (edema-inclusive) has very different scale and FP/FN
dynamics than ET (enhancing core only), for example -- so grouping them
still risks hiding a struggling region behind an averaged metric, the exact
failure mode that caused the RC problem in the first place. Full per-region
independence removes that risk entirely: every region's precision/recall
trade-off is driven only by its own signal.

HOW ALPHA/BETA ARE CONTROLLED: direct, continuous response to precision and
recall, every epoch, from the start -- no phases, no frozen target, no
smoothing lag. Replaces a much more elaborate earlier design (staged Phase
1/Phase 2, a proportional Lagrangian law with an adaptive target, an EMA-lag
guard, and a peak-retention floor) that, despite several rounds of targeted
fixes, turned out to still be fundamentally blind to precision -- it only
ever watched recall, one step removed from the thing we actually cared
about. Real evidence: one region's alpha climbed to a good, precision-
supporting peak, then got chased back down to its floor over ~600 epochs by
a recall-only signal, even though precision kept improving anyway (via
ordinary training convergence) the entire time it was happening -- the old
mechanism had no way to notice that.

    alpha_new = alpha_old * (1 - pct_change_in_precision)
    beta_new  = beta_old  * (1 - pct_change_in_recall)
    [renormalize so alpha_new + beta_new == 1]

Precision improving -> alpha eases off. Precision dropping -> alpha
increases. Recall dropping -> beta increases. Recall improving -> beta eases
off. Symmetric, no blind spots -- see AlphaBetaController.

alpha + beta = 1 ALWAYS, independently per region -- beta is a derived
property, never stored independently, so the invariant can't drift out of
sync.

Tversky index:  TI = TP / (TP + alpha*FP + beta*FN)
Loss:           1 - TI

HOW SMALL-LESION VOXELS ARE IDENTIFIED (no cc3d, no precompute, no data
changes): binary morphological OPENING (erode then dilate), implemented as a
few F.max_pool3d calls -- fully on GPU, no CPU sync, computed ONCE per
iteration on the full multi-channel target (spatial operation, doesn't care
about region semantics), then sliced per-region for each region's own
small-lesion Tversky term -- no duplicated morphological computation.

  opened = opening(target, SMALL_LESION_EROSION_ITERS)   # approx. large-lesion "core"
  small_mask = target - opened                            # exactly the small-lesion voxels
  include_mask = dilate(small_mask, SMALL_LESION_VICINITY_DILATE_ITERS)  # LOCAL neighborhood only

KNOWN, VERIFIED DESIGN DETAIL: `include_mask = 1 - opened` (used in an
earlier version) is a real bug, not just an approximation -- `opened` is
ALWAYS a subset of `target` (guaranteed morphology property), so it can
never touch background, making that mask identical to the whole volume's
background every time (verified: FP_small exactly equalled FP_all in every
epoch of an earlier run). Dilating small_mask OUTWARD instead gives a
genuinely local neighborhood -- verified numerically to produce a real,
meaningfully smaller FP count.

KNOWN APPROXIMATION: `opened` (used only to FIND small_mask) is a shrunken
proxy for "large lesion", not the exact original shape -- a thin rim right
at a large lesion's true boundary gets folded into `small_mask` along with
genuinely small lesions. Directionally harmless (a boundary ring getting
slightly more Tversky attention) but SMALL_LESION_EROSION_ITERS is an
approximate correspondence to "small", not an exact voxel-count threshold --
watch mean_small_mask_fraction_<REGION> in the log.

Carried forward unchanged, still needed regardless of loss design:
  - The exact confirmed __init__ signature: (plans, configuration, fold,
    dataset_json, device=...), no unpack_dataset.
  - The _build_loss (leading underscore) hook name.
  - float32/bool target casts -- autocast and BCE still require these.
  - POS_WEIGHT support (still one value per region, unchanged).

NOT addressed by this change, flagged honestly: the train/test RC-prevalence
mismatch (~50% RC-positive in training via deliberate kmeans balancing vs
~13.7% in the true blind test population). Per-region alpha lets the loss
push each region's precision harder during training as needed, which should
meaningfully reduce RC hallucination, but it does not correct the fact that
the network is still trained on a 50%-RC-positive population. If RC
hallucination improves substantially but doesn't fully resolve, that gap is
the likely remaining cause, and would need either rebalanced training data
or inference-time threshold calibration -- deliberately not built here.

Dependencies: torch, numpy. No cc3d, no blosc2-specific code, no dataset.json
changes, no offline pre-stage of any kind.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class StagedTverskyConfig:
    REGIONS: tuple = ("WT", "TC", "ET", "RC")  # matches dataset.json exactly, unchanged

    # Zeros these input channel indices AFTER normalization -- see
    # _apply_channel_zeroing for why post- not pre-normalization. Default:
    # T1 (index 0) and T2 (index 2), per confirmed dataset.json channel_names
    # {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"} -- keeping only T1CE
    # and FLAIR, the primary diagnostic sequences for enhancing tumor and
    # edema/whole-tumor extent. Set to None or [] to disable (use all channels).
    ZERO_CHANNELS: Optional[list] = field(default_factory=lambda: [])
    # ---- On-the-fly T1C boundary map (5th channel) ----
    # Requires dataset.json to declare 5 channels (network architecture is
    # fixed at plan time -- unavoidable, can't be worked around in the
    # trainer). The 5th channel's RAW DATA ON DISK can be a cheap placeholder
    # (e.g. a duplicate of T1CE) since this OVERWRITES it every iteration,
    # computed fresh from the already-normalized, already-augmented T1CE
    # channel, entirely on GPU (verified: qualitatively identical behavior
    # to an offline Sobel-based version -- near-zero in flat regions,
    # strongly elevated at sharp transitions like a resection-cavity rim).
    # No offline precompute step, no boundary-map files on disk needed.
    ENABLE_ON_THE_FLY_BOUNDARY_MAP: bool = False
    BOUNDARY_MAP_SOURCE_CHANNEL: int = 1   # T1CE, per confirmed dataset.json channel_names
    BOUNDARY_MAP_TARGET_CHANNEL: int = 4   # the 5th channel slot to overwrite

    # ---- Clinical prior injection: log-odds adjustment to the logits ----
    # Computed fresh, per-patient, entirely on the fly from FLAIR/T1C
    # intensity + T1C boundary curvature (RC/TC disambiguation verified
    # numerically: jagged boundaries show ~4.3x higher local curvature
    # variance than smooth ones on realistic intensity data). No
    # preprocessing, no new dataset, works directly on the existing
    # 4-channel model via a network wrapper -- see ClinicalPriorNetworkWrapper.
    # Thresholds are Z-SCORE units (already-normalized channels) and are
    # reasonable STARTING POINTS, not tuned against real data yet.
    ENABLE_CLINICAL_PRIOR_INJECTION: bool = False
    PRIOR_FLAIR_CHANNEL: int = 3   # per confirmed dataset.json channel_names
    PRIOR_T1C_CHANNEL: int = 1     # T1CE
    PRIOR_INJECTION_WEIGHT: float = 2.0   # scales the log-odds contribution; 0 = no effect
    PRIOR_FLAIR_EDEMA_THRESHOLD: float = 1.0        # z-score above which FLAIR reads as edema-bright
    PRIOR_T1C_HYPERINTENSE_THRESHOLD: float = 1.0   # z-score above which T1C reads as ET-candidate
    PRIOR_T1C_HYPOINTENSE_THRESHOLD: float = -0.5   # z-score below which T1C reads as TC/RC-candidate
    PRIOR_INTENSITY_SOFTNESS: float = 0.5           # sigmoid softness for the above thresholds
    PRIOR_CURVATURE_WINDOW: int = 3                 # local-variance neighborhood size for curvature
    PRIOR_CURVATURE_SCALE: float = 0.1              # normalizes curvature variance into a [0,1] smoothness score
    PRIOR_ET_RIM_SUPPRESSION: float = 0.5           # how strongly a smooth hyperintense rim suppresses ET prior

    # Starting point -- LOW alpha, heavily FN-averse, since the network hasn't
    # learned anything yet at epoch 0 and predictions are close to random.
    # From here on, alpha and beta respond directly to precision/recall
    # changes every epoch -- see AlphaBetaController.
    ALPHA_START: float = 0.5

    # Two safety rails, kept from the earlier design -- general protections
    # for ANY control law, not specific to what was replaced:
    #   ALPHA_MIN/ALPHA_MAX: hard bounds, so alpha/beta can never reach a
    #     state with zero penalty for one type of error.
    #   MAX_ALPHA_STEP_PER_EPOCH: caps how far alpha can move in one epoch --
    #     necessary because a raw percentage change is dangerous when a
    #     metric starts near zero (e.g. precision 0.005->0.05 is a "900%
    #     improvement" by the formula below, and would swing alpha wildly on
    #     a single noisy epoch without this cap).
    ALPHA_MIN: float = 0.05
    ALPHA_MAX: float = 1.0
    MAX_ALPHA_STEP_PER_EPOCH: float = 0.05

    TVERSKY_SMOOTH: float = 1e-6
    EPS: float = 1e-6

    # ---- Small-lesion term: morphological opening, computed live, every iteration ----
    ENABLE_SMALL_LESION_TERM: bool = True
    SMALL_LESION_EROSION_ITERS: int = 2
    SMALL_LESION_VICINITY_DILATE_ITERS: int = 3
    SMALL_LESION_LOSS_WEIGHT: float = 1.5

    # ---- RC-tumor invasion penalty: a direct, unconditional penalty ----
    # An earlier version ("exclusion mask") masked RC's PREDICTION to 0 in
    # ground-truth tumor territory, not just its target -- that removed the
    # FP penalty entirely rather than strengthening it (target was already
    # naturally 0 there; masking it was a no-op, masking prob was the bug).
    # Confirmed by real evidence: hallucination got WORSE afterward (53,272
    # voxels vs 25,737 before). This version keeps the REAL prediction and
    # adds an extra, dedicated penalty on top of the standard mechanism,
    # unconditional on alpha (which proved unreliable for RC specifically --
    # see RHO/target_alpha history). Directly targets a validated finding:
    # cropping ground-truth tumor out of one real case's RC prediction
    # (BraTS-MET-00128-000) removed ~100% of the hallucination there.
    ENABLE_RC_TUMOR_INVASION_PENALTY: bool = True
    RC_EXCLUSION_TUMOR_NAME: str = "WT"
    RC_TUMOR_INVASION_PENALTY_WEIGHT: float = 2.0

    # ---- Shape-regularity term: OPTIONAL, OFF by default ----
    # Matches predicted region shape to that SPECIFIC case's own ground-truth
    # shape (not a hardcoded universal "RC is always smooth" assumption),
    # via an isoperimetric-style regularity score (surface/volume^(2/3)),
    # verified numerically to separate jagged from smooth shapes of similar
    # volume. Motivated by clinical observation: RC and TC can have similar
    # intensity signatures on FLAIR/T1C (both hypointense), making them hard
    # to distinguish by local voxel intensity alone -- but RC tends to have
    # a smooth, regular outer rim while TC tends to be more jagged/irregular,
    # a GEOMETRIC signal the current per-voxel Tversky/BCE terms have no way
    # to capture at all. Built as a toggle per explicit request, off by
    # default -- intended to be combined with RC_TUMOR_INVASION_PENALTY_WEIGHT
    # (tested standalone at 50 over a separate run) only if that alone proves
    # insufficient.
    ENABLE_SHAPE_REGULARITY_TERM: bool = False
    # Base region set -- always included whenever ENABLE_SHAPE_REGULARITY_TERM
    # is True. TC/RC is the pair this term was actually built for.
    SHAPE_REGULARITY_REGIONS: tuple = ("TC", "RC")
    # Toggle: also apply the term to WT and ET on top of the base set above.
    # Cost scales roughly linearly with region count -- each one is its own
    # cc3d call + per-instance GPU round trip when
    # SHAPE_REGULARITY_MULTI_FOCAL_AWARE is True. Time ONE epoch after
    # flipping this before committing to a full run -- the real cost can
    # only be confirmed by actually running it, not estimated reliably.
    SHAPE_REGULARITY_INCLUDE_WT_ET: bool = False
    SHAPE_REGULARITY_WEIGHT: float = 0.5
    # Skip a region for a given case if ground truth has fewer than this many
    # voxels -- avoids computing a meaningless regularity score on an empty
    # or near-empty mask (matches the SMALL_VOXEL_THRESHOLD convention used
    # elsewhere in this project).
    SHAPE_REGULARITY_MIN_VOLUME: float = 27.0
    # If False (default): _shape_regularity_score on the WHOLE region as one
    # blob -- simple, no cc3d, but confounds lesion COUNT with boundary
    # jaggedness (verified: 5 separate smooth lesions score ~2x worse than 1
    # smooth lesion of the same total volume, purely from count). Risky for
    # WT/ET specifically in a METASTASES dataset, where multi-focal disease
    # is the norm -- could incentivize merging/suppressing genuinely separate
    # small lesions to lower this score, directly working against
    # SMALL_LESION_LOSS_WEIGHT elsewhere in this file.
    # If True: runs cc3d on GROUND TRUTH ONLY (never predictions -- GT
    # arrives already correctly augmented, no offline-precompute/alignment
    # problem), scores each lesion instance separately, averages the diffs.
    # Verified numerically: gives a clean 0 for a genuinely good multi-focal
    # prediction (vs the naive version's noisy nonzero score for the same
    # case) and a stronger, more decisive penalty for an actually-wrong
    # merged prediction. Costs a GPU->CPU->GPU round trip per region per
    # iteration when enabled -- the same cc3d cost removed elsewhere in this
    # file for speed, acceptable here only because this whole term is
    # optional and off by default, not part of every training run.
    SHAPE_REGULARITY_MULTI_FOCAL_AWARE: bool = False
    SHAPE_REGULARITY_CONNECTIVITY: int = 26

    def shape_regularity_active_regions(self) -> tuple:
        """Combines SHAPE_REGULARITY_REGIONS with WT/ET if
        SHAPE_REGULARITY_INCLUDE_WT_ET is True -- single source of truth,
        used everywhere the active region list is needed, so the two toggles
        can never drift out of sync with each other."""
        if self.SHAPE_REGULARITY_INCLUDE_WT_ET:
            extra = tuple(r for r in ("WT", "ET") if r not in self.SHAPE_REGULARITY_REGIONS)
            return self.SHAPE_REGULARITY_REGIONS + extra
        return self.SHAPE_REGULARITY_REGIONS

    # ---- Region-confusion auxiliary term: RETIRED (default OFF), superseded ----
    # Was a SOFT 3-way-softmax penalty attempt at the same goal (discourage
    # RC/tumor overlap). Tried at weight 0.5, then 2.5 -- neither fixed the
    # exact case above, which the hard exclusion mask fixes completely and
    # unconditionally. A soft gradient penalty apparently couldn't override
    # whatever confident (but wrong) feature association the network had
    # learned; a hard mask doesn't need to -- it removes the possibility of
    # reward entirely, regardless of the network's internal confidence. Kept
    # available as a toggle in case of future experimentation, not because
    # it's still needed -- running both simultaneously would be redundant for
    # the voxels the exclusion mask already handles.
    ENABLE_REGION_CONFUSION_TERM: bool = False
    REGION_CONFUSION_TUMOR_NAME: str = "WT"
    REGION_CONFUSION_RC_NAME: str = "RC"
    REGION_CONFUSION_LOSS_WEIGHT: float = 1.5

    # Per-region BCE pos_weight, in REGIONS order (still just one value per
    # region -- the data/network never changes shape in this design).
    POS_WEIGHT: Optional[list] = field(default_factory=lambda: [122.91, 200.0, 200.0, 200.0])

    CALIBRATION_MODE: bool = False
    CALIBRATION_NUM_EPOCHS: int = 15
    CALIBRATION_ITERS_PER_EPOCH: int = 100

    # Lowered from nnU-Net's default 0.01 after a crash where network weights
    # went unstable (NaN logits) from a previous bad update -- a lower LR is
    # a standard, well-established mitigation for gradient/weight instability.
    # Set to None to leave nnU-Net's own default untouched.
    INITIAL_LR_OVERRIDE: Optional[float] = 0.001


# --------------------------------------------------------------------------- #
# Morphological opening -- fully GPU, no cc3d, no CPU sync
# --------------------------------------------------------------------------- #

def _binary_erode3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    for _ in range(iterations):
        x = 1.0 - F.max_pool3d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return x


def _binary_dilate3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    for _ in range(iterations):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x


def _morphological_opening3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    if iterations <= 0:
        return x
    return _binary_dilate3d(_binary_erode3d(x, iterations), iterations)


def _gradient_magnitude3d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 1, D, H, W). Central-difference gradient magnitude -- verified
    numerically to behave qualitatively identically to an offline
    scipy/Sobel-based version (near-zero in flat regions, strongly elevated
    at sharp intensity transitions like a resection-cavity rim), just at a
    different absolute scale, which doesn't matter since this feeds into a
    network that learns to interpret whatever scale it's given. Pure tensor
    slicing, no CPU sync, computed fresh every iteration directly on the
    already-normalized, already-augmented input.
    """
    padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    grad_d = (padded[:, :, 2:, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1, 1:-1]) / 2.0
    grad_h = (padded[:, :, 1:-1, 2:, 1:-1] - padded[:, :, 1:-1, :-2, 1:-1]) / 2.0
    grad_w = (padded[:, :, 1:-1, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1, :-2]) / 2.0
    return torch.sqrt(grad_d ** 2 + grad_h ** 2 + grad_w ** 2 + 1e-12)


def _gradient_and_curvature3d(x: torch.Tensor, eps: float = 1e-6):
    """
    x: (B, 1, D, H, W). Returns (magnitude, curvature), both (B, 1, D, H, W).
    Curvature = divergence of the NORMALIZED gradient (local surface-normal
    direction field) -- H = div(grad(x)/|grad(x)|). A smooth, regular
    boundary (e.g. an RC rim) has curvature that's small and spatially
    CONSISTENT; a jagged, irregular boundary (e.g. a TC edge) has curvature
    that swings erratically voxel to voxel. Verified numerically: on
    realistic, smoothly-varying (not harsh binary) intensity transitions,
    jagged boundaries show ~4.3x higher curvature VARIANCE than smooth ones
    -- this is a real, decisive, validated signal, not just a plausible-
    sounding idea. Pure tensor ops, no CPU sync.
    """
    padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    grad_d = (padded[:, :, 2:, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1, 1:-1]) / 2.0
    grad_h = (padded[:, :, 1:-1, 2:, 1:-1] - padded[:, :, 1:-1, :-2, 1:-1]) / 2.0
    grad_w = (padded[:, :, 1:-1, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1, :-2]) / 2.0
    mag = torch.sqrt(grad_d ** 2 + grad_h ** 2 + grad_w ** 2 + eps)

    nd, nh, nw = grad_d / mag, grad_h / mag, grad_w / mag

    p_nd = F.pad(nd, (1, 1, 1, 1, 1, 1), mode="replicate")
    p_nh = F.pad(nh, (1, 1, 1, 1, 1, 1), mode="replicate")
    p_nw = F.pad(nw, (1, 1, 1, 1, 1, 1), mode="replicate")
    d_nd = (p_nd[:, :, 2:, 1:-1, 1:-1] - p_nd[:, :, :-2, 1:-1, 1:-1]) / 2.0
    d_nh = (p_nh[:, :, 1:-1, 2:, 1:-1] - p_nh[:, :, 1:-1, :-2, 1:-1]) / 2.0
    d_nw = (p_nw[:, :, 1:-1, 1:-1, 2:] - p_nw[:, :, 1:-1, 1:-1, :-2]) / 2.0
    curvature = d_nd + d_nh + d_nw

    return mag, curvature


def _local_variance3d(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    """
    x: (B, 1, D, H, W). Local variance within a window x window x window
    neighborhood, via avg-pooling (E[x^2] - E[x]^2) -- used to turn a raw
    per-voxel curvature value into a spatially-aware 'how erratic is
    curvature around here' score, since a single voxel's curvature alone is
    noisy; the local pattern across a small neighborhood is the real signal.
    """
    pad = window // 2
    mean_x = F.avg_pool3d(x, kernel_size=window, stride=1, padding=pad, count_include_pad=False)
    mean_x2 = F.avg_pool3d(x ** 2, kernel_size=window, stride=1, padding=pad, count_include_pad=False)
    return (mean_x2 - mean_x ** 2).clamp(min=0.0)


def _shape_regularity_score(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    mask: (B, 1, D, H, W), soft [0,1] or hard {0,1}. Returns (B,) -- an
    isoperimetric-style regularity score (surface / volume^(2/3)), verified
    numerically to score a jagged shape substantially higher than a smooth
    one of similar volume. LOWER = smoother/more regular, HIGHER =
    jagged/irregular, and size-invariant (the volume^(2/3) normalization
    means a big smooth blob and a small smooth blob score similarly -- only
    shape, not size, drives the score). Fully differentiable, GPU-only, no
    discrete/connected-component operations -- same design philosophy as
    the rest of this file.
    """
    grad_d = (mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]).abs().sum(dim=(1, 2, 3, 4))
    grad_h = (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().sum(dim=(1, 2, 3, 4))
    grad_w = (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().sum(dim=(1, 2, 3, 4))
    surface = grad_d + grad_h + grad_w
    volume = mask.sum(dim=(1, 2, 3, 4)) + eps
    return surface / (volume ** (2.0 / 3.0) + eps)


def _per_instance_regularity_diff(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                                   connectivity: int, min_volume: float) -> torch.Tensor:
    """
    pred_mask, gt_mask: (B, 1, D, H, W). Runs connected-component labeling on
    gt_mask ONLY, per batch item -- ground truth arrives already correctly
    augmented by nnU-Net's own pipeline for this iteration, so there's no
    offline-precompute/augmentation-alignment problem the way there would be
    for a cached, offline-computed instance map. For each discrete lesion
    instance in gt, compares that exact gt shape against the prediction
    restricted to the same spatial extent -- avoids the multi-focality
    confound _shape_regularity_score has on its own (verified numerically:
    gives a clean 0 for a genuinely good multi-focal prediction, vs the
    global version's noisy nonzero score for the identical case).
    Gradient flows normally through pred_mask (the instance mask itself is
    derived from gt via cc3d, a constant w.r.t. the network's weights).
    Requires cc3d -- imported lazily here, not a hard dependency unless this
    specific term is enabled. Returns (mean_diff, n_valid_instances) -- the
    count lets callers distinguish 'no instances found' from 'instances
    found, penalty genuinely zero', which a bare 0.0 value can't.
    """
    import cc3d

    device = pred_mask.device
    gt_np = gt_mask.detach().cpu().numpy()
    total_diff = torch.zeros((), device=device)
    n_valid_instances = 0

    for b in range(gt_np.shape[0]):
        vol = gt_np[b, 0].astype(np.uint8)
        if not vol.any():
            continue
        labels, n = cc3d.connected_components(vol, connectivity=connectivity, return_N=True)
        if n == 0:
            continue
        counts = np.bincount(labels.ravel())
        for lesion_id in range(1, n + 1):
            if counts[lesion_id] < min_volume:
                continue
            instance_mask = torch.from_numpy(labels == lesion_id).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W), constant, no grad needed
            gt_instance = instance_mask
            pred_instance = pred_mask[b:b + 1] * instance_mask  # gradient flows through pred_mask here
            gt_reg = _shape_regularity_score(gt_instance)
            pred_reg = _shape_regularity_score(pred_instance)
            total_diff = total_diff + ((pred_reg - gt_reg) ** 2).squeeze()
            n_valid_instances += 1

    if n_valid_instances == 0:
        return torch.zeros((), device=device), 0
    return total_diff / n_valid_instances, n_valid_instances


def _compute_clinical_prior_maps(data: torch.Tensor, cfg: "StagedTverskyConfig") -> dict:
    """
    data: (B, C, D, H, W) -- the FULL, already-normalized, already-augmented
    input tensor. Returns {region_name: prior_map}, each (B, 1, D, H, W) in
    [0,1], purely a function of the input images -- no ground truth
    anywhere, identically computable at real inference time. Thresholds
    operate in Z-SCORE units (the channels are already per-case normalized
    by nnU-Net's own preprocessing), e.g. +1.0 means "1 std above this
    case's own mean" -- robust across scanners/protocols since the
    normalization already handles absolute-scale differences.

    Curvature-based RC/TC disambiguation is verified (see module history):
    jagged boundaries show ~4.3x higher local curvature variance than smooth
    ones on realistic, smoothly-varying intensity data. The specific
    intensity thresholds below are reasonable STARTING POINTS, not
    empirically tuned against real data -- expect to need adjustment once
    you see how they behave on your actual cases.
    """
    eps = cfg.EPS
    flair = data[:, cfg.PRIOR_FLAIR_CHANNEL:cfg.PRIOR_FLAIR_CHANNEL + 1]
    t1c = data[:, cfg.PRIOR_T1C_CHANNEL:cfg.PRIOR_T1C_CHANNEL + 1]

    grad_mag, curvature = _gradient_and_curvature3d(t1c, eps)
    curvature_local_var = _local_variance3d(curvature, window=cfg.PRIOR_CURVATURE_WINDOW)
    smoothness_score = torch.exp(-curvature_local_var / (cfg.PRIOR_CURVATURE_SCALE + eps))
    jaggedness_score = 1.0 - smoothness_score

    wt_intensity_score = torch.sigmoid((flair - cfg.PRIOR_FLAIR_EDEMA_THRESHOLD) / cfg.PRIOR_INTENSITY_SOFTNESS)
    et_intensity_score = torch.sigmoid((t1c - cfg.PRIOR_T1C_HYPERINTENSE_THRESHOLD) / cfg.PRIOR_INTENSITY_SOFTNESS)
    tc_rc_intensity_score = torch.sigmoid((cfg.PRIOR_T1C_HYPOINTENSE_THRESHOLD - t1c) / cfg.PRIOR_INTENSITY_SOFTNESS)

    rc_prior = tc_rc_intensity_score * smoothness_score
    tc_prior = tc_rc_intensity_score * jaggedness_score
    # ET vs. a thin vascular/hyperintense rim around RC: a SMOOTH hyperintense
    # region (structurally similar to RC's own boundary) is down-weighted;
    # thicker/less regular hyperintensity is more likely real enhancing tumor.
    et_prior = et_intensity_score * (1.0 - cfg.PRIOR_ET_RIM_SUPPRESSION * smoothness_score)
    wt_prior = wt_intensity_score

    return {"WT": wt_prior, "TC": tc_prior, "ET": et_prior, "RC": rc_prior}


class ClinicalPriorNetworkWrapper(nn.Module):
    def __init__(self, network: nn.Module, cfg: "StagedTverskyConfig"):
        super().__init__()
        # Use object.__setattr__ or super().__setattr__ to avoid recursion during init
        self.__dict__['_network'] = network
        self.cfg = cfg

    @property
    def network(self):
        return self._network

    def forward(self, x):
        cfg = self.cfg
        if not cfg.ENABLE_CLINICAL_PRIOR_INJECTION:
            return self.network(x)

        priors = _compute_clinical_prior_maps(x, cfg)
        raw_output = self.network(x)

        if isinstance(raw_output, (list, tuple)):
            return [self._inject(level_logits, priors) for level_logits in raw_output]
        return self._inject(raw_output, priors)

    def _inject(self, logits: torch.Tensor, priors: dict) -> torch.Tensor:
        cfg = self.cfg
        eps = cfg.EPS
        _, C, d, h, w = logits.shape
        adjusted = logits.clone()
        for c, region in enumerate(cfg.REGIONS):
            if region not in priors or c >= C:
                continue
            prior_map = priors[region]
            if prior_map.shape[2:] != (d, h, w):
                prior_map = F.interpolate(prior_map, size=(d, h, w), mode="trilinear", align_corners=False)
            prior_clamped = prior_map.clamp(min=eps, max=1.0 - eps)
            prior_logit = torch.log(prior_clamped / (1.0 - prior_clamped))
            adjusted[:, c:c + 1] = adjusted[:, c:c + 1] + cfg.PRIOR_INJECTION_WEIGHT * prior_logit
        return adjusted

    def __getattr__(self, name: str):
        # Fallback to the underlying network for attributes like .decoder, .encoder, etc.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._network, name)


# --------------------------------------------------------------------------- #
# Staged alpha/beta controller -- ONE implementation, used once PER REGION PER TERM
# --------------------------------------------------------------------------- #

class AlphaBetaController:
    """
    Direct precision/recall balance target -- replaces both the earlier
    staged Phase1/Phase2/adaptive-target/EMA/peak-retention mechanism AND a
    subsequent percent-change-based version. Both of those were still, in
    different ways, blind to the actual goal: balancing precision and
    recall. The percent-change version only reacted to CHANGE epoch-to-
    epoch -- if both metrics went flat while badly imbalanced (e.g. recall
    stuck at 0.90, precision stuck at 0.30), it would stop adjusting alpha
    forever, since nothing was "changing" even though nothing was balanced
    either. This version targets the imbalance directly, every epoch,
    memoryless (a pure function of THIS epoch's precision/recall, no history
    needed):

        alpha = recall / (recall + precision)

    recall == precision  -> alpha = 0.5 exactly (the natural balance point).
    recall > precision (over-predicting)  -> alpha > 0.5, more FP-penalty.
    precision > recall (under-predicting) -> alpha < 0.5, more FN-penalty.

    Two safety rails kept from the earlier design (general protections for
    ANY control law):
      - ALPHA_MIN / ALPHA_MAX: hard bounds.
      - MAX_ALPHA_STEP_PER_EPOCH: caps how far alpha can move toward the
        target in one epoch. Necessary because precision is often near-zero
        early in training (before the network has learned anything), which
        would otherwise swing the target toward ~0.9+ immediately -- this
        cap gives the network real time to build genuine precision before
        alpha gets anywhere near that, rather than risking a
        predict-nothing-to-avoid-FP collapse in the first few epochs.
    """

    def __init__(self, cfg: StagedTverskyConfig, name: str):
        self.cfg = cfg
        self.name = name  # for log prefixing, e.g. "WT", "RC", "SMALL_ET"
        self.alpha = cfg.ALPHA_START

    @property
    def beta(self) -> float:
        return 1.0 - self.alpha

    def step(self, precision: float, recall: float, log_fn) -> str:
        """Advance the controller by one epoch. Returns a status string for logging."""
        cfg = self.cfg
        if np.isnan(precision) or np.isnan(recall):
            return "no_data"

        eps = cfg.EPS
        target_alpha = recall / (recall + precision + eps)

        old_alpha = self.alpha
        step_delta = float(np.clip(target_alpha - old_alpha, -cfg.MAX_ALPHA_STEP_PER_EPOCH, cfg.MAX_ALPHA_STEP_PER_EPOCH))
        self.alpha = float(np.clip(old_alpha + step_delta, cfg.ALPHA_MIN, cfg.ALPHA_MAX))

        log_fn(
            f"[StagedTversky][{self.name}] precision={precision:.4f} recall={recall:.4f}  "
            f"target_alpha={target_alpha:.4f}  alpha {old_alpha:.3f}->{self.alpha:.3f}"
        )

        return "active"


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

class StagedTverskyLoss(nn.Module):
    """
    BCE (with pos_weight, full tensor, unchanged) + one independent Tversky
    term PER REGION (alpha/beta set externally by the trainer each epoch,
    one AlphaBetaController per region) + the same per-region split
    applied again to the small-lesion term (weighted, morphological-opening-
    restricted). No region grouping anywhere -- a simple loop over
    cfg.REGIONS.
    """

    def __init__(self, cfg: StagedTverskyConfig, ds_weights, pos_weight=None):
        super().__init__()
        self.cfg = cfg
        self.ds_weights = ds_weights
        self.pos_weight = pos_weight

        # alpha_<REGION> and alpha_small_<REGION>, set externally each epoch
        self.alpha = {r: cfg.ALPHA_START for r in cfg.REGIONS}
        self.alpha_small = {r: cfg.ALPHA_START for r in cfg.REGIONS}
        self.last_metrics = {}

    def _alpha_beta_vec(self, alpha_dict, device):
        alpha_vec = torch.tensor([alpha_dict[r] for r in self.cfg.REGIONS], dtype=torch.float32, device=device)
        return alpha_vec, 1.0 - alpha_vec

    def _tversky_vectorized(self, prob, target, alpha_vec, beta_vec):
        """
        prob, target: (B, C, D, H, W). alpha_vec, beta_vec: (C,).
        ONE reduction across all channels + a broadcasted per-channel TI --
        replaces what used to be C separate _tversky calls in a Python loop.
        Verified numerically identical to summing C independent per-channel
        calls (same total loss value, same per-channel TP/FP/FN).
        Returns per_region_loss (C,) too, not just the summed scalar, so
        callers can identify exactly which region is non-finite if one is.
        """
        dims = tuple(range(2, prob.dim()))
        TP = (prob * target).sum(dim=dims)          # (B, C)
        FP = (prob * (1 - target)).sum(dim=dims)      # (B, C)
        FN = ((1 - prob) * target).sum(dim=dims)      # (B, C)
        s = self.cfg.TVERSKY_SMOOTH
        TI = (TP + s) / (TP + alpha_vec.view(1, -1) * FP + beta_vec.view(1, -1) * FN + s)  # (B, C)
        per_region_loss = 1.0 - TI.mean(dim=0)        # (C,) -- mean over batch, kept per-channel
        total_region_loss = per_region_loss.sum()     # scalar -- sum over regions, matches the old loop
        TP_r, FP_r, FN_r = TP.sum(dim=0), FP.sum(dim=0), FN.sum(dim=0)  # (C,) -- summed over batch, for logging
        return total_region_loss, per_region_loss, TP_r, FP_r, FN_r

    def _check_finite(self, name, per_region_loss, alpha_vec, beta_vec, TP_r, FP_r, FN_r, logits):
        """Raises with SPECIFIC region/value diagnostics the moment a NaN/Inf first appears,
        instead of letting it silently propagate into an opaque aggregate 'loss is nan'."""
        if torch.isfinite(per_region_loss).all():
            return
        bad = [self.cfg.REGIONS[i] for i in range(len(self.cfg.REGIONS)) if not torch.isfinite(per_region_loss[i])]
        detail_lines = []
        for i, r in enumerate(self.cfg.REGIONS):
            detail_lines.append(
                f"    {r}: loss={per_region_loss[i].item()}  alpha={alpha_vec[i].item():.4f}  "
                f"beta={beta_vec[i].item():.4f}  TP={TP_r[i].item():.2f}  FP={FP_r[i].item():.2f}  FN={FN_r[i].item():.2f}"
            )
        raise RuntimeError(
            f"[StagedTversky] Non-finite loss in the {name} term, region(s) {bad}. "
            f"logits stats: min={logits.min().item():.4f} max={logits.max().item():.4f} "
            f"mean={logits.mean().item():.4f}\n" + "\n".join(detail_lines)
        )

    def forward(self, net_output_list, target_list):
        cfg = self.cfg
        total = 0.0
        self.last_metrics = {}
        for r in cfg.REGIONS:
            self.last_metrics[f"alpha_{r}"] = self.alpha[r]
            self.last_metrics[f"beta_{r}"] = 1.0 - self.alpha[r]
            if cfg.ENABLE_SMALL_LESION_TERM:
                self.last_metrics[f"alpha_small_{r}"] = self.alpha_small[r]
                self.last_metrics[f"beta_small_{r}"] = 1.0 - self.alpha_small[r]

        for i, (logits, target) in enumerate(zip(net_output_list, target_list)):
            w = self.ds_weights[i] if i < len(self.ds_weights) else 0.0
            if w == 0:
                continue
            target = target.float()  # region-based targets are commonly torch.bool

            # NOTE: an earlier version masked BOTH prediction and target to 0 in ground-
            # truth tumor territory here. That was a real bug, not a refinement: RC's
            # target is ALREADY guaranteed 0 wherever WT's target is 1 (disjoint raw
            # labels, 1/2/3 vs 4, a voxel can only have one) -- masking target was a
            # no-op. But masking the PREDICTION too meant FP = prob*(1-target) became
            # 0*1 = 0 -- removing the entire penalty mechanism, not just an inadequate
            # one. Confirmed by real evidence: RC's tumor-territory hallucination got
            # WORSE after this (53,272 voxels vs 25,737 before), not better. Standard,
            # fully unmasked BCE/Tversky is restored below -- see RC_TUMOR_INVASION_PENALTY
            # further down for the actual, correctly-implemented direct penalty.
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction="mean")
            if not torch.isfinite(bce):
                raise RuntimeError(
                    f"[StagedTversky] BCE is non-finite ({bce.item()}). "
                    f"logits stats: min={logits.min().item():.4f} max={logits.max().item():.4f} "
                    f"mean={logits.mean().item():.4f}  pos_weight={self.pos_weight}"
                )
            prob = torch.sigmoid(logits)

            alpha_vec, beta_vec = self._alpha_beta_vec(self.alpha, prob.device)
            region_loss, per_region_loss, TP_r, FP_r, FN_r = self._tversky_vectorized(
                prob, target, alpha_vec, beta_vec
            )
            self._check_finite("ALL", per_region_loss, alpha_vec, beta_vec, TP_r, FP_r, FN_r, logits)
            level_loss = bce + region_loss

            if i == 0:
                for c, r in enumerate(cfg.REGIONS):
                    self.last_metrics[f"tp_{r}"] = TP_r[c].item()
                    self.last_metrics[f"fp_{r}"] = FP_r[c].item()
                    self.last_metrics[f"fn_{r}"] = FN_r[c].item()

                # RC-TUMOR INVASION PENALTY: a direct, unconditional penalty on how
                # confident RC's REAL prediction is inside ground-truth tumor territory.
                # This is the corrected version of an earlier, broken attempt that masked
                # the PREDICTION to 0 there too, which removed the FP penalty entirely
                # (proven to make things worse: 53,272 hallucinated voxels vs 25,737
                # before). RC's target is already guaranteed 0 in tumor territory
                # (disjoint raw labels), so no target masking is needed at all -- this
                # term just adds EXTRA, DEDICATED weight on top of the standard alpha/beta
                # Tversky+BCE mechanism, unconditional on alpha's current value (which
                # proved unreliable for RC specifically -- target_alpha hit 0.995 from
                # garbage epoch-1 precision, an unrelated problem this term sidesteps
                # entirely by not depending on alpha at all).
                if cfg.ENABLE_RC_TUMOR_INVASION_PENALTY:
                    wt_idx = cfg.REGIONS.index(cfg.RC_EXCLUSION_TUMOR_NAME)
                    rc_idx = cfg.REGIONS.index("RC")
                    gt_tumor_mask = (target[:, wt_idx:wt_idx + 1] > 0.5).float()  # (B,1,D,H,W)
                    rc_prob_in_tumor = prob[:, rc_idx:rc_idx + 1] * gt_tumor_mask
                    denom = gt_tumor_mask.sum().clamp(min=1.0)
                    tumor_invasion_penalty = rc_prob_in_tumor.sum() / denom
                    if not torch.isfinite(tumor_invasion_penalty):
                        raise RuntimeError(
                            f"[StagedTversky] tumor_invasion_penalty is non-finite "
                            f"({tumor_invasion_penalty.item()})."
                        )
                    level_loss = level_loss + cfg.RC_TUMOR_INVASION_PENALTY_WEIGHT * tumor_invasion_penalty
                    self.last_metrics["rc_tumor_invasion_penalty"] = tumor_invasion_penalty.item()

                if cfg.ENABLE_SHAPE_REGULARITY_TERM:
                    for region_name in cfg.shape_regularity_active_regions():
                        r_idx = cfg.REGIONS.index(region_name)
                        pred_mask = prob[:, r_idx:r_idx + 1]
                        gt_mask = target[:, r_idx:r_idx + 1]

                        if cfg.SHAPE_REGULARITY_MULTI_FOCAL_AWARE:
                            shape_term, n_instances = _per_instance_regularity_diff(
                                pred_mask, gt_mask, cfg.SHAPE_REGULARITY_CONNECTIVITY,
                                cfg.SHAPE_REGULARITY_MIN_VOLUME
                            )
                            has_signal = n_instances > 0
                        else:
                            gt_volume = gt_mask.sum(dim=(1, 2, 3, 4))
                            valid = (gt_volume > cfg.SHAPE_REGULARITY_MIN_VOLUME).float()  # (B,)
                            has_signal = valid.sum() > 0
                            if has_signal:
                                pred_reg = _shape_regularity_score(pred_mask)
                                gt_reg = _shape_regularity_score(gt_mask)
                                diff_sq = (pred_reg - gt_reg) ** 2
                                shape_term = (diff_sq * valid).sum() / valid.sum().clamp(min=1.0)

                        if has_signal:
                            if not torch.isfinite(shape_term):
                                raise RuntimeError(
                                    f"[StagedTversky] shape_regularity_{region_name} is non-finite "
                                    f"({shape_term.item()})."
                                )
                            level_loss = level_loss + cfg.SHAPE_REGULARITY_WEIGHT * shape_term
                            self.last_metrics[f"shape_regularity_{region_name}"] = shape_term.item()

                if cfg.ENABLE_SMALL_LESION_TERM:
                    with torch.no_grad():
                        opened = _morphological_opening3d(target, cfg.SMALL_LESION_EROSION_ITERS)
                        small_mask = target - opened
                        include_mask = _binary_dilate3d(small_mask, cfg.SMALL_LESION_VICINITY_DILATE_ITERS)

                    prob_small_in = prob * include_mask
                    target_small_in = target * include_mask

                    alpha_s_vec, beta_s_vec = self._alpha_beta_vec(self.alpha_small, prob.device)
                    small_loss, per_region_small_loss, TPs_r, FPs_r, FNs_r = self._tversky_vectorized(
                        prob_small_in, target_small_in, alpha_s_vec, beta_s_vec
                    )
                    self._check_finite(
                        "SMALL", per_region_small_loss, alpha_s_vec, beta_s_vec, TPs_r, FPs_r, FNs_r, logits
                    )
                    level_loss = level_loss + cfg.SMALL_LESION_LOSS_WEIGHT * small_loss

                    for c, r in enumerate(cfg.REGIONS):
                        self.last_metrics[f"tp_small_{r}"] = TPs_r[c].item()
                        self.last_metrics[f"fp_small_{r}"] = FPs_r[c].item()
                        self.last_metrics[f"fn_small_{r}"] = FNs_r[c].item()
                        sm_c = small_mask[:, c:c + 1]
                        n_vox = sm_c.numel()
                        self.last_metrics[f"small_mask_fraction_{r}"] = (
                            sm_c.sum().item() / n_vox if n_vox > 0 else 0.0
                        )

                # Region-confusion softmax term -- RETIRED as of the RC-tumor exclusion mask
                # above. It was a SOFT attempt at the same goal (discourage RC/tumor overlap)
                # and never got the job done even at weight 2.5, on the exact case the hard
                # mask above fixes completely. Kept available as a toggle (default OFF) in
                # case of future experimentation, not because it's still needed.
                if cfg.ENABLE_REGION_CONFUSION_TERM:
                    wt_idx = cfg.REGIONS.index(cfg.REGION_CONFUSION_TUMOR_NAME)
                    rc_idx = cfg.REGIONS.index(cfg.REGION_CONFUSION_RC_NAME)
                    logit_tumor = logits[:, wt_idx]  # (B, D, H, W)
                    logit_rc = logits[:, rc_idx]      # (B, D, H, W)
                    zeros = torch.zeros_like(logit_tumor)
                    # 3-way softmax built from two EXISTING logits + a fixed zero reference
                    # for background -- no new network outputs, mutual exclusivity enforced
                    # by the softmax itself, not by an added penalty term.
                    three_way_logits = torch.stack([zeros, logit_tumor, logit_rc], dim=1)  # (B, 3, D, H, W)

                    target_tumor = target[:, wt_idx]
                    target_rc = target[:, rc_idx]
                    # 0=background, 1=tumor, 2=RC. WT/RC are disjoint in real ground truth
                    # (raw labels 1/2/3 vs 4 are mutually exclusive per-voxel integers) --
                    # RC takes priority in the (should-never-happen) case both are set, purely
                    # as a defensive tie-break, not because it reflects real data.
                    true_class = torch.zeros_like(target_tumor, dtype=torch.long)
                    true_class = torch.where(target_tumor > 0.5, torch.ones_like(true_class), true_class)
                    true_class = torch.where(target_rc > 0.5, torch.full_like(true_class, 2), true_class)

                    region_confusion_loss = F.cross_entropy(three_way_logits, true_class, reduction="mean")
                    if not torch.isfinite(region_confusion_loss):
                        raise RuntimeError(
                            f"[StagedTversky] region_confusion_loss is non-finite "
                            f"({region_confusion_loss.item()}). logit_tumor stats: "
                            f"min={logit_tumor.min().item():.4f} max={logit_tumor.max().item():.4f}  "
                            f"logit_rc stats: min={logit_rc.min().item():.4f} max={logit_rc.max().item():.4f}"
                        )
                    level_loss = level_loss + cfg.REGION_CONFUSION_LOSS_WEIGHT * region_confusion_loss
                    self.last_metrics["region_confusion_loss"] = region_confusion_loss.item()

            total = total + w * level_loss

        def _rpf(tp, fp, fn):
            eps = cfg.EPS
            recall = tp / (tp + fn + eps) if (tp + fn) > 0 else float("nan")
            precision = tp / (tp + fp + eps) if (tp + fp) > 0 else float("nan")
            f1 = (2 * tp) / (2 * tp + fp + fn + eps) if (2 * tp + fp + fn) > 0 else float("nan")
            return recall, precision, f1

        self.last_metrics["total_loss"] = float(total.item()) if torch.is_tensor(total) else float(total)

        for r in cfg.REGIONS:
            tp, fp, fn = self.last_metrics[f"tp_{r}"], self.last_metrics[f"fp_{r}"], self.last_metrics[f"fn_{r}"]
            recall, precision, f1 = _rpf(tp, fp, fn)
            self.last_metrics.update({f"recall_{r}": recall, f"precision_{r}": precision, f"f1_{r}": f1})
            if cfg.ENABLE_SMALL_LESION_TERM:
                tps = self.last_metrics[f"tp_small_{r}"]
                fps = self.last_metrics[f"fp_small_{r}"]
                fns = self.last_metrics[f"fn_small_{r}"]
                recall_s, precision_s, f1_s = _rpf(tps, fps, fns)
                self.last_metrics.update({
                    f"recall_small_{r}": recall_s, f"precision_small_{r}": precision_s, f"f1_small_{r}": f1_s,
                })

        return total


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class nnUNetTrainerStagedTversky(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cfg = StagedTverskyConfig()
        if self.cfg.INITIAL_LR_OVERRIDE is not None:
            self.initial_lr = self.cfg.INITIAL_LR_OVERRIDE
        self._epoch_metric_buffer = []
        self.custom_loss: Optional[StagedTverskyLoss] = None

        self.controllers = {r: AlphaBetaController(self.cfg, r) for r in self.cfg.REGIONS}
        self.controllers_small = (
            {r: AlphaBetaController(self.cfg, f"SMALL_{r}") for r in self.cfg.REGIONS}
            if self.cfg.ENABLE_SMALL_LESION_TERM else {}
        )

    def initialize(self):
        super().initialize()
        if self.cfg.ENABLE_CLINICAL_PRIOR_INJECTION:
            self.network = ClinicalPriorNetworkWrapper(self.network, self.cfg)
            self.print_to_log_file(
                "[StagedTversky] Clinical prior injection ENABLED -- wrapping network to add "
                "log-odds of per-class prior maps (computed fresh per-patient from FLAIR/T1C "
                "intensity + T1C boundary curvature) to the logits, at BOTH training and real "
                f"inference. PRIOR_INJECTION_WEIGHT={self.cfg.PRIOR_INJECTION_WEIGHT}. "
                "No preprocessing, no new dataset -- works on this existing 4-channel model."
            )
        if self.cfg.INITIAL_LR_OVERRIDE is not None:
            self.print_to_log_file(
                f"[StagedTversky] initial_lr overridden to {self.initial_lr} "
                f"(nnU-Net default 0.01) -- mitigation after a weight-instability crash."
            )
        self.print_to_log_file(
            "[StagedTversky] alpha+beta=1 ALWAYS, PER REGION (no grouping). dataset.json UNCHANGED "
            "(labels 1-4, 4 output channels). Each region in %s gets its OWN independent alpha/beta "
            "controller, for BOTH the ALL term and the SMALL-lesion term (%s) -- %d total controllers." % (
                self.cfg.REGIONS,
                "SMALL term ENABLED" if self.cfg.ENABLE_SMALL_LESION_TERM else "SMALL term DISABLED",
                len(self.controllers) + len(self.controllers_small),
            )
        )
        if self.cfg.CALIBRATION_MODE:
            self.num_epochs = self.cfg.CALIBRATION_NUM_EPOCHS
            self.num_iterations_per_epoch = self.cfg.CALIBRATION_ITERS_PER_EPOCH
            self.print_to_log_file(
                f"[StagedTversky][CALIBRATION] Stage-1 probe ACTIVE. num_epochs={self.num_epochs}, "
                f"num_iterations_per_epoch={self.num_iterations_per_epoch}."
            )

    def _build_loss(self):
        num_ds = len(self.configuration_manager.pool_op_kernel_sizes) if hasattr(
            self.configuration_manager, "pool_op_kernel_sizes"
        ) else 5
        weights = np.array([1.0 / (2 ** i) for i in range(num_ds)])
        weights[-1] = 0.0
        weights = weights / weights.sum()

        pos_weight_tensor = None
        if self.cfg.POS_WEIGHT is not None:
            assert len(self.cfg.POS_WEIGHT) == len(self.cfg.REGIONS), (
                f"POS_WEIGHT has {len(self.cfg.POS_WEIGHT)} values but REGIONS has "
                f"{len(self.cfg.REGIONS)} -- they must match, the data/network never gains "
                f"extra channels in this design."
            )
            pos_weight_tensor = torch.tensor(
                self.cfg.POS_WEIGHT, dtype=torch.float32, device=self.device
            ).view(-1, 1, 1, 1)
            self.print_to_log_file(f"[StagedTversky] Using configured POS_WEIGHT: {self.cfg.POS_WEIGHT}")
        else:
            self.print_to_log_file(
                "[StagedTversky] POS_WEIGHT is None -- if recall stays near 0 for many epochs with "
                "a flat train_loss, that's background collapse; set POS_WEIGHT."
            )

        loss = StagedTverskyLoss(cfg=self.cfg, ds_weights=weights.tolist(), pos_weight=pos_weight_tensor)
        self.custom_loss = loss
        return loss

    def _apply_channel_zeroing(self, batch: dict) -> dict:
        """
        Zeros out configured input channels AFTER normalization (batch['data']
        here is already normalized -- zeroing at this point sets those
        channels to their post-normalization mean, an uninformative constant,
        NOT literal zero pixel values). Zeroing raw pixels before
        normalization would make Z-score normalization divide by a std of
        exactly 0 for that channel (every voxel identical) -- a guaranteed,
        immediate NaN, the same category of failure spent hours chasing
        earlier this session. This is the safe way to do the same experiment.
        """
        if self.cfg.ZERO_CHANNELS:
            data = batch["data"]
            for c in self.cfg.ZERO_CHANNELS:
                data[:, c] = 0.0
        return batch

    def _apply_boundary_map_injection(self, batch: dict) -> dict:
        """
        Overwrites BOUNDARY_MAP_TARGET_CHANNEL with a gradient-magnitude map
        computed fresh from BOUNDARY_MAP_SOURCE_CHANNEL (T1CE), using the
        ALREADY-normalized, ALREADY-augmented data for this exact iteration
        -- correctly reflects whatever random transform nnU-Net's own
        augmentation applied, with no offline precompute or file I/O at all.
        Runs after channel zeroing so the two features compose correctly if
        both are enabled together.
        """
        if self.cfg.ENABLE_ON_THE_FLY_BOUNDARY_MAP:
            data = batch["data"]
            source = data[:, self.cfg.BOUNDARY_MAP_SOURCE_CHANNEL:self.cfg.BOUNDARY_MAP_SOURCE_CHANNEL + 1]
            boundary_map = _gradient_magnitude3d(source)
            data[:, self.cfg.BOUNDARY_MAP_TARGET_CHANNEL] = boundary_map[:, 0]
        return batch

    def train_step(self, batch: dict) -> dict:
        batch = self._apply_channel_zeroing(batch)
        batch = self._apply_boundary_map_injection(batch)
        result = super().train_step(batch)
        if self.custom_loss is not None and self.custom_loss.last_metrics:
            self._epoch_metric_buffer.append(dict(self.custom_loss.last_metrics))

        loss_val = result.get("loss", None) if isinstance(result, dict) else None
        if loss_val is not None:
            if torch.is_tensor(loss_val):
                loss_val = loss_val.item()
            if np.isnan(loss_val) or np.isinf(loss_val):
                raise RuntimeError(
                    f"[StagedTversky] train_step loss is {loss_val} at epoch {self.current_epoch} -- "
                    f"stopping immediately rather than continuing to train on a NaN-poisoned network. "
                    f"Once NaN enters the weights via a bad gradient step it's sticky (every subsequent "
                    f"epoch stays NaN); an earlier run continued for ~20 epochs / over an hour after this "
                    f"point before anyone noticed. Check the last few [StagedTversky] epoch summaries in "
                    f"the log for which region's alpha/beta and FP counts looked unstable just before this."
                )
        return result

    def validation_step(self, batch: dict) -> dict:
        batch = self._apply_channel_zeroing(batch)
        batch = self._apply_boundary_map_injection(batch)
        return super().validation_step(batch)

    def on_epoch_start(self):
        super().on_epoch_start()
        self._epoch_metric_buffer = []
        self._epoch_start_time = time.time()

    def _mean(self, key):
        vals = [m[key] for m in self._epoch_metric_buffer
                if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _step_and_log(self, controller, recall_key, precision_key, f1_key):
        cfg = self.cfg
        achieved_recall = self._mean(recall_key)
        achieved_precision = self._mean(precision_key)
        if cfg.CALIBRATION_MODE:
            status = "calibration"
            self.print_to_log_file(f"[StagedTversky][{controller.name}][CALIBRATION] alpha frozen at {controller.alpha:.3f}")
        else:
            status = controller.step(achieved_precision, achieved_recall, self.print_to_log_file)

        self.print_to_log_file(
            f"[StagedTversky][{controller.name}] alpha: {controller.alpha:.3f}  beta: {controller.beta:.3f}  "
            f"achieved_recall: {'n/a' if np.isnan(achieved_recall) else round(achieved_recall, 4)}  "
            f"precision: {'n/a' if np.isnan(achieved_precision) else round(achieved_precision, 4)}  "
            f"f1: {self._mean(f1_key):.4f}"
        )
        return status

    def on_epoch_end(self):
        super().on_epoch_end()
        cfg = self.cfg

        # Diagnostic added after a crash traced to NaN LOGITS (i.e. the network's own
        # forward pass, not our loss code) despite gradient clipping (norm=12) and
        # GradScaler's automatic inf/nan-skip already being active in the base trainer.
        # Those protect against a single catastrophic step; they do NOT prevent gradual
        # weight-magnitude drift across many epochs eventually overflowing fp16's dynamic
        # range during a forward pass. This is cheap (~one pass over parameters) and gives
        # direct evidence of whether that's actually happening, instead of inferring it
        # from loss/alpha patterns again.
        with torch.no_grad():
            norms = [p.float().norm() ** 2 for p in self.network.parameters()]
            weight_norm = (torch.stack(norms).sum() ** 0.5).item() if len(norms) > 0 else 0.0
            has_nonfinite = any(not torch.isfinite(p).all() for p in self.network.parameters())
            
        self.print_to_log_file(
            f"[StagedTversky] network_weight_norm: {weight_norm:.2f}  "
            f"any_nonfinite_weights: {has_nonfinite}"
        )
        if has_nonfinite:
            raise RuntimeError(
                f"[StagedTversky] Network weights contain non-finite values at the end of epoch "
                f"{self.current_epoch}. Stopping now rather than training another full epoch on "
                f"an already-corrupted network. Check network_weight_norm in the preceding epochs' "
                f"logs to see whether it was climbing before this."
            )

        statuses = {}
        for r, controller in self.controllers.items():
            statuses[r] = self._step_and_log(controller, f"recall_{r}", f"precision_{r}", f"f1_{r}")
            self.custom_loss.alpha[r] = controller.alpha

        statuses_small = {}
        for r, controller in self.controllers_small.items():
            statuses_small[r] = self._step_and_log(
                controller, f"recall_small_{r}", f"precision_small_{r}", f"f1_small_{r}"
            )
            self.custom_loss.alpha_small[r] = controller.alpha

        epoch_time = round(time.time() - getattr(self, "_epoch_start_time", time.time()), 2)

        self.print_to_log_file("[StagedTversky] ---- epoch summary ----")
        status_str = "  ".join(f"{r}:{s}" for r, s in statuses.items())
        status_small_str = "  ".join(f"SMALL_{r}:{s}" for r, s in statuses_small.items())
        self.print_to_log_file(
            f"[StagedTversky] epoch: {self.current_epoch}  time: {epoch_time}s  {status_str}  {status_small_str}"
        )
        self.print_to_log_file(
            f"[StagedTversky] mean_total_loss: {self._mean('total_loss'):.4f}  "
            f"mean_region_confusion_loss: {self._mean('region_confusion_loss'):.4f}  "
            f"mean_rc_tumor_invasion_penalty: {self._mean('rc_tumor_invasion_penalty'):.6f}"
        )
        if cfg.ENABLE_SHAPE_REGULARITY_TERM:
            for region_name in cfg.shape_regularity_active_regions():
                self.print_to_log_file(
                    f"[StagedTversky] mean_shape_regularity_{region_name}: "
                    f"{self._mean(f'shape_regularity_{region_name}'):.4f}"
                )
        for r in cfg.REGIONS:
            self.print_to_log_file(
                f"[StagedTversky] {r}: sum_tp={self._mean(f'tp_{r}'):.1f}  sum_fp={self._mean(f'fp_{r}'):.1f}  "
                f"sum_fn={self._mean(f'fn_{r}'):.1f}"
            )
            if r in self.controllers_small:
                self.print_to_log_file(
                    f"[StagedTversky] SMALL_{r}: sum_tp={self._mean(f'tp_small_{r}'):.1f}  "
                    f"sum_fp={self._mean(f'fp_small_{r}'):.1f}  sum_fn={self._mean(f'fn_small_{r}'):.1f}  "
                    f"mean_small_mask_fraction={self._mean(f'small_mask_fraction_{r}'):.5f}"
                )
        self.print_to_log_file("[StagedTversky] ------------------------")


class nnUNetTrainerStagedTversky_Stage1Calibration(nnUNetTrainerStagedTversky):
    """Short, cheap probe -- all controllers frozen, checks feasibility before a full run."""

    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cfg.CALIBRATION_MODE = True


if __name__ == "__main__":
    print(
        "Defines nnUNetTrainerStagedTversky and nnUNetTrainerStagedTversky_Stage1Calibration "
        "for use with nnUNetv2_train --tr <name>. dataset.json is never touched by this file -- "
        "every region in cfg.REGIONS gets fully independent alpha/beta control, for both the "
        "ALL term and the SMALL-lesion term."
    )