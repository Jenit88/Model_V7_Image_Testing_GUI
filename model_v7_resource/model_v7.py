"""Model_v7: scratch instance segmentation for 4-class PCB features.

This single file deliberately owns every stage that must agree:

0. ``selftest`` - assert the decode, target, ranking and augmentation
   contracts on synthetic data. Needs no dataset, GPU or checkpoint.
1. ``prepare``  - create leakage-safe, letterboxed train/val/test arrays.
2. ``preview``  - verify prepared labels and online augmentation visually.
3. ``train``    - train the four-head model from random initialization.
4. ``transfer`` - selectively transplant compatible Model_v5 encoder blocks,
   warm up the new V7 paths, progressively unfreeze the transplanted blocks,
   and then fine-tune the complete model.
5. ``fine_tune``- continue a trained V7 checkpoint with a fresh optimizer.
6. ``predict``  - run the same letterbox transform used during preparation.
7. ``evaluate`` - report semantic metrics, instance P/R/F1 at IoU 0.50,
   and confidence-ranked mask mAP50/mAP50-95.

The model predicts:

* a full-resolution five-class semantic map;
* class-specific centre heatmaps at half resolution;
* half-resolution pixel-to-centre offset vectors;
* a full-resolution instance-boundary map.

Model design and augmentation contract
--------------------------------------

* Preparation and prediction call the exact same ``letterbox_rgb`` function.
* The 95-mode Model_v5 augmentation cycle is retained exactly: one original,
  45 exact rotations, and 49 realistic variants. Its length is load-bearing
  for early stopping, the epoch schedule and the configuration validator, so
  every augmentation added since is a separate, orthogonal RNG stream.
* Model_v5 zoom-to-fill rotation, translation, brightness, contrast, gamma,
  exposure, blur/sharpen, camera noise, and colour operations are unchanged.
* Added since Model_v5, because both were costing real accuracy:
  - a dihedral stream (8 exact flips/quarter-turns). Every class here is
    symmetric under the full dihedral group, so all eight are lossless.
  - a zoom-out branch. V5 zoom-to-fill is always >= 1.0, so the model only
    ever saw objects larger than labelled while inference never zooms at all.
* Original and augmented centre/offset targets use one target generator, and
  the dihedral transform is applied before it, so the offset vector field is
  regenerated rather than rotated.
* Centre/offset prediction is performed at 1/2 resolution and centre logits
  start near 1% probability, preventing the sparse centre loss from dominating.
* The instance decoder actually uses the predicted boundary map.
* Decoded instances are ranked by their centre-peak score, not by mean
  semantic probability. The semantic term saturates near 1.0 for a real
  object and a fragment alike, so on its own it cannot order detections and
  mask AP collapses to a flat precision-recall line.
* Components far smaller than the largest one in the same candidate are rim
  fragments, not objects, and are rejected before they become detections.
* Separate best-semantic and best-instance checkpoints are saved. The
  instance checkpoint and the final candidate comparison both select on
  INSTANCE_SELECTION_METRIC, which defaults to the reported mAP50-95:
  instance F1 at IoU 0.50 cannot tell a barely-passing mask from an exact one.
* Final model selection compares saved candidates on the complete validation
  split, then reports untouched test performance separately.

Scratch-training guarantee
--------------------------

The scratch-training path creates a new model and never imports an application
model or loads pretrained weights. The transfer path can load only the explicit
local Model_v5 checkpoint configured below. It copies only audited layer pairs
whose complete weight lists have identical shapes; incompatible low-level
features and all four V7 heads retain their native initialization. No ImageNet,
Keras Applications, COCO weights, or downloaded weights are introduced.
BackupAndRestore may resume the same interrupted run.

Recommended order
-----------------

Run ``RUN_MODE = "selftest"`` first; it takes seconds and fails loudly if the
decoder cannot reproduce its own ground-truth targets.

Set ``RUN_MODE = "prepare"`` and run once if the V6 letterboxed arrays do not
already exist. Preparation now reports overlapping polygons: a later polygon
overwrites an earlier one, so a non-zero count means some targets are holed
or missing and no metric computed against them means what it says.

The augmentation distribution changed, so a run started under the previous
version cannot be resumed into. Point MODEL_OUTPUT_DIR at a fresh directory. Inspect the preview. Use ``RUN_MODE = "transfer"`` for the
audited Model_v5 -> Model_v7 run, or ``RUN_MODE = "train"`` for a random
initialization baseline. Prediction and evaluation use the checkpoint path you
select below. To refine an existing complete V7 checkpoint, configure
``FINE_TUNE_SOURCE_MODEL`` and use ``RUN_MODE = "fine_tune"``.
"""

from __future__ import annotations

import csv
import gc
import inspect
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from numpy.lib.format import open_memmap
from tensorflow.keras import Model, layers




# =============================================================================
# 1. CONFIGURATION
# =============================================================================

IMG_SIZE = 512
INSTANCE_HEAD_SIZE = IMG_SIZE // 2
NUM_CLASSES = 5
INSTANCE_ID_DTYPE = np.uint16
MAX_INSTANCE_ID = int(np.iinfo(INSTANCE_ID_DTYPE).max)

CLASS_NAMES = {
    0: "background",
    1: "Rectangle",
    2: "Rectangle_concave",
    3: "circle",
    4: "circle_full",
}

CLASS_COLOURS_RGB = np.asarray(
    [
        [0, 0, 0],
        [230, 65, 65],
        [255, 165, 45],
        [60, 180, 90],
        [65, 135, 230],
    ],
    dtype=np.uint8,
)

# The user's YOLO polygon files contain foreground IDs 1, 2, 3, and 4.
YOLO_TO_SEMANTIC_ID = {1: 1, 2: 2, 3: 3, 4: 4}

# Overridable so the same file runs on the training box and anywhere the
# dataset has been copied, without editing paths back and forth.
DATASET_ROOT = Path(
    os.environ.get(
        "PCB_DATASET_ROOT",
        "/home/u117134c/Data_preprocessing/45000 images/Split_Data",
    )
)

# These are the corrected V6 letterboxed arrays, not the old stretched arrays.
# Overridable separately from DATASET_ROOT: the source images may sit on a slow
# or read-only mount (a Windows drive under /mnt/c, a network share) while the
# ~10 GB of memmapped arrays that every training step reads must live on fast
# local storage.
ARRAY_DIR = Path(
    os.environ.get(
        "PCB_ARRAY_DIR",
        str(DATASET_ROOT / "instance_npy_512_letterbox_v6"),
    )
)
MODEL_OUTPUT_DIR = Path(
    os.environ.get(
        "PCB_MODEL_OUTPUT_DIR",
        "/home/u117134c/Models/Model_v7_v5_augmentation",
    )
)

# prepare | preview | train | transfer | fine_tune | predict | evaluate
RUN_MODE = "transfer"

PREDICT_SOURCE = DATASET_ROOT / "images" / "test"
MODEL_FOR_INFERENCE = Path(
    "/home/u117134c/Models/Model_v7_from_v5_transfer/"
    "best_transfer_model_v7_instance.keras"
)

# Image preparation. Value 114 is the same neutral padding used at prediction.
LETTERBOX_FILL_VALUE = 114

# Thresholds `rasterize_instances` uses to resolve overlapping label polygons.
# Measured from the labels themselves (classify_label_overlap.py): overlapping
# pairs separate cleanly into near-identical (IoU >= 0.90, the same object
# annotated twice) and one-inside-another (>= 0.90 of the smaller polygon lies
# within the larger). Everything between those is edge bleed a few pixels wide.
LABEL_DUPLICATE_IOU = 0.90
LABEL_CONTAINMENT = 0.90
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PREPARE_SPLITS = ("train", "val", "test")

# Training on the RTX PRO 1000 8 GB.
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 300
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 50
USE_MIXED_PRECISION = True
USE_EMA = True
EMA_MOMENTUM = 0.999
REQUIRE_GPU = True
SEED = 42

# Exact Model_v5 online augmentation. Do not change this block independently.
USE_ONLINE_AUGMENTATION = True
AUGMENTATION_CYCLE_LENGTH = 95
MIN_ROTATION_DEGREES = 2
MAX_ROTATION_DEGREES = 90
ROTATION_STEP_DEGREES = 2
ROTATE_BOTH_DIRECTIONS = True
EXACT_ROTATION_ANGLES = tuple(
    range(
        MIN_ROTATION_DEGREES,
        MAX_ROTATION_DEGREES + 1,
        ROTATION_STEP_DEGREES,
    )
)
EXACT_ROTATION_MODE_COUNT = len(EXACT_ROTATION_ANGLES)
REALISTIC_VARIANT_MODE_COUNT = 49
REALISTIC_VARIANT_MODE_OFFSET = EXACT_ROTATION_MODE_COUNT

# Early stopping cannot occur before every image has completed one full cycle.
EARLY_STOPPING_START_EPOCH = AUGMENTATION_CYCLE_LENGTH

# Fine-tuning starts from a complete trained V7 model (the transfer-selected
# model by default). It uses a fresh low-rate optimizer. The source checkpoint
# is also included in final candidate comparison, so a degraded fine-tuned
# epoch can never become authoritative merely because it was trained last.
FINE_TUNE_SOURCE_MODEL = MODEL_FOR_INFERENCE
FINE_TUNE_OUTPUT_DIR = Path(
    "/home/u117134c/Models/Model_v7_fine_tuned"
)
# Keep this equal to ARRAY_DIR to refine on the original prepared data, or point
# it at another V6-compatible prepared dataset for domain adaptation.
FINE_TUNE_ARRAY_DIR = ARRAY_DIR
FINE_TUNE_EPOCHS = 160
FINE_TUNE_LEARNING_RATE = 3e-5
FINE_TUNE_MIN_LEARNING_RATE = 3e-7
FINE_TUNE_WEIGHT_DECAY = 5e-5
FINE_TUNE_WARMUP_EPOCHS = 2
# Fine-tuning is selected and stopped by the actual instance objective. A value
# of zero evaluates every validation image rather than a convenient subset.
FINE_TUNE_INSTANCE_CHECKPOINT_EVERY_N_EPOCHS = 5
FINE_TUNE_INSTANCE_CHECKPOINT_MAX_IMAGES = 0
FINE_TUNE_INSTANCE_EARLY_STOPPING_PATIENCE_EVALUATIONS = 8
FINE_TUNE_INSTANCE_EARLY_STOPPING_START_EPOCH = AUGMENTATION_CYCLE_LENGTH
FINE_TUNED_MODEL_FOR_INFERENCE = (
    FINE_TUNE_OUTPUT_DIR / "best_fine_tuned_model_v7_instance.keras"
)

# Selective Model_v5 -> Model_v7 transfer learning. The V5 checkpoint is
# opened read-only and is never overwritten. V5 used full-resolution instance
# heads and stretched inputs, while V7 uses half-resolution instance heads
# and letterboxed inputs, so the four task heads and incompatible early encoder
# layers must NOT be copied. Only the audited homologous p4/p5/SPPF/attention
# layers are eligible, and every complete layer weight list must match exactly.
TRANSFER_SOURCE_MODEL = Path(
    "/home/u117134c/Models/Model_v5_instance/best_model_v5_instance.keras"
)
TRANSFER_OUTPUT_DIR = Path(
    "/home/u117134c/Models/Model_v7_from_v5_transfer"
)
TRANSFER_ARRAY_DIR = ARRAY_DIR
TRANSFER_EPOCHS = 220
TRANSFER_WEIGHT_DECAY = 5e-5

# Discriminative three-stage optimization. New V7 layers always receive the
# base learning rate. Transferred layers receive a per-group gradient
# multiplier: 0 while logically frozen, then a smooth ramp to 1.0. This keeps
# every optimizer variable registered from the beginning, so EMA, gradient
# accumulation, and crash recovery remain valid across the unfreezing stages.
TRANSFER_NEW_LAYER_LEARNING_RATE = 3e-4
TRANSFER_ADAPTATION_LEARNING_RATE = 1e-4
TRANSFER_P4_UNFREEZE_LEARNING_RATE = 5e-5
TRANSFER_FINE_TUNE_LEARNING_RATE = 3e-5
TRANSFER_MIN_LEARNING_RATE = 3e-7
TRANSFER_LR_WARMUP_EPOCHS = 2
TRANSFER_HEAD_WARMUP_EPOCHS = 8
TRANSFER_DEEP_UNFREEZE_EPOCH = 8
TRANSFER_P5_UNFREEZE_EPOCH = 16
TRANSFER_P4_UNFREEZE_EPOCH = 24
TRANSFER_FULL_FINE_TUNE_EPOCH = 40
TRANSFER_INSTANCE_CHECKPOINT_EVERY_N_EPOCHS = 5
TRANSFER_INSTANCE_CHECKPOINT_MAX_IMAGES = 0
TRANSFER_INSTANCE_EARLY_STOPPING_PATIENCE_EVALUATIONS = 10
TRANSFER_INSTANCE_EARLY_STOPPING_START_EPOCH = AUGMENTATION_CYCLE_LENGTH
TRANSFERRED_MODEL_FOR_INFERENCE = (
    TRANSFER_OUTPUT_DIR / "best_transfer_model_v7_instance.keras"
)

# (source-prefix, target-prefix, progressive-unfreeze group). Prefixes are
# deliberately explicit: this is an architectural transplant, not a broad
# "same name and shape" sweep that could copy semantically unrelated layers.
TRANSFER_LAYER_PREFIX_RULES = (
    ("p4_", "backbone_p4_", "p4"),
    ("backbone_p5_down_", "backbone_p5_down_", "p5"),
    ("p5_", "backbone_p5_", "p5"),
    ("sppf_", "sppf_", "deep"),
    ("deep_attention_", "deep_attention_", "deep"),
)
TRANSFER_GROUP_NAMES = ("deep", "p5", "p4")
TRANSFER_REQUIRED_TARGET_LAYERS = (
    "backbone_p4_branch1_conv",
    "backbone_p4_out_conv",
    "backbone_p5_down_conv",
    "backbone_p5_branch1_conv",
    "backbone_p5_out_conv",
    "sppf_reduce_conv",
    "sppf_out_conv",
    "deep_attention_dense1",
    "deep_attention_dense2",
)
TRANSFER_EXPECTED_WEIGHTED_LAYER_COUNT = 40
TRANSFER_EXPECTED_PARAMETER_COUNT = 3_936_048
TRANSFER_EXPECTED_GROUP_PARAMETER_COUNTS = {
    "deep": 407_088,
    "p5": 2_510_592,
    "p4": 1_018_368,
}

# Dihedral (flip / 90-degree rotation) augmentation. Rectangles, concave
# rectangles, circles and annuli are symmetric under the full dihedral group,
# so all eight transforms are exactly label-preserving here. This is applied
# as a separate RNG stream AFTER the 95-mode V5 cycle, which is left intact
# because its length is load-bearing for early stopping and the validator.
# Drop the mirrored half the day an orientation-sensitive output is added.
USE_DIHEDRAL_AUGMENTATION = True

# Exact V5 zoom-to-fill geometry and photometric ranges.
ZOOM_SAFETY_FACTOR = 1.002
EXTRA_ZOOM_RANGE = (1.06, 1.12)
# V5 zoom-to-fill is always >= 1.0, so training never saw an object smaller
# than labelled while inference is never zoomed at all. This branch restores
# the missing half of the scale range and pads with the letterbox value.
#
# The range is set from the measured split statistics, not by taste. Median
# equivalent object diameter, train -> test:
#     Rectangle    35.7 -> 21.9 px  (0.61x)
#     circle       33.8 -> 16.1 px  (0.48x)
#     circle_full  21.0 -> 16.0 px  (0.76x)
# Test objects are roughly half to three-quarters the linear size of training
# objects, so without this branch the model never sees test-scale objects at
# all. The lower bound covers the worst case with margin; at 0.45 a median
# circle_full is still ~5 px across at the instance-head resolution, which
# the centre gaussian (sigma floor 1.5) can still represent.
#
# The zoom-out amount is drawn from a discrete ladder rather than a continuous
# range: 5% steps from 5% out to 50% out, so every decade of the scale range
# gets the same number of training samples instead of the uniform draw
# over-weighting whatever the interval happens to be.
ZOOM_OUT_STEP_PERCENT = 5
ZOOM_OUT_MAX_PERCENT = 50
ZOOM_OUT_STEPS = tuple(
    round(1.0 - percent / 100.0, 4)
    for percent in range(
        ZOOM_OUT_STEP_PERCENT,
        ZOOM_OUT_MAX_PERCENT + 1,
        ZOOM_OUT_STEP_PERCENT,
    )
)
ZOOM_OUT_RANGE = (min(ZOOM_OUT_STEPS), 1.00)  # reported/validated bound
ZOOM_OUT_PROBABILITY = 0.50

# The other half of the scale range, added for the mirror-image defect and set
# from measurement in the same way.
#
# `EXTRA_ZOOM_RANGE` reaches only variant modes 30-35 and 48-49 -- 8 of 95 --
# and stops at 1.12x, so with zoom-to-fill the widest object training ever
# produced was about 1.7x labelled scale. Measured on 16 real boards
# (measure_falt_scale.py), against the training medians:
#
#     class          detected p50   max    train p50   beyond 1.7x ceiling
#     Rectangle          47.9 px   140.8      35.7            30.1%
#     circle_full        26.0 px    70.1      21.0            10.1%
#     circle             32.9 px    68.7      33.8             4.8%
#
# That last column orders the classes exactly as the failure overlays do:
# Rectangles shattered into strips, circles stayed intact. 2.5x carries the
# ceiling to ~89 px for Rectangle and covers the p90 of what real boards
# contain; the 140 px tail is left to the watershed grouping, because no
# amount of augmentation fixes an offset field asked to span half a frame.
#
# Same discrete-ladder reasoning as zoom-out: even coverage per decade of
# scale rather than a uniform draw that over-weights the interval's top end.
ZOOM_IN_STEP_PERCENT = 15
ZOOM_IN_MAX_PERCENT = 150
ZOOM_IN_STEPS = tuple(
    round(1.0 + percent / 100.0, 4)
    for percent in range(
        ZOOM_IN_STEP_PERCENT,
        ZOOM_IN_MAX_PERCENT + 1,
        ZOOM_IN_STEP_PERCENT,
    )
)
ZOOM_IN_RANGE = (1.00, max(ZOOM_IN_STEPS))  # reported/validated bound
# Lower than zoom-out's 0.50: the zoom-out branch corrects a shift the whole
# test split exhibits, while this one covers a tail. Zoom-in also crops, so a
# high probability would spend a large share of every epoch on partial boards.
# Off in V7, on evidence from both directions.
#
# *The reason for it is gone.* The ladder existed because the offset head had to
# emit a vector proportional to object size, and 30.1% of real-board Rectangles
# sat beyond the range training produced. A normalised inner distance is bounded
# in [0,1] for every object at every scale, so that defect cannot occur here.
#
# *And it was measured to cost.* The V6.2 Tier-1 run is this ladder, and nothing
# else, against the 05-09 baseline. Over 15 shared evaluations it took mask
# mAP50-95 down 0.003-0.013 while taking precision, F1 and mAP50 *up*, and
# semantic foreground mIoU down 0.0048 -- better at finding objects, worse at
# outlining them. Per class the benefit was monotone in object size (+0.0235 F1
# on the ~108 px class, +0.0001 on the ~16 px ones). Zoom-in upsamples: it
# cannot create detail, and a nearest-neighbour-resampled label boundary is a
# staircase, so 15% of every epoch trained the boundary on manufactured edges.
#
# Zoom-*out* keeps its p=0.50 -- it crops nothing and invents nothing, and the
# audit shows the test split is more zoomed-out than train, not less.
# Set this back above zero only alongside a measurement that says it earns it.
ZOOM_IN_PROBABILITY = 0.00
# The area a zoom-out uncovers is filled with black, not with the 114 grey used
# for the letterbox bars. Black is unambiguous: 114 also occurs inside real
# board images, so padding with it would teach the model that a mid-grey region
# is sometimes background and sometimes not. The two fills stay independent
# for that reason; set this to LETTERBOX_FILL_VALUE to match the bars instead.
ZOOM_OUT_FILL_VALUE = 0
# Mirror-side augmentation (mirror_side_sample). The board is replaced by its
# mirror image -- the original is not kept, nothing is attached, nothing is
# zoomed. "left" mirrors across a vertical edge (left-right flip) and "top"
# across a horizontal edge (top-bottom flip); right and bottom are left out
# because they would repeat those same two images. Only the real board is
# flipped, in place inside its letterbox bars. Labels are flipped with the
# pixels and keep their ids. The side follows the epoch: every mirrored sample
# in an epoch uses the same side, alternating left -> top. Whether a sample
# is mirrored at all is drawn from its own RNG stream (salt 4), so no existing
# augmentation draw shifts. Runs before the 95-mode cycle, so it combines with
# every mode. Set the probability to 1.0 to mirror every sample.
USE_MIRROR_SIDE_AUGMENTATION = True
MIRROR_SIDE_PROBABILITY = 0.25
MIRROR_SIDES = ("left", "top")
MAX_TRANSLATION_FRACTION = 0.04
BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10
GAMMA_RANGE = (0.90, 1.10)
EXPOSURE_GAIN_RANGE = (0.95, 1.05)
NOISE_STD_RANGE = (0.005, 0.020)
HUE_SHIFT_DEGREES = 3.0
SATURATION_GAIN_RANGE = (0.92, 1.08)
MIN_AUGMENTED_INSTANCE_AREA = max(
    4, int(round(0.000015 * IMG_SIZE * IMG_SIZE))
)

# Copy-paste augmentation (Ghiasi et al., 2021). Per-pixel instance ids are
# already stored, so extraction is free. Two measured problems make this the
# highest-value augmentation available here:
#   * Rectangle_concave has 680 training polygons (0.47% of all polygons) and
#     is the worst class by a wide margin. Donors are drawn by picking a class
#     uniformly first, which lifts its exposure from 0.47% to ~25%.
#   * V5 epoch 356 had train centre loss 0.032 against val 1.032 - the centre
#     head memorises. Pasting objects into new contexts breaks that.
# The paste alpha is feathered so a hard cut never becomes the boundary cue.
USE_COPY_PASTE_AUGMENTATION = True
# Mosaic: four samples downscaled into one 512 frame, 2x2.
#
# It is the only lever that closes the density gap, and the gap is large and
# measured. Objects per image and median diameter, this pipeline versus the
# split it is scored on:
#
#                            obj/img   Rectangle   circle   circle_full
#   train, no augmentation      17.0        37.2     33.5          21.3
#   train, augmented            14.5        41.1     43.5          26.2
#   TEST                        75.0        22.1     16.0          15.9
#
# The augmented row moves *away* from test on both axes -- fewer, larger --
# because V5 zoom-to-fill enlarges on every rotation. Zoom cannot fix the count:
# zooming out shrinks the objects in a frame but never adds any, and copy-paste
# adds at most COPY_PASTE_MAX_OBJECTS. A 2x2 mosaic multiplies count by four and
# halves diameters in one operation, which lands close to the test regime.
#
# **Default 0.0, deliberately.** The V6.2 Tier-1 run was an augmentation change
# argued from exactly this kind of distribution reasoning, and it measurably cost
# mask precision: any resampling trades image fidelity for coverage, and mosaic
# downscales by 2. Unlike zoom-in it discards detail rather than inventing it --
# the direction test objects genuinely occupy -- so the risk is lower, not
# absent. Turn it on for a short A/B against 0.0 before adopting it in a full
# run. MOSAIC_PROBABILITY = 0.25 is the value to try first.
MOSAIC_PROBABILITY = 0.00
USE_MOSAIC_AUGMENTATION = True

COPY_PASTE_PROBABILITY = 0.50
COPY_PASTE_MAX_OBJECTS = 3
COPY_PASTE_EDGE_FEATHER_PX = 1.5

# One consistent target definition for original and augmented samples.
CENTER_SIGMA_RANGE = (1.5, 5.0)
MIN_TARGET_INSTANCE_AREA_FULL_RES = MIN_AUGMENTED_INSTANCE_AREA

# Lovasz sorts every pixel of every present class in-graph. A uniform sample
# is statistically equivalent and removes four full 512x512 sorts per step.
LOVASZ_MAX_PIXELS = 65536

# Device for that sort. Measured on an Apple M3 via tensorflow-metal:
#
#     top_k / sort / argsort, 65,536 elements   GPU 636 ms   CPU 7 ms
#     top_k / sort / argsort, 262,144 elements  GPU 5,870 ms CPU 32 ms
#     same sort pinned to /CPU:0 inside a GPU graph          9 ms
#
# Metal has no usable sort kernel, and one bad op made the whole GPU 2.5x
# slower than the CPU even though it is ~10x faster at every convolution in
# this model. Pinning only the sort moves 256 KB per class and leaves
# everything else on the GPU. CUDA sorts perfectly well, so this must stay
# None there; set it to "/CPU:0" only on Metal.
LOVASZ_SORT_DEVICE = None

# cv2/numpy release the GIL, so threads avoid pickling the memmaps.
DATA_LOADER_WORKERS = 8
DATA_LOADER_QUEUE_SIZE = 16
# One-off dataset preparation is pure PNG decode plus polygon rasterisation,
# both of which release the GIL.
PREPARE_WORKERS = max(1, min(16, (os.cpu_count() or 4)))

# Per-class IoU as a live Keras metric costs one extra full-resolution argmax
# and one float64 confusion matrix per class per step, for numbers nothing
# consumes. Set True to watch them per epoch at that price.
PER_CLASS_IOU_METRICS_DURING_TRAINING = False

# Number of batches TensorFlow executes per Python callback round trip. With a
# small batch the per-step Python/dispatch overhead is a real fraction of the
# step, and this amortises it. Progress-bar granularity coarsens to this value.
STEPS_PER_EXECUTION = 8

# XLA. Off until a benchmark on this machine shows it both compiles and wins.
USE_XLA_JIT = False

# Training steps between TensorBoard scalar writes. 0 means epoch-level only,
# which on this machine is one data point every ~35 minutes.
#
# This is in real steps, and the Keras callback is NOT: its update_freq counts
# its own invocations, and with steps_per_execution=N it is invoked once per
# group of N steps (keras/src/callbacks/tensorboard.py: on_train_batch_begin
# increments _global_train_batch by one per call, and should_record tests that
# counter against update_freq). Passing 96 directly therefore writes every
# 96*8 = 768 steps, not every 96. compile_model divides it out.
TENSORBOARD_UPDATE_STEPS = 96

# Balanced losses. The centre head has fewer negative locations and lower weight.
SEMANTIC_LOSS_WEIGHT = 1.00
CENTER_LOSS_WEIGHT = 0.25          # retained for reading V7 configs
OFFSET_LOSS_WEIGHT = 1.00          # retained for reading V7 configs
# One head replaces two, so it carries the weight both used to share.
# Kept at the offset head's value rather than their sum: the centre head
# was weighted 0.25 because sparse Gaussian peaks are an easy target that
# would otherwise dominate, and the inner-distance field is dense in the
# way the offset field was.
# Measured, not inherited. 1.00 was carried over from OFFSET_LOSS_WEIGHT, but
# the offset loss was only 2.3% of V6.2's loss budget -- the *centre* head at
# 26.8% carried instance-geometry learning. Deleting the centre head and keeping
# offset's weight silently cut the one head that produces instance separation
# from 29.1% of the budget to 3.8%.
#
# Measured at epoch 73 of the Tier-1 run (the arithmetic reconstructs the logged
# total 0.17761 exactly, so these shares are exact, not estimated):
#
#     semantic  0.0751 x 1.00 = 0.0751   42.3%
#     centre    0.1907 x 0.25 = 0.0477   26.8%   } 29.1% instance geometry
#     offset    0.0040 x 1.00 = 0.0040    2.3%   }
#     boundary  0.1694 x 0.30 = 0.0508   28.6%
#
# MaskedInnerDistanceLoss on real targets: 0.024 at initialisation, 0.005 at a
# mid-training error of 0.10, 0.0016 at 0.05, 0 when fitted. A Huber on a [0,1]
# quantity with delta 0.10 is simply an order of magnitude smaller than the
# losses beside it. At weight 10 it contributes 0.050 at mid-training -- 28.4%
# of the budget, matching the 29.1% the heads it replaces held.
#
# This is one constant, and the single most likely thing to need revisiting once
# a run exists: the value is calibrated at the mid-training operating point, and
# a dense regression loss decays as it fits while cross-entropy plateaus, so the
# share is not constant across a run.
INNER_DISTANCE_LOSS_WEIGHT = 10.00
BOUNDARY_LOSS_WEIGHT = 0.30
BOUNDARY_POSITIVE_WEIGHT = 4.0
BOUNDARY_DILATION_ITERATIONS = 1

# Decoder parameters. Tune only on validation data, never on the test set.
SEMANTIC_CONFIDENCE_THRESHOLD = 0.30
CENTER_CONFIDENCE_THRESHOLD = 0.10
# Instance-head pixels, not full-resolution pixels. One head pixel spans
# IMG_SIZE / INSTANCE_HEAD_SIZE = 2 full-resolution pixels, so 2 here is the
# 4-full-resolution-pixel radius that Model_v5 was tuned to.
CENTER_NMS_RADIUS = 2
# Outlier filter on the offset residual: the distance from a pixel's PROJECTED
# position (pixel + predicted offset) to its nearest centre. It is not an
# object-size limit; a correct offset lands on its centre whatever the size.
MAX_CENTER_ASSIGNMENT_DISTANCE = int(round(0.10 * IMG_SIZE))
MIN_INSTANCE_AREA = max(12, int(round(0.00005 * IMG_SIZE * IMG_SIZE)))
# One decoded candidate may legitimately split into several touching objects,
# but a component far smaller than the largest one in the same candidate is a
# rim fragment, not an object. Rejecting those removes false positives that
# would otherwise rank as highly as real detections.
MIN_COMPONENT_AREA_FRACTION = 0.10
MIN_BOUNDARY_CORE_AREA = 6
BOUNDARY_CONFIDENCE_THRESHOLD = 0.50
MAX_CENTERS_PER_CLASS = 150
ASSIGNMENT_CHUNK_SIZE = 65536
# Instances recovered without a centre peak are real detections but carry no
# centre evidence, so they must rank below every centre-backed instance.
FALLBACK_INSTANCE_SCORE = 0.05

# --- Tiled inference (V7, Tier 3) --------------------------------------
#
# A 1536x1024 board letterboxed into one 512x512 frame wastes 33.4% of the
# input on grey bars and shrinks every object by 1.5x before the model sees
# it. Measured on real boards, that pushes 30.1% of Rectangles past anything
# training produced and leaves a test-split `circle` as a ring ~1.5 px thick,
# which is where AP50-95 0.5630 comes from.
#
# Tiling removes both at once: each tile covers a 512x512 region of the source
# at native resolution, so objects arrive ~3x larger and no pixel is padding.
# The alternative -- a non-square 768x512 input -- needs IMG_SIZE split into
# height and width across ~380 references, with a numpy-(H,W)-versus-cv2-(W,H)
# transposition hazard at every paired site. Tiling changes no shape at all.
TILE_INFERENCE_ENABLED = True

# Overlap between neighbouring tiles, in source pixels.
#
# Sized from the objects, not by taste: the largest object measured on real
# boards is ~141 px in the letterboxed frame, which is ~423 px at source scale.
# An object smaller than the overlap is always wholly inside at least one tile,
# so it is never split across a seam and never has its per-instance distance
# normalised against a partial view of itself. 192 px covers the p99 of what
# these boards contain while keeping the tile count low.
TILE_OVERLAP = 192

# Cosine taper applied when blending overlapping tiles, in source pixels. A
# hard join leaves a visible discontinuity in the stitched field, and the
# decoder reads a discontinuity as a boundary -- so a seam would manufacture
# instance splits exactly where two tiles meet.
TILE_BLEND_MARGIN = 64


# --- Inner-distance grouping (V7) -------------------------------------
#
# Where a pixel counts as an object's core rather than its rim, on the
# per-instance normalised inner distance (0 at the boundary, 1 at the
# innermost pixel of every object at every scale).
#
# 0.5 is the midpoint of a normalised field, which makes it scale-free in the
# same way the field is: it is "the inner half of the object" whether that
# object is 16 px or 130 px across. Two objects merge only if the predicted
# field fails to dip below half between them, which for touching pads means
# the boundary evidence would have to be wrong across the entire join rather
# than merely weak at one pixel.
#
# Lower splits more (rim noise becomes its own core); higher merges more
# (a slightly flattened prediction loses the dip between neighbours).
INNER_DISTANCE_CORE_THRESHOLD = 0.50

# Gaussian sigma applied to the predicted field before cores are extracted.
#
# Zero, and that is a measured result rather than a default. Smoothing looks
# right -- a true inner-distance field is a distance transform and has no high
# frequencies -- but it destroys exactly the class this work exists to fix. A
# 3 px annulus has a maximum inner distance of ~1.5 px, so its core is a
# one-pixel-wide centreline; any sigma comparable to that width breaks the ring
# into arcs. Swept against both requirements (sweep_tier2_params.py):
#
#     sigma   min core   clean failures   noisy count error
#      0.0          8                0                   3
#      0.5          4                0                   3
#      1.0          4                1                   1
#
# Sigma 1.0 buys two units of noise robustness and costs the annulus every
# time. The area floor below reaches almost the same robustness for free.
INNER_DISTANCE_CORE_SMOOTHING = 0.0

# Smallest connected core, in head pixels, that may seed an instance.
#
# This is what actually suppresses speckle, and it does so without a length
# scale that thin structures can fall below: noise cores are a few pixels,
# while a one-pixel-wide ring of radius 26 still has ~163 core pixels. Area
# separates them where blur cannot. Must be applied to the cores rather than
# to the finished instances -- after expansion every speck has grown into a
# plausible-sized object.
MIN_INNER_DISTANCE_CORE_AREA = 8

# Radius, in head pixels, of the closing that rejoins a core split by noise.
# One pixel: enough to heal a single broken pixel in an annulus centreline,
# too small to bridge two real cores, which sit half an object apart. 0 off.
INNER_DISTANCE_CORE_CLOSING = 1

# Deployment-only confidence floor, applied by `predict` and nothing else.
#
# It is 0.0 here on purpose. Every published number -- `evaluate`, the mask AP
# reports, the YOLO comparison -- must stay reproducible from the shipped
# configuration, and a floor would silently change them by discarding the tail
# of the ranking that AP is defined over. The deployment entry points set it
# instead (see deployment.py), because there the tail is not ranked-and-ignored
# but drawn on the picture.
DEPLOYMENT_MIN_CONFIDENCE = 0.0

# What to do with foreground pixels whose predicted offset does not land near
# any centre. They are the residue of `nearest_center_assignments`: mostly the
# rims of real objects and specular highlights, not missed objects.
#
# "new_instances" was the original behaviour -- mint an instance from each
# connected component of the residue. Measured on 150 validation images
# (`measure_fallback.py`), that path returned **75 false positives against 2
# true positives**, a 37:1 ratio, and was the entire cause of
# Rectangle_concave's precision collapse on test (0.4414, with 124 FP against
# 98 TP): dropping it takes that class from 5 FP to 0 on validation.
#
# "drop" discards the residue. Overall validation precision 0.9262 -> 0.9386
# and F1 0.9493 -> 0.9557 for a cost of 2 true positives.
#
# This does not touch the other fallback, which fires only when the centre head
# finds no centre at all for a class in an image. That one is a genuine last
# resort and is rare.
UNASSIGNED_PIXEL_POLICY = "drop"  # "drop" | "new_instances"

# Dihedral test-time augmentation at evaluation. Every class here is symmetric
# under all eight flips and quarter-turns, so their predictions can be averaged.
# Costs 8x the forward pass and nothing else; the decode is unchanged. Off by
# default so the reported number is the deployed single-pass one unless a run
# deliberately asks for the ensemble. See tta.py.
USE_TTA_AT_EVALUATION = False
TTA_TRANSFORMS = tuple(range(8))
# How many of the eight transforms to push through the model at once. The same
# 8 GB that caps training at batch 2 caps this: a batch of 8 raises
# ResourceExhausted in the 512x512 stem even without a gradient tape.
TTA_BATCH_SIZE = 2

# Validation, checkpoints, previews, and final reporting.
INSTANCE_EVALUATION_IOU = 0.50
# Mask average precision follows the common YOLO/COCO IoU sweep. Predictions
# are ranked by their mean class-specific semantic probability and AP is
# integrated over 101 recall samples. The decoder thresholds above remain
# fixed, so these numbers measure the complete deployed decoding pipeline.
MASK_AP_IOU_THRESHOLDS = tuple(
    round(0.50 + 0.05 * index, 2)
    for index in range(10)
)
MASK_AP_RECALL_POINTS = 101
MASK_AP_MAX_DETECTIONS_PER_IMAGE = 300
# Select the checkpoint on the metric that is actually reported. Instance F1
# at IoU 0.50 is indifferent to mask tightness above 0.5, so selecting on it
# while reporting mAP50-95 optimises one quantity and grades another.
# One of "mask_map50_95", "mask_map50" or "f1".
INSTANCE_SELECTION_METRIC = "mask_map50_95"
# Every N epochs the running report is snapshotted so progress can be read
# milestone by milestone instead of only as a final number.
MILESTONE_ANALYSIS_EVERY_N_EPOCHS = 50
INSTANCE_CHECKPOINT_EVERY_N_EPOCHS = 5
INSTANCE_CHECKPOINT_MAX_IMAGES = 256  # fixed validation subset; 0 = all
PREVIEW_EVERY_N_EPOCHS = 10
# One 2x4 panel sheet is written per image, so this is how many validation
# images you can inspect per preview epoch rather than how many fit in one file.
PREVIEW_IMAGE_COUNT = 20
# Above this, the combined contact sheet is skipped: 20 images stacked would be
# 2048x20480, which is writable but not viewable.
PREVIEW_CONTACT_SHEET_MAX_IMAGES = 6
RUN_FINAL_EVALUATION_AFTER_TRAINING = True
FINAL_EVALUATION_SPLITS = ("val", "test")
EVALUATION_MAX_IMAGES = 0  # final evaluation: 0 means the complete split
# Used when RUN_MODE = "evaluate". Keep "val" for threshold development;
# use the untouched "test" split for the final YOLO comparison.
EVALUATION_SPLIT = "test"


# =============================================================================
# 2. SHARED IMAGE PREPROCESSING
# =============================================================================

def read_rgb_image(path: Path) -> np.ndarray:
    """Read an RGB image reliably, including paths containing spaces."""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return finite float32 RGB data in [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains NaN or infinity.")
    if image.size and float(image.max()) > 1.5:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def letterbox_rgb(
    image_rgb: np.ndarray,
    size: int = IMG_SIZE,
    fill_value: int = LETTERBOX_FILL_VALUE,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Resize without distortion and pad to a square canvas.

    This exact function is called by both dataset preparation and prediction.
    """
    original_height, original_width = image_rgb.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError(f"Invalid image shape: {image_rgb.shape}")

    scale = min(size / original_width, size / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = size - resized_width - left
    bottom = size - resized_height - top
    canvas = np.full((size, size, 3), fill_value, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    metadata: dict[str, int | float] = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "scale": scale,
    }
    return canvas, metadata


def restore_map_to_original(
    array: np.ndarray,
    metadata: dict[str, int | float],
    interpolation: int,
) -> np.ndarray:
    """Remove letterbox padding and resize a map to the original image."""
    left = int(metadata["left"])
    top = int(metadata["top"])
    width = int(metadata["resized_width"])
    height = int(metadata["resized_height"])
    cropped = array[top : top + height, left : left + width]
    if cropped.size == 0:
        raise ValueError(f"Invalid letterbox metadata: {metadata}")
    return cv2.resize(
        cropped,
        (int(metadata["original_width"]), int(metadata["original_height"])),
        interpolation=interpolation,
    )


# =============================================================================
# 3. DATASET PREPARATION FROM YOLO POLYGONS
# =============================================================================

def list_records(split: str) -> list[tuple[Path, Path]]:
    """Return matching image/label paths, preserving nested subdirectories."""
    image_root = DATASET_ROOT / "images" / split
    label_root = DATASET_ROOT / "labels" / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    image_paths = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_root}")

    records: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        relative = image_path.relative_to(image_root)
        records.append((image_path, label_root / relative.with_suffix(".txt")))
    return records


def parse_yolo_polygons(label_path: Path) -> list[tuple[int, np.ndarray]]:
    """Read a YOLO segmentation label as (semantic_id, normalized points)."""
    if not label_path.is_file():
        return []

    polygons: list[tuple[int, np.ndarray]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        values = raw_line.strip().split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2 != 0:
            raise ValueError(
                f"Invalid polygon at {label_path}, line {line_number}: "
                "expected a class followed by at least three x/y pairs."
            )
        try:
            raw_class_value = float(values[0])
            points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        except ValueError as error:
            raise ValueError(
                f"Non-numeric label at {label_path}, line {line_number}."
            ) from error
        if not raw_class_value.is_integer():
            raise ValueError(
                f"Non-integer class ID at {label_path}, line {line_number}."
            )
        raw_class_id = int(raw_class_value)
        if raw_class_id not in YOLO_TO_SEMANTIC_ID:
            raise ValueError(
                f"Unexpected class ID {raw_class_id} in {label_path}; "
                f"expected {sorted(YOLO_TO_SEMANTIC_ID)}."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(f"NaN or infinity in {label_path}, line {line_number}.")
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError(
                f"Coordinates outside [0, 1] in {label_path}, line {line_number}."
            )
        polygons.append((YOLO_TO_SEMANTIC_ID[raw_class_id], points))
    return polygons


def letterboxed_polygon_pixels(
    normalized_points: np.ndarray,
    metadata: dict[str, int | float],
) -> np.ndarray:
    """Map normalized source coordinates onto the letterboxed 512 grid."""
    points = np.asarray(normalized_points, dtype=np.float64).copy()
    points[:, 0] = (
        points[:, 0] * max(int(metadata["resized_width"]) - 1, 0)
        + int(metadata["left"])
    )
    points[:, 1] = (
        points[:, 1] * max(int(metadata["resized_height"]) - 1, 0)
        + int(metadata["top"])
    )
    points[:, 0] = np.clip(points[:, 0], 0, IMG_SIZE - 1)
    points[:, 1] = np.clip(points[:, 1], 0, IMG_SIZE - 1)
    return np.rint(points).astype(np.int32)


def rasterize_instances(
    label_path: Path,
    metadata: dict[str, int | float],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Rasterize semantic and separate instance IDs after letterboxing.

    Contested pixels are resolved explicitly rather than by file order.
    Previously each polygon was painted in the order it appeared in the label
    file and simply overwrote whatever was already there, which meant the
    annotation tool's serialisation order silently decided the training
    targets. Measured across the dataset that destroyed 104 training and 16
    test instances outright and left 292,656 / 15,164 pixels claimed twice.

    Classifying the overlaps first (`classify_label_overlap.py`) showed they
    are one phenomenon, not three: on test, 36 nested pairs account for 98.3%
    of the shared pixels, 78 "bleed" pairs share 258 pixels between them
    (3.3 each -- outline rounding), and there are no cross-class duplicates
    anywhere. Every nested pair is the same convention: a small `Rectangle`
    inside a `Rectangle_concave` at a very consistent 3.5-3.8% area ratio. The
    inner polygon is a real, separately annotated object, and the old rule
    erased it whenever the outer one happened to be written later.

    So the resolution order is:

    1. Same-class near-duplicates collapse into one instance. A duplicate pair
       that disagrees about the class is an annotation conflict rather than a
       rasterisation one; none exist today, and one appearing raises instead of
       being resolved by an arbitrary rule.
    2. Where one polygon is essentially contained in another, the SMALLER owns
       the shared region -- an inner feature keeps its own interior.
    3. Otherwise the pixel goes to whichever polygon it lies deepest inside,
       by distance transform.
    4. No polygon may finish with zero pixels: one that would be entirely
       consumed keeps its innermost pixel.

    The result does not depend on the order polygons appear in the file.
    ``overlapping_pixels`` still reports how many pixels were contested, so the
    labels can still be audited, but it is now a description of the input
    rather than a count of damage done to the output. ``destroyed_instances``
    is structurally zero and `prepare_split` treats any other value as fatal.
    """
    semantic = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    instance = np.zeros(
        (IMG_SIZE, IMG_SIZE), dtype=INSTANCE_ID_DTYPE
    )

    # Rasterise every polygon on its own first; nothing is committed until all
    # claims are known. Objects here are 16-36 px across, so each is rendered
    # inside its own bounding box rather than a full 512x512 canvas.
    entries: list[dict[str, object]] = []
    for semantic_id, normalized_points in parse_yolo_polygons(label_path):
        polygon = letterboxed_polygon_pixels(normalized_points, metadata)
        if len(polygon) < 3:
            continue
        left = int(polygon[:, 0].min())
        right = int(polygon[:, 0].max()) + 1
        top = int(polygon[:, 1].min())
        bottom = int(polygon[:, 1].max()) + 1
        if left >= right or top >= bottom:
            continue
        object_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillPoly(object_mask, [polygon - np.array([[left, top]], np.int32)],
                     color=1)
        if not object_mask.any():
            continue
        entries.append({
            "semantic_id": int(semantic_id),
            "box": (top, bottom, left, right),
            "mask": object_mask.astype(bool),
            "area": int(object_mask.sum()),
        })

    # Which kind of conflict each overlapping pair was. Reported per split by
    # `prepare_split`, because otherwise the only way to know whether a split's
    # overlaps are nesting or duplicates is to infer it from instance-count
    # arithmetic afterwards -- and an inference in a permanent record gets
    # quoted later as though it had been measured.
    conflict_counts = {"duplicate": 0, "nested": 0, "bleed": 0}

    if not entries:
        return semantic, instance, 0, 0, 0, conflict_counts

    def shared_pixels(a: dict, b: dict) -> int:
        at, ab, al, ar = a["box"]
        bt, bb, bl, br = b["box"]
        top, bottom = max(at, bt), min(ab, bb)
        left, right = max(al, bl), min(ar, br)
        if top >= bottom or left >= right:
            return 0
        return int(np.count_nonzero(
            a["mask"][top - at:bottom - at, left - al:right - al]
            & b["mask"][top - bt:bottom - bt, left - bl:right - bl]
        ))

    # 1. Collapse same-class near-duplicates, keeping the earlier polygon.
    dropped: set[int] = set()
    contained_in: dict[int, set[int]] = {i: set() for i in range(len(entries))}
    for i in range(len(entries)):
        if i in dropped:
            continue
        for j in range(i + 1, len(entries)):
            if j in dropped:
                continue
            shared = shared_pixels(entries[i], entries[j])
            if shared <= 0:
                continue
            area_i = int(entries[i]["area"])
            area_j = int(entries[j]["area"])
            union = area_i + area_j - shared
            if union > 0 and shared / union >= LABEL_DUPLICATE_IOU:
                if entries[i]["semantic_id"] != entries[j]["semantic_id"]:
                    raise ValueError(
                        f"{label_path} annotates one object twice with "
                        f"different classes ({entries[i]['semantic_id']} and "
                        f"{entries[j]['semantic_id']}, IoU "
                        f"{shared / union:.3f}). A rasterisation rule cannot "
                        "decide which is correct; fix the label."
                    )
                dropped.add(j)
                conflict_counts["duplicate"] += 1
                continue
            # 2. Record containment so an inner feature can win its interior.
            if shared / area_i >= LABEL_CONTAINMENT:
                contained_in[i].add(j)
            if shared / area_j >= LABEL_CONTAINMENT:
                contained_in[j].add(i)
            if (shared / area_i >= LABEL_CONTAINMENT
                    or shared / area_j >= LABEL_CONTAINMENT):
                conflict_counts["nested"] += 1
            else:
                conflict_counts["bleed"] += 1

    kept = [i for i in range(len(entries)) if i not in dropped]
    if len(kept) > MAX_INSTANCE_ID:
        raise OverflowError(
            f"{label_path} contains more than {MAX_INSTANCE_ID:,} "
            "non-empty polygons; uint16 instance IDs are exhausted."
        )

    # Count claims per pixel to find what is actually contested.
    claims = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int16)
    for index in kept:
        top, bottom, left, right = entries[index]["box"]
        claims[top:bottom, left:right] += entries[index]["mask"]
    overlapping_pixels = int(np.count_nonzero(claims > 1))

    # Depth inside each polygon, for rule 3 and for the survival guarantee.
    for index in kept:
        mask = entries[index]["mask"]
        entries[index]["depth"] = cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, 3
        )

    # Uncontested pixels can be written directly.
    for position, index in enumerate(kept, start=1):
        top, bottom, left, right = entries[index]["box"]
        mask = entries[index]["mask"] & (claims[top:bottom, left:right] == 1)
        if not mask.any():
            continue
        semantic[top:bottom, left:right][mask] = entries[index]["semantic_id"]
        instance[top:bottom, left:right][mask] = position

    # Contested pixels, one at a time. There are few of them -- nesting is rare
    # and bleed averages 3.3 pixels per pair -- so per-pixel resolution costs
    # little and keeps the rule readable.
    position_of = {index: position for position, index in enumerate(kept, start=1)}
    if overlapping_pixels:
        rows, columns = np.nonzero(claims > 1)
        for row, column in zip(rows.tolist(), columns.tolist()):
            claimants = [
                index for index in kept
                if entries[index]["box"][0] <= row < entries[index]["box"][1]
                and entries[index]["box"][2] <= column < entries[index]["box"][3]
                and entries[index]["mask"][row - entries[index]["box"][0],
                                           column - entries[index]["box"][2]]
            ]
            if not claimants:
                continue
            competitors = set(claimants)
            best_index = None
            best_key = None
            for index in claimants:
                box = entries[index]["box"]
                # Inner features first, then how deep the pixel sits inside
                # this polygon, then the smaller polygon. On today's labels the
                # first term decides every nested pair and the rest only order
                # bleed, where the contested run is a few pixels wide and any
                # deterministic rule yields the same mask -- but depth is the
                # right rule if that ever stops being true, and the transform
                # is already computed for the survival guarantee below.
                key = (
                    bool(contained_in[index] & competitors),
                    float(entries[index]["depth"][row - box[0], column - box[2]]),
                    -int(entries[index]["area"]),
                    -index,
                )
                if best_key is None or key > best_key:
                    best_key, best_index = key, index
            semantic[row, column] = entries[best_index]["semantic_id"]
            instance[row, column] = position_of[best_index]

    # 4. Nothing may be consumed entirely: give any lost polygon its own
    #    innermost pixel back. Without this the class that gets erased is
    #    Rectangle_concave's nested Rectangle, which is exactly the rarest
    #    labelling in the set.
    present = set(np.unique(instance).tolist()) - {0}
    for position, index in enumerate(kept, start=1):
        if position in present:
            continue
        top, bottom, left, right = entries[index]["box"]
        depth = entries[index]["depth"]
        local_row, local_column = np.unravel_index(int(depth.argmax()), depth.shape)
        semantic[top + local_row, left + local_column] = entries[index]["semantic_id"]
        instance[top + local_row, left + local_column] = position

    surviving_instances = int(
        np.count_nonzero(np.unique(instance))
    )
    destroyed_instances = len(kept) - surviving_instances

    return (
        semantic,
        instance,
        surviving_instances,
        overlapping_pixels,
        destroyed_instances,
        conflict_counts,
    )


def prepared_split_paths(
    split: str,
    array_dir: Path = ARRAY_DIR,
) -> dict[str, Path]:
    array_dir = Path(array_dir)
    return {
        "images": array_dir / f"X_{split}.npy",
        "semantic": array_dir / f"Y_semantic_{split}.npy",
        "instance": array_dir / f"Y_instance_{split}.npy",
    }


def prepare_split(split: str) -> dict[str, object]:
    """Create compact uint8/uint16 letterboxed arrays for one split."""
    records = list_records(split)
    paths = prepared_split_paths(split)
    count = len(records)

    images_mm = open_memmap(
        paths["images"],
        mode="w+",
        dtype=np.uint8,
        shape=(count, IMG_SIZE, IMG_SIZE, 3),
    )
    semantic_mm = open_memmap(
        paths["semantic"],
        mode="w+",
        dtype=np.uint8,
        shape=(count, IMG_SIZE, IMG_SIZE),
    )
    instance_mm = open_memmap(
        paths["instance"],
        mode="w+",
        dtype=INSTANCE_ID_DTYPE,
        shape=(count, IMG_SIZE, IMG_SIZE),
    )

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_instances = 0
    total_overlapping_pixels = 0
    total_destroyed_instances = 0
    total_conflicts = {"duplicate": 0, "nested": 0, "bleed": 0}
    images_with_overlap = 0
    records_metadata: list[dict[str, object] | None] = [None] * count

    def prepare_one(index: int) -> tuple:
        image_path, label_path = records[index]
        original = read_rgb_image(image_path)
        letterboxed, metadata = letterbox_rgb(original)
        (
            semantic,
            instance,
            object_count,
            overlapping_pixels,
            destroyed_instances,
            conflict_counts,
        ) = rasterize_instances(label_path, metadata)
        # Distinct rows of each memmap; no two workers touch the same bytes.
        images_mm[index] = letterboxed
        semantic_mm[index] = semantic
        instance_mm[index] = instance
        return (
            index,
            image_path,
            label_path,
            metadata,
            object_count,
            overlapping_pixels,
            destroyed_instances,
            np.bincount(semantic.ravel(), minlength=NUM_CLASSES),
            conflict_counts,
        )

    # PNG decode dominates here and cv2 releases the GIL, so threads give real
    # parallelism without pickling a 512x512x3 array back from a child process.
    completed = 0
    with ThreadPoolExecutor(max_workers=int(PREPARE_WORKERS)) as pool:
        for result in pool.map(prepare_one, range(count)):
            (
                index,
                image_path,
                label_path,
                metadata,
                object_count,
                overlapping_pixels,
                destroyed_instances,
                semantic_counts,
                conflict_counts,
            ) = result
            for kind, number in conflict_counts.items():
                total_conflicts[kind] += int(number)
            total_overlapping_pixels += overlapping_pixels
            total_destroyed_instances += destroyed_instances
            if overlapping_pixels:
                images_with_overlap += 1
            class_counts += semantic_counts
            total_instances += object_count
            records_metadata[index] = {
                "index": index,
                "image": str(image_path),
                "label": str(label_path),
                "objects": object_count,
                "overlapping_pixels": overlapping_pixels,
                "letterbox": metadata,
            }
            completed += 1
            if completed % 200 == 0 or completed == count:
                print(f"{split}: prepared {completed}/{count}", flush=True)

    if any(record is None for record in records_metadata):
        raise RuntimeError(f"{split}: not every record was prepared.")

    for array in (images_mm, semantic_mm, instance_mm):
        array.flush()

    # Overlapping polygons are now resolved rather than allowed to overwrite
    # one another, so this is a description of the labels and not of damage:
    # the contested pixels went to the inner feature or to whichever object
    # they lie deepest inside, deterministically and independently of the order
    # the annotation tool happened to write them in.
    if total_overlapping_pixels:
        print(
            f"\n{split}: {images_with_overlap:,} of {count:,} images contain "
            f"overlapping polygons ({total_overlapping_pixels:,} pixels "
            f"claimed by more than one).\n"
            f"         resolved: {total_conflicts['nested']:,} nested, "
            f"{total_conflicts['duplicate']:,} duplicate, "
            f"{total_conflicts['bleed']:,} bleed. No instance was lost.\n"
            "         NOTE: this pixel count is not comparable with the "
            "dataset audit's. The old counter summed 'pixels already\n"
            "         claimed' per polygon, so a triple overlap counted twice; "
            "this counts distinct pixels with >1 claim."
        )

    # Structurally impossible now -- every polygon is guaranteed at least its
    # innermost pixel. If it ever fires, the resolution rule has a hole in it
    # and every metric computed downstream is measuring the wrong targets, so
    # this fails the prepare rather than printing another warning nobody reads.
    if total_destroyed_instances:
        raise RuntimeError(
            f"{split}: {total_destroyed_instances:,} instances were destroyed "
            "during rasterisation. rasterize_instances guarantees every "
            "polygon at least one pixel, so this is a bug in the overlap "
            "resolution, not a label problem."
        )

    records_path = ARRAY_DIR / f"{split}_records.json"
    records_path.write_text(
        json.dumps(records_metadata, indent=2), encoding="utf-8"
    )
    return {
        "split": split,
        "images": count,
        "instances": total_instances,
        "class_pixel_counts": class_counts.tolist(),
        "overlapping_pixels": int(total_overlapping_pixels),
        "images_with_overlap": int(images_with_overlap),
        "destroyed_instances": int(total_destroyed_instances),
        "files": {key: str(value) for key, value in paths.items()},
        "records": str(records_path),
    }


def prepare_dataset() -> None:
    """Create all new arrays without touching the old direct-resize arrays."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "version": "Model_v6.1_v5_augmentation",
        "dataset_root": str(DATASET_ROOT),
        "array_dir": str(ARRAY_DIR),
        "image_size": IMG_SIZE,
        "preprocessing": "aspect-ratio-preserving letterbox",
        "letterbox_fill_value": LETTERBOX_FILL_VALUE,
        "classes": CLASS_NAMES,
        "yolo_to_semantic_id": YOLO_TO_SEMANTIC_ID,
        "stored_image_dtype": "uint8",
        "splits": [],
    }
    for split in PREPARE_SPLITS:
        split_summary = prepare_split(split)
        summary["splits"].append(split_summary)
        print(
            f"{split}: {split_summary['images']:,} images, "
            f"{split_summary['instances']:,} instances"
        )

    metadata_path = ARRAY_DIR / "instance_dataset_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nLetterboxed instance dataset created successfully:", ARRAY_DIR)
    print("Metadata:", metadata_path)


# =============================================================================
# 4. CONSISTENT INSTANCE TARGET GENERATION
# =============================================================================

def validate_spatial_configuration() -> None:
    """Fail early if input, target, and network resolutions can diverge."""
    if IMG_SIZE <= 0:
        raise ValueError(f"IMG_SIZE must be positive, got {IMG_SIZE}.")
    if IMG_SIZE % 32 != 0:
        raise ValueError(
            f"IMG_SIZE must be divisible by 32 for this encoder, got {IMG_SIZE}."
        )
    expected_head_size = IMG_SIZE // 2
    if INSTANCE_HEAD_SIZE != expected_head_size:
        raise ValueError(
            "INSTANCE_HEAD_SIZE must match the stride-2 centre/offset head: "
            f"expected {expected_head_size}, got {INSTANCE_HEAD_SIZE}."
        )


def make_instance_boundary(instance_map: np.ndarray) -> np.ndarray:
    """Create a boundary target from changes in true instance IDs."""
    instance_map = np.asarray(instance_map)
    boundary = np.zeros(instance_map.shape, dtype=np.uint8)

    horizontal = (
        (instance_map[:, 1:] != instance_map[:, :-1])
        & ((instance_map[:, 1:] > 0) | (instance_map[:, :-1] > 0))
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical = (
        (instance_map[1:, :] != instance_map[:-1, :])
        & ((instance_map[1:, :] > 0) | (instance_map[:-1, :] > 0))
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical

    if BOUNDARY_DILATION_ITERATIONS > 0:
        boundary = cv2.dilate(
            boundary,
            np.ones((3, 3), dtype=np.uint8),
            iterations=BOUNDARY_DILATION_ITERATIONS,
        )
    return boundary[..., np.newaxis].astype(np.float32)


def draw_gaussian_peak(
    heatmap: np.ndarray,
    center_x: int,
    center_y: int,
    sigma: float,
) -> None:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x0 = max(0, center_x - radius)
    x1 = min(heatmap.shape[1], center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(heatmap.shape[0], center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32) - float(center_x)
    ys = np.arange(y0, y1, dtype=np.float32) - float(center_y)
    gaussian = np.exp(
        -(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma * sigma)
    ).astype(np.float32)
    heatmap[y0:y1, x0:x1] = np.maximum(
        heatmap[y0:y1, x0:x1], gaussian
    )


def instance_area_and_majority_class(
    flat_instance: np.ndarray,
    flat_semantic: np.ndarray,
    max_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-instance (area, majority class, has-any-class) in one pass.

    The obvious loop — ``mask = instance == id`` for each id — costs a full
    pass over the image per instance, and this data has 32 objects per training
    image and 77 per test image. A pair of ``bincount`` calls over the flattened
    arrays computes the same table in two passes total, independent of the
    object count. ``argmax`` breaks ties towards the lowest class id, matching
    ``np.bincount(...).argmax()`` on the per-instance pixel list.
    """
    areas = np.bincount(flat_instance, minlength=max_id + 1)
    positive = flat_semantic > 0
    if positive.any():
        histogram = np.bincount(
            flat_instance[positive] * NUM_CLASSES + flat_semantic[positive],
            minlength=(max_id + 1) * NUM_CLASSES,
        ).reshape(max_id + 1, NUM_CLASSES)
    else:
        histogram = np.zeros((max_id + 1, NUM_CLASSES), dtype=np.int64)
    return areas, histogram.argmax(axis=1).astype(np.int32), histogram.any(axis=1)


def sanitize_semantic_and_instances(
    semantic: np.ndarray,
    instance: np.ndarray,
    minimum_area: int = MIN_TARGET_INSTANCE_AREA_FULL_RES,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove invalid fragments and make every instance single-class.

    Bit-identical to the per-instance loop it replaces, but single-pass: it is
    called once per sample in ``__getitem__`` on the data-loader threads, so its
    cost is paid 4,680 times per epoch.
    """
    instance = np.asarray(instance, dtype=np.int32)
    semantic = np.asarray(semantic, dtype=np.int32)
    instance = np.where(instance > 0, instance, 0).astype(np.int32)
    semantic = np.where(instance > 0, semantic, 0).astype(np.int32)

    max_id = int(instance.max()) if instance.size else 0
    if max_id <= 0:
        return semantic, instance

    areas, majority, has_class = instance_area_and_majority_class(
        instance.ravel(), semantic.ravel(), max_id
    )
    keep = (areas >= int(minimum_area)) & has_class
    keep[0] = False

    class_lut = np.where(keep, majority, 0).astype(np.int32)
    id_lut = np.where(keep, np.arange(max_id + 1), 0).astype(np.int32)
    return class_lut[instance], id_lut[instance]


def build_center_and_offset_targets(
    semantic_full: np.ndarray,
    instance_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all centre/offset targets at the model head resolution."""
    if semantic_full.shape != instance_full.shape:
        raise ValueError(
            "Semantic and instance target shapes differ: "
            f"{semantic_full.shape} versus {instance_full.shape}."
        )
    if semantic_full.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Expected full targets {(IMG_SIZE, IMG_SIZE)}, "
            f"got {semantic_full.shape}."
        )
    semantic_small = cv2.resize(
        semantic_full.astype(np.uint8),
        (INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    instance_small = cv2.resize(
        instance_full.astype(np.float32),
        (INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    instance_small = np.rint(instance_small).astype(np.int32)
    semantic_small[instance_small <= 0] = 0
    head_height, head_width = semantic_small.shape
    if instance_small.shape != (head_height, head_width):
        raise RuntimeError(
            "Downsampled semantic and instance target shapes differ."
        )
    if (head_height, head_width) != (
        INSTANCE_HEAD_SIZE,
        INSTANCE_HEAD_SIZE,
    ):
        raise RuntimeError(
            "Generated instance-head targets do not match the configured "
            f"head size: {(head_height, head_width)} versus "
            f"{(INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE)}."
        )

    center = np.zeros(
        (head_height, head_width, NUM_CLASSES - 1),
        dtype=np.float32,
    )
    offset = np.zeros(
        (head_height, head_width, 3), dtype=np.float32
    )

    # Single-pass equivalent of the former per-instance loop. Every quantity the
    # loop derived from `mask = instance_small == id` is a group reduction over
    # the foreground pixel list, which bincount computes for all instances at
    # once. Only the Gaussian stamp stays a loop, and that touches at most a
    # 31x31 window per object rather than the whole 256x256 head.
    max_id = int(instance_small.max()) if instance_small.size else 0
    if max_id <= 0:
        return center, offset

    ys_all, xs_all = np.nonzero(instance_small > 0)
    if ys_all.size == 0:
        return center, offset
    ids = instance_small[ys_all, xs_all]
    semantic_at = semantic_small[ys_all, xs_all]

    positive = semantic_at > 0
    if not positive.any():
        return center, offset
    histogram = np.bincount(
        ids[positive] * NUM_CLASSES + semantic_at[positive],
        minlength=(max_id + 1) * NUM_CLASSES,
    ).reshape(max_id + 1, NUM_CLASSES)
    majority = histogram.argmax(axis=1).astype(np.int32)
    keep = histogram.any(axis=1)
    keep[0] = False

    counts = np.bincount(ids, minlength=max_id + 1)
    safe_counts = np.maximum(counts, 1)
    mean_x = np.bincount(ids, weights=xs_all, minlength=max_id + 1) / safe_counts
    mean_y = np.bincount(ids, weights=ys_all, minlength=max_id + 1) / safe_counts

    # Representative pixel: the instance pixel nearest its centroid, ties broken
    # by row-major order — exactly what argmin over np.nonzero output produced.
    distance = (xs_all - mean_x[ids]) ** 2 + (ys_all - mean_y[ids]) ** 2
    order = np.lexsort((np.arange(ids.size), distance, ids))
    ids_sorted = ids[order]
    first = np.empty(ids_sorted.size, dtype=bool)
    first[0] = True
    np.not_equal(ids_sorted[1:], ids_sorted[:-1], out=first[1:])
    representative = order[first]

    center_x_lut = np.zeros(max_id + 1, dtype=np.int32)
    center_y_lut = np.zeros(max_id + 1, dtype=np.int32)
    center_x_lut[ids_sorted[first]] = xs_all[representative]
    center_y_lut[ids_sorted[first]] = ys_all[representative]

    sigma_lut = np.clip(
        np.sqrt(counts.astype(np.float64)) / 8.0,
        CENTER_SIGMA_RANGE[0],
        CENTER_SIGMA_RANGE[1],
    )
    for instance_id in np.nonzero(keep)[0]:
        draw_gaussian_peak(
            center[..., int(majority[instance_id]) - 1],
            int(center_x_lut[instance_id]),
            int(center_y_lut[instance_id]),
            float(sigma_lut[instance_id]),
        )

    written = keep[ids]
    if written.any():
        write_y = ys_all[written]
        write_x = xs_all[written]
        write_id = ids[written]
        offset[write_y, write_x, 0] = (
            center_x_lut[write_id].astype(np.float32)
            - write_x.astype(np.float32)
        ) / float(head_width)
        offset[write_y, write_x, 1] = (
            center_y_lut[write_id].astype(np.float32)
            - write_y.astype(np.float32)
        ) / float(head_height)
        offset[write_y, write_x, 2] = 1.0

    return center, offset


def build_inner_distance_target(
    semantic_full: np.ndarray,
    instance_full: np.ndarray,
) -> np.ndarray:
    """Per-instance normalised inner distance at head resolution: [H, W, 2].

    Channel 0 is distance-to-own-boundary divided by that instance's own
    maximum, so it runs 0 at the rim to 1 at the innermost pixel of every
    object regardless of size. Channel 1 is the foreground validity mask, the
    same convention the offset target uses in its third channel.

    This replaces the two-component offset field. The offset head asked every
    pixel to name a vector to its object's centre, which is inherently
    scale-dependent: a pixel on a 130 px object must emit a vector an order of
    magnitude longer than one on a 16 px object, and the head was only ever
    trained on the short end. Measured on real boards, 30.1% of Rectangles sit
    beyond anything training produced, and those are exactly the objects that
    shatter into strips. A normalised inner distance has no such range -- it is
    bounded in [0,1] for every object at every scale, which is the property the
    offset field never had.

    One distance transform serves all instances at once. A pixel is treated as
    an edge when it is background or 4-adjacent to a different instance id, so
    the transform of the remaining interior measures distance to each object's
    own boundary, including the boundary it shares with a touching neighbour.
    Frame-edge pixels count as edges too: a truncated object's distance then
    peaks inside its visible part, which is what the decoder has to work with
    anyway.
    """
    if semantic_full.shape != instance_full.shape:
        raise ValueError(
            "Semantic and instance target shapes differ: "
            f"{semantic_full.shape} versus {instance_full.shape}."
        )
    if semantic_full.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Expected full targets {(IMG_SIZE, IMG_SIZE)}, "
            f"got {semantic_full.shape}."
        )

    semantic_small = cv2.resize(
        semantic_full.astype(np.uint8),
        (INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    instance_small = cv2.resize(
        instance_full.astype(np.float32),
        (INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    instance_small = np.rint(instance_small).astype(np.int32)
    semantic_small[instance_small <= 0] = 0

    target = np.zeros(
        (INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE, 2), dtype=np.float32
    )
    foreground = (instance_small > 0) & (semantic_small > 0)
    if not foreground.any():
        return target

    # Zero-pad so an object touching the frame is bounded by it rather than
    # running off the edge with no boundary to measure against.
    padded = np.pad(instance_small, 1, mode="constant", constant_values=0)
    differs = (
        (padded[1:-1, 2:] != instance_small)
        | (padded[1:-1, :-2] != instance_small)
        | (padded[2:, 1:-1] != instance_small)
        | (padded[:-2, 1:-1] != instance_small)
    )
    interior = foreground & ~differs

    distance = cv2.distanceTransform(
        interior.astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)

    # Normalise by each instance's own maximum. `np.maximum.at` is the
    # unbuffered group reduction; a plain fancy-index assignment would keep
    # only the last write per instance rather than the largest.
    ids = instance_small[foreground]
    values = distance[foreground]
    peaks = np.zeros(int(instance_small.max()) + 1, dtype=np.float32)
    np.maximum.at(peaks, ids, values)
    # A one-pixel-wide object has every pixel on its boundary and so peaks at
    # zero. Its single pixel is still a valid core, so give it 1.0 rather than
    # dividing by zero and erasing an object the decoder could have found.
    safe_peaks = np.where(peaks > 0.0, peaks, 1.0)
    normalised = np.where(values > 0.0, values / safe_peaks[ids], 0.0)
    degenerate = peaks[ids] <= 0.0
    normalised = np.where(degenerate, 1.0, normalised)

    target[..., 0][foreground] = normalised.astype(np.float32)
    target[..., 1][foreground] = 1.0
    return target


def build_all_targets(
    semantic: np.ndarray,
    instance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One authoritative target generator for every training mode.

    Three heads now, not four: the centre and offset targets are replaced by a
    single inner-distance field. `build_center_and_offset_targets` is retained,
    uncalled, so a V6.2 checkpoint can still be decoded and compared against.
    """
    semantic, instance = sanitize_semantic_and_instances(semantic, instance)
    distance = build_inner_distance_target(semantic, instance)
    boundary = make_instance_boundary(instance)
    return semantic, distance, boundary


# =============================================================================
# 5. EXACT MODEL_V5 PAIRED ONLINE AUGMENTATION
# =============================================================================

def validate_augmentation_configuration() -> None:
    expected_cycle_length = (
        1 + EXACT_ROTATION_MODE_COUNT + REALISTIC_VARIANT_MODE_COUNT
    )
    if AUGMENTATION_CYCLE_LENGTH != expected_cycle_length:
        raise ValueError(
            "AUGMENTATION_CYCLE_LENGTH must equal one original mode + all "
            "exact rotations + all realistic variants: "
            f"expected {expected_cycle_length}."
        )
    if MIN_ROTATION_DEGREES <= 0:
        raise ValueError("MIN_ROTATION_DEGREES must be positive.")
    if MAX_ROTATION_DEGREES < MIN_ROTATION_DEGREES:
        raise ValueError(
            "MAX_ROTATION_DEGREES is smaller than the minimum."
        )
    if ROTATION_STEP_DEGREES <= 0:
        raise ValueError("ROTATION_STEP_DEGREES must be positive.")
    expected_angles = tuple(
        range(
            MIN_ROTATION_DEGREES,
            MAX_ROTATION_DEGREES + 1,
            ROTATION_STEP_DEGREES,
        )
    )
    if EXACT_ROTATION_ANGLES != expected_angles:
        raise ValueError(
            "EXACT_ROTATION_ANGLES must contain every configured angle."
        )
    if EXACT_ROTATION_MODE_COUNT != 45:
        raise ValueError(
            "Expected exactly 45 exact-rotation modes; "
            f"MIN/MAX/STEP_ROTATION_DEGREES currently give "
            f"{EXACT_ROTATION_MODE_COUNT} "
            f"({EXACT_ROTATION_ANGLES[0]} to {EXACT_ROTATION_ANGLES[-1]} "
            f"degrees in steps of {ROTATION_STEP_DEGREES})."
        )
    if REALISTIC_VARIANT_MODE_COUNT != 49:
        raise ValueError(
            "The Model_v5 realistic augmentation block must keep 49 modes."
        )
    if ZOOM_SAFETY_FACTOR < 1.0:
        raise ValueError("ZOOM_SAFETY_FACTOR must be at least 1.0.")
    if not (0.0 <= MAX_TRANSLATION_FRACTION < 0.5):
        raise ValueError(
            "MAX_TRANSLATION_FRACTION must be in [0, 0.5)."
        )


def realistic_variant_name(variant_mode: int) -> str:
    """Describe one of the original 49 Model_v5 realistic variants."""
    if 1 <= variant_mode <= 15:
        return "rotation"
    if 16 <= variant_mode <= 23:
        return "rotation + brightness/contrast"
    if 24 <= variant_mode <= 29:
        return "rotation + gamma/exposure"
    if 30 <= variant_mode <= 35:
        return "rotation + zoom/translation"
    if 36 <= variant_mode <= 40:
        return "rotation + blur/sharpen"
    if 41 <= variant_mode <= 44:
        return "rotation + camera noise"
    if 45 <= variant_mode <= 47:
        return "rotation + colour variation"
    if 48 <= variant_mode <= 49:
        return "combined realistic augmentation"
    raise ValueError(f"Invalid realistic variant mode: {variant_mode}")


def augmentation_mode_name(mode: int) -> str:
    """Describe the exact 1 + 45 + 49 Model_v5 schedule."""
    if mode == 0:
        return "original"
    if 1 <= mode <= EXACT_ROTATION_MODE_COUNT:
        angle = EXACT_ROTATION_ANGLES[mode - 1]
        return f"exact pure rotation +{angle} degrees"
    if mode < AUGMENTATION_CYCLE_LENGTH:
        variant_mode = mode - REALISTIC_VARIANT_MODE_OFFSET
        return realistic_variant_name(variant_mode)
    raise ValueError(f"Invalid augmentation mode: {mode}")


def sample_rotation_angle(rng: np.random.Generator) -> float:
    """Sample one configured exact rotation, optionally in either direction."""
    choices = np.arange(
        MIN_ROTATION_DEGREES,
        MAX_ROTATION_DEGREES + 1,
        ROTATION_STEP_DEGREES,
        dtype=np.int32,
    )
    angle = float(rng.choice(choices))
    if ROTATE_BOTH_DIRECTIONS and rng.random() < 0.5:
        angle = -angle
    return angle


def zoom_to_fill_affine_matrix(
    image_width: int,
    image_height: int,
    angle_degrees: float,
    extra_zoom: float,
    allow_translation: bool,
    rng: np.random.Generator,
    fill_frame: bool = True,
) -> np.ndarray:
    """Return the Model_v5 zoom-to-fill affine matrix.

    ``fill_frame=False`` selects the zoom-out branch: the rotated content is
    deliberately allowed not to cover the canvas so that the model sees
    objects smaller than they were labelled. Uncovered pixels are padded with
    the letterbox value by ``warp_image_semantic_and_instances``.
    """
    radians = math.radians(angle_degrees)
    absolute_cosine = abs(math.cos(radians))
    absolute_sine = abs(math.sin(radians))
    aspect_factor = max(
        image_width / image_height,
        image_height / image_width,
    )
    fill_zoom = (
        (absolute_cosine + aspect_factor * absolute_sine)
        * ZOOM_SAFETY_FACTOR
        if fill_frame
        else 1.0
    )
    total_zoom = fill_zoom * float(extra_zoom)

    matrix = cv2.getRotationMatrix2D(
        (image_width / 2.0, image_height / 2.0),
        angle_degrees,
        total_zoom,
    ).astype(np.float64)

    if allow_translation:
        extra_margin = max(0.0, float(extra_zoom) - 1.0)
        maximum_x = min(
            MAX_TRANSLATION_FRACTION * image_width,
            0.45 * extra_margin * image_width,
        )
        maximum_y = min(
            MAX_TRANSLATION_FRACTION * image_height,
            0.45 * extra_margin * image_height,
        )
        matrix[0, 2] += float(
            rng.uniform(-maximum_x, maximum_x)
        )
        matrix[1, 2] += float(
            rng.uniform(-maximum_y, maximum_y)
        )
    return matrix


def warp_image_semantic_and_instances(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    matrix: np.ndarray,
    fill_value: int = LETTERBOX_FILL_VALUE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the unchanged V5 interpolation and border modes.

    ``fill_value`` is the 0-255 level written where the warp uncovers the
    canvas. It is a no-op for every zoom-in mode, which covers the frame by
    construction; the zoom-out branch passes ``ZOOM_OUT_FILL_VALUE``.
    """
    height, width = semantic.shape
    output_size = (width, height)
    # V5 used BORDER_REPLICATE, which is unreachable while zoom-to-fill
    # guarantees coverage, and which would smear the border pixels of a
    # zoomed-out board across the padding instead of marking it as not-board.
    warped_image = cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            float(fill_value) / 255.0,
        ) * 3,
    )
    warped_semantic = cv2.warpAffine(
        semantic.astype(np.uint8),
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)
    warped_instance = cv2.warpAffine(
        instance.astype(np.float32),
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (
        warped_image,
        warped_semantic,
        np.rint(warped_instance).astype(np.int32),
    )


def letterboxed_board_bounds(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) of the board inside its letterbox.

    Letterbox bars are whole rows or columns of exactly LETTERBOX_FILL_VALUE,
    so the board is everything between the first and last row/column that
    holds any other value.
    """
    fill = float(LETTERBOX_FILL_VALUE) / 255.0
    is_bar = np.all(np.abs(np.asarray(image) - fill) < 1e-4, axis=2)
    rows = np.flatnonzero(~is_bar.all(axis=1))
    cols = np.flatnonzero(~is_bar.all(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, image.shape[0], 0, image.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def mirror_side_sample(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace the board by its mirror image; the original is not kept.

    "left" mirrors the board across a vertical edge (a left-right flip) and
    "top" across a horizontal edge (a top-bottom flip).
    Only the real board is flipped, in place inside its letterbox, so the bars
    and the board position are unchanged. Labels are flipped with the pixels
    and every object keeps its instance id: the mirror image has exactly the
    objects of the original, and nothing is added.
    """
    if side not in MIRROR_SIDES:
        raise ValueError(
            f"side must be one of {MIRROR_SIDES}; received {side!r}."
        )
    top, bottom, left, right = letterboxed_board_bounds(image)
    window = (slice(top, bottom), slice(left, right))
    axis = 1 if side == "left" else 0

    out_image = np.array(image, dtype=np.float32, copy=True)
    out_semantic = np.array(semantic, dtype=np.int32, copy=True)
    out_instance = np.array(instance, dtype=np.int32, copy=True)
    out_image[window] = np.flip(out_image[window], axis)
    out_semantic[window] = np.flip(out_semantic[window], axis)
    out_instance[window] = np.flip(out_instance[window], axis)
    return out_image, out_semantic, out_instance


def apply_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    brightness = float(
        rng.uniform(-BRIGHTNESS_LIMIT, BRIGHTNESS_LIMIT) * strength
    )
    contrast = float(
        1.0
        + rng.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT) * strength
    )
    return np.clip(
        (image - 0.5) * contrast + 0.5 + brightness,
        0.0,
        1.0,
    )


def apply_gamma_exposure(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    gamma_low = 1.0 + (GAMMA_RANGE[0] - 1.0) * strength
    gamma_high = 1.0 + (GAMMA_RANGE[1] - 1.0) * strength
    gain_low = 1.0 + (EXPOSURE_GAIN_RANGE[0] - 1.0) * strength
    gain_high = 1.0 + (EXPOSURE_GAIN_RANGE[1] - 1.0) * strength
    gamma = float(rng.uniform(gamma_low, gamma_high))
    gain = float(rng.uniform(gain_low, gain_high))
    return np.clip(
        np.power(np.clip(image, 0.0, 1.0), gamma) * gain,
        0.0,
        1.0,
    )


def apply_blur_or_sharpen(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    sigma = float(
        rng.uniform(0.30, 0.85) * max(strength, 0.25)
    )
    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if rng.random() < 0.5:
        return np.clip(blurred, 0.0, 1.0)
    amount = float(rng.uniform(0.15, 0.35) * strength)
    return np.clip(
        image + amount * (image - blurred), 0.0, 1.0
    )


def apply_camera_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    standard_deviation = float(
        rng.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])
        * strength
    )
    noise = rng.normal(
        0.0, standard_deviation, image.shape
    ).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def apply_colour_variation(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(
        image.astype(np.float32), cv2.COLOR_RGB2HSV
    )
    hue_shift = float(
        rng.uniform(-HUE_SHIFT_DEGREES, HUE_SHIFT_DEGREES)
        * strength
    )
    saturation_low = (
        1.0 + (SATURATION_GAIN_RANGE[0] - 1.0) * strength
    )
    saturation_high = (
        1.0 + (SATURATION_GAIN_RANGE[1] - 1.0) * strength
    )
    saturation_gain = float(
        rng.uniform(saturation_low, saturation_high)
    )
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 360.0)
    hsv[..., 1] = np.clip(
        hsv[..., 1] * saturation_gain, 0.0, 1.0
    )
    return np.clip(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0
    )


def augment_training_sample(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    mode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one Model_v5 paired augmentation mode.

    The 95-mode schedule, the photometric ranges and the zoom-to-fill geometry
    are unchanged from Model_v5. The one addition is a zoom-out branch on the
    realistic-variant modes: V5 could only ever enlarge an object, so the
    model never saw anything smaller than it was labelled while inference is
    never zoomed at all.
    """
    if mode <= 0 or mode >= AUGMENTATION_CYCLE_LENGTH:
        raise ValueError(
            f"Expected augmentation mode "
            f"1..{AUGMENTATION_CYCLE_LENGTH - 1}, got {mode}."
        )

    image = normalize_image(image)
    if mode <= EXACT_ROTATION_MODE_COUNT:
        angle = float(EXACT_ROTATION_ANGLES[mode - 1])
        variant_mode = 0
        use_translation = False
        extra_zoom = 1.0
    else:
        variant_mode = mode - REALISTIC_VARIANT_MODE_OFFSET
        angle = sample_rotation_angle(rng)
        use_translation = (
            30 <= variant_mode <= 35
            or 48 <= variant_mode <= 49
        )
        if 30 <= variant_mode <= 35:
            extra_zoom = float(rng.uniform(*EXTRA_ZOOM_RANGE))
        elif 48 <= variant_mode <= 49:
            extra_zoom = float(
                rng.uniform(1.04, EXTRA_ZOOM_RANGE[1])
            )
        else:
            extra_zoom = 1.0

    # Zoom-out replaces the zoom-in factor for this sample rather than
    # compounding with it. Exact-rotation modes stay bit-for-bit V5.
    fill_frame = True
    warp_fill_value = LETTERBOX_FILL_VALUE
    if (
        variant_mode > 0
        and ZOOM_OUT_PROBABILITY > 0.0
        and ZOOM_OUT_STEPS
        and rng.random() < float(ZOOM_OUT_PROBABILITY)
    ):
        fill_frame = False
        # One of the 5%-spaced steps from 5% to 50% out, uniformly.
        extra_zoom = float(ZOOM_OUT_STEPS[rng.integers(len(ZOOM_OUT_STEPS))])
        warp_fill_value = ZOOM_OUT_FILL_VALUE
    elif (
        variant_mode > 0
        and ZOOM_IN_PROBABILITY > 0.0
        and ZOOM_IN_STEPS
        and rng.random() < float(ZOOM_IN_PROBABILITY)
    ):
        # The mirror of the branch above, and added for the mirror-image
        # defect. `extra_zoom` alone reaches only 8 of the 95 modes and stops
        # at 1.12x, so training topped out around 1.7x labelled scale once
        # zoom-to-fill is included -- while 30.1% of Rectangles on real boards
        # sit beyond that. Zoom-out already had a p=0.50 branch of its own for
        # exactly the opposite reason; this restores the symmetry.
        extra_zoom = float(ZOOM_IN_STEPS[rng.integers(len(ZOOM_IN_STEPS))])

    matrix = zoom_to_fill_affine_matrix(
        image_width=image.shape[1],
        image_height=image.shape[0],
        angle_degrees=angle,
        extra_zoom=extra_zoom,
        allow_translation=use_translation,
        rng=rng,
        fill_frame=fill_frame,
    )
    image, semantic, instance = warp_image_semantic_and_instances(
        image, semantic, instance, matrix, fill_value=warp_fill_value
    )
    # Sanitization is intentionally deferred to build_all_targets(), which is
    # the single authoritative pass for original and augmented samples.

    if 16 <= variant_mode <= 23:
        image = apply_brightness_contrast(image, rng)
    elif 24 <= variant_mode <= 29:
        image = apply_gamma_exposure(image, rng)
    elif 36 <= variant_mode <= 40:
        image = apply_blur_or_sharpen(image, rng)
    elif 41 <= variant_mode <= 44:
        image = apply_camera_noise(image, rng)
    elif 45 <= variant_mode <= 47:
        image = apply_colour_variation(image, rng)
    elif 48 <= variant_mode <= 49:
        image = apply_brightness_contrast(
            image, rng, strength=0.60
        )
        image = apply_gamma_exposure(
            image, rng, strength=0.50
        )
        image = apply_colour_variation(
            image, rng, strength=0.50
        )
        if rng.random() < 0.5:
            image = apply_camera_noise(
                image, rng, strength=0.50
            )
        else:
            image = apply_blur_or_sharpen(
                image, rng, strength=0.50
            )

    image = np.asarray(
        np.clip(image, 0.0, 1.0), dtype=np.float32
    )
    if not np.all(np.isfinite(image)):
        raise ValueError(
            "Online augmentation produced NaN or infinity."
        )
    return image, semantic, instance


def apply_dihedral_transform(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    transform_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one of the eight exactly label-preserving dihedral transforms.

    Index 0 is the identity, 1..3 are 90-degree rotations, and 4..7 are the
    same rotations followed by a horizontal mirror. Every transform is exact:
    it moves whole pixels, so unlike the arbitrary-angle rotations it cannot
    blur a label edge or resample an instance ID.

    This must run BEFORE ``build_all_targets``. The centre, offset and
    boundary targets are then regenerated from the transformed instance map,
    which is why the offset vector field needs no separate rotation here.
    """
    transform_index = int(transform_index)
    if not 0 <= transform_index <= 7:
        raise ValueError(
            f"transform_index must be in 0..7; received {transform_index}."
        )
    if semantic.shape != instance.shape:
        raise ValueError(
            "Semantic and instance shapes differ: "
            f"{semantic.shape} versus {instance.shape}."
        )
    if image.shape[:2] != semantic.shape:
        raise ValueError(
            "Image and label spatial shapes differ: "
            f"{image.shape[:2]} versus {semantic.shape}."
        )

    if transform_index == 0:
        return image, semantic, instance

    quarter_turns = transform_index % 4
    mirror = transform_index >= 4

    if quarter_turns:
        image = np.rot90(image, quarter_turns, axes=(0, 1))
        semantic = np.rot90(semantic, quarter_turns)
        instance = np.rot90(instance, quarter_turns)

    if mirror:
        image = image[:, ::-1]
        semantic = semantic[:, ::-1]
        instance = instance[:, ::-1]

    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(semantic),
        np.ascontiguousarray(instance),
    )


def mosaic_samples(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile four (image, semantic, instance) samples 2x2 into one frame.

    Each quadrant is downscaled by two: images bilinearly, semantic and instance
    maps by nearest neighbour, because an interpolated class id or instance id is
    meaningless. Instance ids are re-based per quadrant so two objects from
    different source boards can never collide on one id.

    Objects below MIN_AUGMENTED_INSTANCE_AREA after downscaling are dropped from
    both maps together. Halving a diameter quarters an area, so a small object
    can fall under the floor, and leaving it in the instance map while it is too
    small to be a valid target is how a phantom label reaches the loss.
    """
    if len(samples) != 4:
        raise ValueError(f"mosaic needs exactly four samples; got {len(samples)}.")
    half = IMG_SIZE // 2
    image_out = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    semantic_out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    instance_out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

    next_id = 1
    corners = ((0, 0), (0, half), (half, 0), (half, half))
    for (top, left), (image, semantic, instance) in zip(corners, samples):
        image_small = cv2.resize(
            np.asarray(image, dtype=np.float32), (half, half),
            interpolation=cv2.INTER_AREA,
        )
        semantic_small = cv2.resize(
            np.asarray(semantic, dtype=np.uint8), (half, half),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        instance_small = np.rint(
            cv2.resize(
                np.asarray(instance, dtype=np.float32), (half, half),
                interpolation=cv2.INTER_NEAREST,
            )
        ).astype(np.int32)

        rebased = np.zeros_like(instance_small)
        for source_id in [int(v) for v in np.unique(instance_small) if v > 0]:
            mask = instance_small == source_id
            if int(mask.sum()) < int(MIN_AUGMENTED_INSTANCE_AREA):
                semantic_small[mask] = 0
                continue
            rebased[mask] = next_id
            next_id += 1
        semantic_small[rebased == 0] = 0

        image_out[top:top + half, left:left + half] = image_small
        semantic_out[top:top + half, left:left + half] = semantic_small
        instance_out[top:top + half, left:left + half] = rebased

    return image_out, semantic_out, instance_out


def copy_paste_objects(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    donor_image: np.ndarray,
    donor_semantic: np.ndarray,
    donor_instance: np.ndarray,
    rng: np.random.Generator,
    max_objects: int = COPY_PASTE_MAX_OBJECTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste a few whole donor instances onto this sample.

    Occlusion remnants are not handled here: ``build_all_targets`` runs
    ``sanitize_semantic_and_instances``, which drops any instance a paste has
    cut below ``MIN_TARGET_INSTANCE_AREA_FULL_RES``.
    """
    donor_ids = np.unique(donor_instance)
    donor_ids = donor_ids[donor_ids > 0]
    if donor_ids.size == 0:
        return image, semantic, instance

    image = np.array(image, dtype=np.float32, copy=True)
    semantic = np.array(semantic, dtype=np.int32, copy=True)
    instance = np.array(instance, dtype=np.int32, copy=True)
    next_id = int(instance.max()) + 1

    wanted = int(rng.integers(1, int(max_objects) + 1))
    chosen = rng.choice(
        donor_ids, size=min(wanted, donor_ids.size), replace=False
    )
    for donor_id in chosen:
        if next_id > MAX_INSTANCE_ID:
            break
        mask = donor_instance == int(donor_id)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        classes = np.bincount(
            donor_semantic[mask].astype(np.int64), minlength=NUM_CLASSES
        )
        classes[0] = 0
        if classes.sum() == 0:
            continue
        class_id = int(classes.argmax())

        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
        crop_mask = mask[top:bottom, left:right]
        crop_rgb = donor_image[top:bottom, left:right]

        # An exact dihedral transform on the crop so a paste is never a pixel
        # copy of the donor. Every class here is dihedrally symmetric.
        quarter_turns = int(rng.integers(4))
        crop_mask = np.rot90(crop_mask, quarter_turns)
        crop_rgb = np.rot90(crop_rgb, quarter_turns)
        if rng.random() < 0.5:
            crop_mask = crop_mask[:, ::-1]
            crop_rgb = crop_rgb[:, ::-1]
        crop_mask = np.ascontiguousarray(crop_mask)
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.float32)

        height, width = crop_mask.shape
        if (
            height >= IMG_SIZE
            or width >= IMG_SIZE
            or int(crop_mask.sum()) < MIN_TARGET_INSTANCE_AREA_FULL_RES
        ):
            continue
        y = int(rng.integers(0, IMG_SIZE - height))
        x = int(rng.integers(0, IMG_SIZE - width))
        window = (slice(y, y + height), slice(x, x + width))

        alpha = cv2.GaussianBlur(
            crop_mask.astype(np.float32),
            (0, 0),
            float(COPY_PASTE_EDGE_FEATHER_PX),
        )[..., None]
        image[window] = image[window] * (1.0 - alpha) + crop_rgb * alpha
        semantic[window][crop_mask] = class_id
        instance[window][crop_mask] = next_id
        next_id += 1

    return image, semantic, instance


# =============================================================================
# 6. MEMORY-MAPPED BATCH LOADER
# =============================================================================

def load_split_arrays(
    split: str,
    array_dir: Path = ARRAY_DIR,
) -> dict[str, np.ndarray]:
    array_dir = Path(array_dir)
    paths = prepared_split_paths(split, array_dir=array_dir)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared Model_v6 arrays are missing. Run RUN_MODE='prepare' "
            "first:\n" + "\n".join(str(path) for path in missing)
        )
    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    expected_images = (len(arrays["images"]), IMG_SIZE, IMG_SIZE, 3)
    expected_masks = (len(arrays["images"]), IMG_SIZE, IMG_SIZE)
    if arrays["images"].shape != expected_images:
        raise ValueError(
            f"Unexpected {split} image shape {arrays['images'].shape}; "
            f"expected {expected_images}."
        )
    for key in ("semantic", "instance"):
        if arrays[key].shape != expected_masks:
            raise ValueError(
                f"Unexpected {split} {key} shape {arrays[key].shape}; "
                f"expected {expected_masks}."
            )
    if arrays["instance"].dtype != np.dtype(INSTANCE_ID_DTYPE):
        raise ValueError(
            f"Unexpected {split} instance dtype "
            f"{arrays['instance'].dtype}; expected "
            f"{np.dtype(INSTANCE_ID_DTYPE)}."
        )
    print(
        f"{split}: {len(arrays['images']):,} prepared images, "
        f"image dtype={arrays['images'].dtype}"
    )
    return arrays


class InstanceArraySequence(tf.keras.utils.Sequence):
    """Load images and rebuild mutually consistent four-head targets."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        batch_size: int,
        training: bool,
        seed_offset: int = 0,
        shuffle: bool = True,
    ):
        # cv2 and numpy release the GIL, so threads give real parallelism here
        # without pickling a memmap into a child process every batch.
        super().__init__(
            workers=(
                int(DATA_LOADER_WORKERS)
                if training
                else max(1, int(DATA_LOADER_WORKERS) // 4)
            ),
            use_multiprocessing=False,
            max_queue_size=int(DATA_LOADER_QUEUE_SIZE),
        )
        self.images = arrays["images"]
        self.semantic = arrays["semantic"]
        self.instance = arrays["instance"]
        self.batch_size = int(batch_size)
        self.training = bool(training)
        self.shuffle = bool(shuffle)
        self.seed_offset = int(seed_offset)
        self.epoch_index = 0
        self.indices = np.arange(len(self.images), dtype=np.int64)
        self._donor_index: dict[int, np.ndarray] | None = None
        self.order_rng = np.random.default_rng(
            SEED + self.seed_offset + (1 if training else 2)
        )
        if self.training and self.shuffle:
            self.order_rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.training:
            self.epoch_index += 1
            if self.shuffle:
                self.order_rng.shuffle(self.indices)

    def donor_sample_index(self, rng: np.random.Generator) -> int:
        """Draw a donor biased towards the rare classes.

        A class is chosen uniformly first and the donor only then, so
        Rectangle_concave (0.47% of polygons) donates as often as
        circle_full (52.6%). The index costs one pass over the semantic
        memmap at first use and is kept for the life of the sequence.
        """
        if self._donor_index is None:
            present: dict[int, list[int]] = {
                class_id: [] for class_id in range(1, NUM_CLASSES)
            }
            for position in range(len(self.semantic)):
                counts = np.bincount(
                    np.asarray(self.semantic[position]).ravel(),
                    minlength=NUM_CLASSES,
                )
                for class_id in range(1, NUM_CLASSES):
                    if counts[class_id]:
                        present[class_id].append(position)
            self._donor_index = {
                class_id: np.asarray(positions, dtype=np.int64)
                for class_id, positions in present.items()
                if positions
            }
            print(
                "Copy-paste donor pool: "
                + ", ".join(
                    f"{CLASS_NAMES[class_id]}={len(positions)}"
                    for class_id, positions in sorted(
                        self._donor_index.items()
                    )
                ),
                flush=True,
            )
        if not self._donor_index:
            return int(rng.integers(len(self.images)))
        class_id = int(rng.choice(sorted(self._donor_index)))
        return int(rng.choice(self._donor_index[class_id]))

    def augmentation_mode_for_sample(self, sample_index: int) -> int:
        """Give every image all 95 V5 modes once per 95 epochs."""
        return (
            self.epoch_index + int(sample_index)
        ) % AUGMENTATION_CYCLE_LENGTH

    def augmentation_rng_for_sample(
        self, sample_index: int, salt: int = 0
    ) -> np.random.Generator:
        """Return a deterministic per-sample generator.

        ``salt`` selects an independent stream so that a new augmentation can
        be added without shifting the draws of an existing one.
        """
        sequence = np.random.SeedSequence(
            [
                SEED,
                self.seed_offset,
                self.epoch_index,
                int(sample_index),
                int(salt),
            ]
        )
        return np.random.default_rng(sequence)

    def __getitem__(self, batch_index: int):
        start = batch_index * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        n = len(batch_indices)

        image_batch = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        semantic_batch = np.empty((n, IMG_SIZE, IMG_SIZE), dtype=np.int32)
        distance_batch = np.empty(
            (n, INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE, 2), dtype=np.float32
        )
        boundary_batch = np.empty(
            (n, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32
        )

        for position, sample_index in enumerate(batch_indices):
            image = normalize_image(self.images[sample_index])
            semantic = np.asarray(
                self.semantic[sample_index], dtype=np.int32
            ).copy()
            instance = np.asarray(
                self.instance[sample_index], dtype=np.int32
            ).copy()

            # Mirror-side first, from its own stream, so every later
            # augmentation (the 95-mode cycle included) acts on it. The side is
            # the same for the whole epoch: left, top, repeat.
            if self.training and USE_MIRROR_SIDE_AUGMENTATION:
                mirror_rng = self.augmentation_rng_for_sample(
                    int(sample_index), salt=4
                )
                if mirror_rng.random() < float(MIRROR_SIDE_PROBABILITY):
                    image, semantic, instance = mirror_side_sample(
                        image,
                        semantic,
                        instance,
                        MIRROR_SIDES[
                            self.epoch_index % len(MIRROR_SIDES)
                        ],
                    )

            if self.training and USE_ONLINE_AUGMENTATION:
                mode = self.augmentation_mode_for_sample(
                    int(sample_index)
                )
            else:
                mode = 0

            if mode != 0:
                image, semantic, instance = augment_training_sample(
                    image=image,
                    semantic=semantic,
                    instance=instance,
                    mode=mode,
                    rng=self.augmentation_rng_for_sample(
                        int(sample_index)
                    ),
                )

            # Dihedral symmetry is orthogonal to the 95-mode cycle, so it is
            # applied to the original mode too, from its own RNG stream, and
            # before the targets are generated from the instance map.
            if self.training and USE_DIHEDRAL_AUGMENTATION:
                dihedral_rng = self.augmentation_rng_for_sample(
                    int(sample_index), salt=1
                )
                image, semantic, instance = apply_dihedral_transform(
                    image,
                    semantic,
                    instance,
                    int(dihedral_rng.integers(8)),
                )

            # Mosaic runs before copy-paste so a pasted object lands at the
            # mosaic's scale rather than at four times it, and after the 95-mode
            # cycle and the dihedral stream so each quadrant is independently
            # augmented -- which is what makes one mosaic four samples rather
            # than one sample repeated.
            if (
                self.training
                and USE_MOSAIC_AUGMENTATION
                and MOSAIC_PROBABILITY > 0.0
            ):
                mosaic_rng = self.augmentation_rng_for_sample(
                    int(sample_index), salt=3
                )
                if mosaic_rng.random() < MOSAIC_PROBABILITY:
                    quadrants = [(image, semantic, instance)]
                    for _ in range(3):
                        donor = self.donor_sample_index(mosaic_rng)
                        donor_image = normalize_image(self.images[donor])
                        donor_semantic = np.asarray(
                            self.semantic[donor], dtype=np.int32
                        )
                        donor_instance = np.asarray(
                            self.instance[donor], dtype=np.int32
                        )
                        donor_mode = int(
                            mosaic_rng.integers(1, AUGMENTATION_CYCLE_LENGTH)
                        )
                        quadrants.append(
                            augment_training_sample(
                                image=donor_image,
                                semantic=donor_semantic,
                                instance=donor_instance,
                                mode=donor_mode,
                                rng=mosaic_rng,
                            )
                        )
                    image, semantic, instance = mosaic_samples(quadrants)

            if self.training and USE_COPY_PASTE_AUGMENTATION:
                copy_paste_rng = self.augmentation_rng_for_sample(
                    int(sample_index), salt=2
                )
                if copy_paste_rng.random() < COPY_PASTE_PROBABILITY:
                    donor = self.donor_sample_index(copy_paste_rng)
                    image, semantic, instance = copy_paste_objects(
                        image,
                        semantic,
                        instance,
                        normalize_image(self.images[donor]),
                        np.asarray(self.semantic[donor], dtype=np.int32),
                        np.asarray(self.instance[donor], dtype=np.int32),
                        copy_paste_rng,
                    )

            semantic, distance, boundary = build_all_targets(
                semantic, instance
            )
            image_batch[position] = image
            semantic_batch[position] = semantic
            distance_batch[position] = distance
            boundary_batch[position] = boundary

        return image_batch, {
            "semantic": semantic_batch,
            "inner_distance": distance_batch,
            "boundary": boundary_batch,
        }


class AugmentationEpochSyncCallback(tf.keras.callbacks.Callback):
    """Keep deterministic online augmentation correct after crash recovery."""

    def __init__(self, sequence: InstanceArraySequence):
        super().__init__()
        self.sequence = sequence

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        del logs
        self.sequence.epoch_index = int(epoch)


# =============================================================================
# CLASS COUNTS AND WEIGHTS
# =============================================================================

def semantic_class_pixel_counts(
    semantic_labels: np.ndarray,
) -> np.ndarray:
    """Count training pixels belonging to every semantic class."""

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for start in range(0, len(semantic_labels), 32):
        chunk = np.asarray(
            semantic_labels[start:start + 32],
            dtype=np.uint8,
        )

        batch_counts = np.bincount(
            chunk.ravel(),
            minlength=NUM_CLASSES,
        )

        if len(batch_counts) != NUM_CLASSES:
            raise ValueError(
                f"Semantic labels must contain only classes "
                f"0 to {NUM_CLASSES - 1}. "
                f"Observed bincount length: {len(batch_counts)}"
            )

        counts += batch_counts

    return counts


def compute_class_weights(
    semantic_labels: np.ndarray,
    background_class: int = 0,
    power: float = 0.5,
    maximum_weight: float = 8.0,
) -> np.ndarray:
    """
    Calculate inverse-square-root class weights.

    Class weights are used by the focal component.
    Dice and Lovasz provide additional class balancing.
    """

    counts = semantic_class_pixel_counts(semantic_labels).astype(
        np.float64
    )

    if np.any(counts <= 0):
        raise ValueError(
            f"Every training class must contain pixels: "
            f"{counts.astype(np.int64).tolist()}"
        )

    if not 0 <= background_class < NUM_CLASSES:
        raise ValueError(
            f"Invalid background class: {background_class}"
        )

    if not 0.0 < power <= 1.0:
        raise ValueError("power must be between 0 and 1")

    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least 1")

    frequencies = counts / counts.sum()

    weights = (
        frequencies[background_class] / frequencies
    ) ** power

    weights = np.minimum(weights, maximum_weight)
    weights = weights.astype(np.float32)

    print(
        "Training class counts:",
        counts.astype(np.int64).tolist(),
    )
    print(
        "Training class frequencies:",
        np.round(frequencies, 6).tolist(),
    )
    print(
        "Training class weights:",
        np.round(weights, 3).tolist(),
    )

    return weights


# =============================================================================
# LOVASZ-SOFTMAX FUNCTIONS
# =============================================================================

def lovasz_gradient(
    sorted_ground_truth: tf.Tensor,
) -> tf.Tensor:
    """
    Calculate the Lovasz extension gradient.

    sorted_ground_truth must contain binary values sorted using
    descending prediction errors.
    """

    sorted_ground_truth = tf.cast(
        sorted_ground_truth,
        tf.float32,
    )

    total_positive = tf.reduce_sum(sorted_ground_truth)

    intersection = (
        total_positive
        - tf.cumsum(sorted_ground_truth)
    )

    union = (
        total_positive
        + tf.cumsum(1.0 - sorted_ground_truth)
    )

    jaccard = 1.0 - intersection / tf.maximum(
        union,
        1e-6,
    )

    # Convert cumulative Jaccard values into discrete gradients.
    gradient = tf.concat(
        [
            jaccard[:1],
            jaccard[1:] - jaccard[:-1],
        ],
        axis=0,
    )

    return gradient


def foreground_lovasz_softmax_loss(
    y_true: tf.Tensor,
    probabilities: tf.Tensor,
) -> tf.Tensor:
    """
    Calculate Lovasz-Softmax over foreground classes 1..4.

    Background is excluded because the model checkpoint is selected
    using foreground mean IoU.
    """

    y_true = tf.cast(y_true, tf.int32)
    probabilities = tf.cast(probabilities, tf.float32)

    flat_labels = tf.reshape(y_true, [-1])

    flat_probabilities = tf.reshape(
        probabilities,
        [-1, NUM_CLASSES],
    )

    # The Lovasz extension needs the errors of one class sorted, which costs a
    # full sort of every pixel for every present class. A uniform sample
    # without replacement estimates the same surrogate and turns four
    # 512x512 sorts per step into four 65k sorts.
    maximum_pixels = int(LOVASZ_MAX_PIXELS)

    if maximum_pixels > 0:
        # Draw the sample directly instead of permuting the whole index range:
        # tf.random.shuffle on N elements is itself a sort of N, so the earlier
        # version paid the 524,288-element sort it was written to avoid.
        # Sampling with replacement costs O(k) and estimates the same
        # surrogate; at k=65,536 out of 524,288 roughly 94% of the draws are
        # distinct, and repeats only reweight, never bias.
        #
        # This is unconditional and the result has a *static* shape. A
        # tf.cond on "is the batch bigger than the sample" would make the
        # downstream sort length dynamic, and XLA cannot compile a top_k whose
        # k is only known at run time. One image is already 262,144 pixels, so
        # the branch this replaces was never taken during training anyway.
        kept = tf.random.uniform(
            [maximum_pixels],
            maxval=tf.shape(flat_labels)[0],
            dtype=tf.int32,
        )
        flat_labels = tf.gather(flat_labels, kept)
        flat_probabilities = tf.gather(flat_probabilities, kept)
        flat_labels = tf.ensure_shape(flat_labels, [maximum_pixels])
        flat_probabilities = tf.ensure_shape(
            flat_probabilities, [maximum_pixels, NUM_CLASSES]
        )

    class_losses = []
    present_classes = []

    for class_id in range(1, NUM_CLASSES):
        foreground = tf.cast(
            tf.equal(flat_labels, class_id),
            tf.float32,
        )

        class_probability = flat_probabilities[:, class_id]

        class_is_present = tf.reduce_any(
            foreground > 0.0
        )

        def calculate_present_class_loss():
            errors = tf.abs(
                foreground - class_probability
            )

            # A python int when the sample above fixed the length, so the sort
            # is statically shaped and XLA-compilable; tf.shape otherwise.
            sort_length = (
                errors.shape[0]
                if errors.shape[0] is not None
                else tf.shape(errors)[0]
            )
            if LOVASZ_SORT_DEVICE:
                # Placement only; the maths is identical either way.
                with tf.device(LOVASZ_SORT_DEVICE):
                    sorted_errors, permutation = tf.math.top_k(
                        errors,
                        k=sort_length,
                        sorted=True,
                    )
            else:
                sorted_errors, permutation = tf.math.top_k(
                    errors,
                    k=sort_length,
                    sorted=True,
                )

            sorted_foreground = tf.gather(
                foreground,
                permutation,
            )

            gradient = lovasz_gradient(
                sorted_foreground
            )

            return tf.reduce_sum(
                sorted_errors * gradient
            )

        class_loss = tf.cond(
            class_is_present,
            calculate_present_class_loss,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        present = tf.cast(
            class_is_present,
            tf.float32,
        )

        class_losses.append(class_loss)
        present_classes.append(present)

    total_loss = tf.add_n(class_losses)
    number_of_present_classes = tf.add_n(present_classes)

    return total_loss / (
        number_of_present_classes + 1e-6
    )


# =============================================================================
# COMPLETE SEMANTIC LOSS
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class SemanticFocalDiceLovaszFromLogits(
    tf.keras.losses.Loss
):
    """
    Combined semantic loss:

        45% class-weighted focal loss
        30% foreground Dice loss
        25% foreground Lovasz-Softmax loss
    """

    def __init__(
        self,
        class_weights,
        gamma: float = 2.0,
        focal_weight: float = 0.45,
        dice_weight: float = 0.30,
        lovasz_weight: float = 0.25,
        name: str = "semantic_focal_dice_lovasz",
    ):
        super().__init__(name=name)

        self.class_weights = [
            float(value) for value in class_weights
        ]

        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.lovasz_weight = float(lovasz_weight)

        total_weight = (
            self.focal_weight
            + self.dice_weight
            + self.lovasz_weight
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "focal_weight + dice_weight + lovasz_weight "
                f"must equal 1.0, received {total_weight}"
            )

        if len(self.class_weights) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} class weights, "
                f"received {len(self.class_weights)}"
            )

    def call(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
    ) -> tf.Tensor:

        y_true = tf.cast(y_true, tf.int32)

        if y_true.shape.rank == logits.shape.rank:
            y_true = tf.squeeze(
                y_true,
                axis=-1,
            )

        # Loss calculations remain float32 when mixed precision is used.
        logits = tf.cast(logits, tf.float32)

        probabilities = tf.nn.softmax(
            logits,
            axis=-1,
        )

        # ---------------------------------------------------------
        # 1. CLASS-WEIGHTED FOCAL LOSS
        # ---------------------------------------------------------

        cross_entropy = (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true,
                logits=logits,
            )
        )

        true_probability = tf.exp(-cross_entropy)

        class_weights_tensor = tf.constant(
            self.class_weights,
            dtype=tf.float32,
        )

        pixel_weights = tf.gather(
            class_weights_tensor,
            y_true,
        )

        focal_pixels = (
            pixel_weights
            * tf.pow(
                1.0 - true_probability,
                self.gamma,
            )
            * cross_entropy
        )

        focal_loss = tf.reduce_sum(
            focal_pixels
        ) / (
            tf.reduce_sum(pixel_weights) + 1e-6
        )

        # ---------------------------------------------------------
        # 2. FOREGROUND DICE LOSS
        # ---------------------------------------------------------

        one_hot = tf.one_hot(
            y_true,
            depth=NUM_CLASSES,
            dtype=tf.float32,
        )

        target_foreground = one_hot[..., 1:]
        predicted_foreground = probabilities[..., 1:]

        reduction_axes = tf.range(
            tf.rank(target_foreground) - 1
        )

        intersection = tf.reduce_sum(
            target_foreground * predicted_foreground,
            axis=reduction_axes,
        )

        target_sum = tf.reduce_sum(
            target_foreground,
            axis=reduction_axes,
        )

        predicted_sum = tf.reduce_sum(
            predicted_foreground,
            axis=reduction_axes,
        )

        dice_score = (
            2.0 * intersection + 1e-6
        ) / (
            target_sum + predicted_sum + 1e-6
        )

        present_classes = tf.cast(
            target_sum > 0.0,
            tf.float32,
        )

        dice_loss = tf.reduce_sum(
            (1.0 - dice_score) * present_classes
        ) / (
            tf.reduce_sum(present_classes) + 1e-6
        )

        # ---------------------------------------------------------
        # 3. FOREGROUND LOVASZ-SOFTMAX LOSS
        # ---------------------------------------------------------

        lovasz_loss = foreground_lovasz_softmax_loss(
            y_true,
            probabilities,
        )

        # ---------------------------------------------------------
        # FINAL COMBINED LOSS
        # ---------------------------------------------------------

        return (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.lovasz_weight * lovasz_loss
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "class_weights": self.class_weights,
                "gamma": self.gamma,
                "focal_weight": self.focal_weight,
                "dice_weight": self.dice_weight,
                "lovasz_weight": self.lovasz_weight,
            }
        )

        return config

@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class CenterNetFocalFromLogits(tf.keras.losses.Loss):
    """
    Numerically stable CenterNet Gaussian focal loss.

    Improvements:
    - Normalizes each image separately by its number of centres.
    - Uses stable softplus calculations directly from logits.
    - Handles empty images without ignoring their false detections.
    - Safely handles centre targets slightly above or below 1.0.
    - Compatible with mixed-precision training.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        positive_tolerance: float = 1e-4,
        empty_image_weight: float = 1.0,
        name: str = "centernet_focal",
    ):
        super().__init__(name=name)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.positive_tolerance = float(positive_tolerance)
        self.empty_image_weight = float(empty_image_weight)

        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")

        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")

        if not 0.0 < self.positive_tolerance < 0.1:
            raise ValueError(
                "positive_tolerance must be between 0 and 0.1"
            )

        if self.empty_image_weight < 0.0:
            raise ValueError(
                "empty_image_weight must be non-negative"
            )

    def call(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
    ) -> tf.Tensor:

        # Always calculate the loss in float32, even when the network
        # uses mixed_float16.
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)

        # Centre heatmaps must be probabilities in the range 0..1.
        # Clipping also prevents an accidental target such as 1.6 from
        # being incorrectly treated as a negative centre.
        y_true = tf.clip_by_value(
            y_true,
            0.0,
            1.0,
        )

        probabilities = tf.nn.sigmoid(logits)

        # Values sufficiently close to 1 are true centre positions.
        positives = tf.cast(
            y_true >= (1.0 - self.positive_tolerance),
            tf.float32,
        )

        negatives = 1.0 - positives

        # Gaussian pixels surrounding a centre are treated as reduced
        # negatives. Pixels close to the centre receive a smaller
        # negative penalty.
        negative_weights = tf.pow(
            1.0 - y_true,
            self.beta,
        )

        # Numerically stable equivalents:
        #
        # -log(sigmoid(logit))     = softplus(-logit)
        # -log(1-sigmoid(logit))   = softplus(logit)
        positive_cross_entropy = tf.nn.softplus(
            -logits
        )

        negative_cross_entropy = tf.nn.softplus(
            logits
        )

        positive_loss = (
            positive_cross_entropy
            * tf.pow(
                1.0 - probabilities,
                self.alpha,
            )
            * positives
        )

        negative_loss = (
            negative_cross_entropy
            * tf.pow(
                probabilities,
                self.alpha,
            )
            * negative_weights
            * negatives
        )

        # Reduce height, width and class channels, but preserve the
        # batch dimension. CenterNet normalizes by the number of
        # keypoints in each individual image.
        reduction_axes = tf.range(
            1,
            tf.rank(logits),
        )

        positive_loss_per_image = tf.reduce_sum(
            positive_loss,
            axis=reduction_axes,
        )

        negative_loss_per_image = tf.reduce_sum(
            negative_loss,
            axis=reduction_axes,
        )

        positive_count_per_image = tf.reduce_sum(
            positives,
            axis=reduction_axes,
        )

        # Normal CenterNet loss for images containing instances.
        loss_with_objects = (
            positive_loss_per_image
            + negative_loss_per_image
        ) / tf.maximum(
            positive_count_per_image,
            1.0,
        )

        # For an image with no objects, train the model to suppress all
        # false centre predictions. Do not use reduce_mean here because
        # it would make empty-image supervision almost zero.
        loss_without_objects = (
            self.empty_image_weight
            * negative_loss_per_image
        )

        loss_per_image = tf.where(
            positive_count_per_image > 0.0,
            loss_with_objects,
            loss_without_objects,
        )

        return tf.reduce_mean(loss_per_image)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "positive_tolerance": self.positive_tolerance,
                "empty_image_weight": self.empty_image_weight,
            }
        )

        return config

@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class MaskedInnerDistanceLoss(tf.keras.losses.Loss):
    """Masked Huber loss on the per-instance normalised inner distance.

        y_true[..., 0] = normalised inner distance, 0 at the rim to 1 inside
        y_true[..., 1] = foreground validity mask
        y_pred[..., 0] = predicted normalised inner distance

    Replaces `MaskedOffsetPixelHuberLoss`. That loss had to convert its Huber
    transition into normalised coordinates separately for x and y, because the
    offset target was measured in head widths and heights -- a quantity whose
    meaning changes with the object's size. This target is already
    dimensionless and identically scaled for every object, so the transition is
    a plain constant and needs no geometry at all.

    The head that matters most is the boundary region, where the field is near
    zero and neighbouring objects have to be told apart. `rim_weight` lets
    those pixels count for more than the flat interior, which is otherwise the
    easy majority of every object and would dominate a uniform mean.
    """

    def __init__(
        self,
        delta: float = 0.10,
        rim_weight: float = 2.0,
        name: str = "masked_inner_distance",
    ):
        super().__init__(name=name)
        self.delta = float(delta)
        self.rim_weight = float(rim_weight)
        if not 0.0 < self.delta <= 1.0:
            raise ValueError("delta must lie in (0,1]")
        if self.rim_weight < 1.0:
            raise ValueError(
                "rim_weight must be at least 1.0; weighting the interior above "
                "the rim would train away the only signal that separates "
                "touching objects."
            )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape[-1] is not None and y_true.shape[-1] != 2:
            raise ValueError(
                "Inner-distance target must contain two channels: distance "
                "and valid mask."
            )
        if y_pred.shape[-1] is not None and y_pred.shape[-1] != 1:
            raise ValueError(
                "Inner-distance prediction must contain one channel."
            )

        target = y_true[..., 0:1]
        valid = tf.clip_by_value(y_true[..., 1:2], 0.0, 1.0)

        absolute_error = tf.abs(target - y_pred)
        quadratic = tf.minimum(absolute_error, self.delta)
        linear = absolute_error - quadratic
        elementwise = 0.5 * tf.square(quadratic) + self.delta * linear

        # Weight toward the rim: 1.0 at the innermost pixel rising to
        # `rim_weight` at the boundary. Linear in the target itself, so it
        # needs no second field to carry it.
        weight = 1.0 + (self.rim_weight - 1.0) * (1.0 - target)
        weighted = elementwise * weight * valid

        # Per-image mean over valid pixels only. An image whose foreground is
        # empty contributes zero rather than a division by zero.
        axes = [1, 2, 3]
        valid_total = tf.reduce_sum(valid * weight, axis=axes)
        loss_total = tf.reduce_sum(weighted, axis=axes)
        return tf.math.divide_no_nan(loss_total, valid_total)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {"delta": self.delta, "rim_weight": self.rim_weight}
        )
        return config


@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class MaskedOffsetPixelHuberLoss(tf.keras.losses.Loss):
    """
    Masked, per-image Huber loss for normalized pixel-to-centre offsets.

    The target format must be:

        y_true[..., 0] = normalized x offset
        y_true[..., 1] = normalized y offset
        y_true[..., 2] = valid-instance mask

    The model prediction contains:

        y_pred[..., 0] = normalized x offset
        y_pred[..., 1] = normalized y offset
    """

    def __init__(
        self,
        delta_pixels: float = 1.0,
        name: str = "masked_offset_pixel_huber",
    ):
        super().__init__(name=name)

        self.delta_pixels = float(delta_pixels)

        if self.delta_pixels <= 0.0:
            raise ValueError(
                "delta_pixels must be greater than zero"
            )

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:

        # Keep loss calculations in float32 during mixed-precision
        # training.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if (
            y_true.shape[-1] is not None
            and y_true.shape[-1] != 3
        ):
            raise ValueError(
                "Offset target must contain three channels: "
                "x-offset, y-offset and valid mask."
            )

        if (
            y_pred.shape[-1] is not None
            and y_pred.shape[-1] != 2
        ):
            raise ValueError(
                "Offset prediction must contain two channels."
            )

        target_offsets = y_true[..., :2]

        # This remains compatible with the existing binary 0/1 mask.
        valid_mask = tf.clip_by_value(
            y_true[..., 2:3],
            0.0,
            1.0,
        )

        # Determine the actual output-head dimensions dynamically.
        head_height = tf.cast(
            tf.shape(y_pred)[1],
            tf.float32,
        )

        head_width = tf.cast(
            tf.shape(y_pred)[2],
            tf.float32,
        )

        head_height = tf.maximum(head_height, 1.0)
        head_width = tf.maximum(head_width, 1.0)

        # Convert the pixel-space Huber transition into normalized
        # coordinate units independently for x and y.
        delta_x = self.delta_pixels / head_width
        delta_y = self.delta_pixels / head_height

        delta_per_coordinate = tf.reshape(
            tf.stack([delta_x, delta_y]),
            [1, 1, 1, 2],
        )

        absolute_error = tf.abs(
            target_offsets - y_pred
        )

        quadratic_error = tf.minimum(
            absolute_error,
            delta_per_coordinate,
        )

        linear_error = (
            absolute_error - quadratic_error
        )

        # Smooth-L1/Huber:
        #
        # 0.5 * error² / delta       when error <= delta
        # error - 0.5 * delta        when error > delta
        huber_loss = (
            0.5
            * tf.square(quadratic_error)
            / delta_per_coordinate
            + linear_error
        )

        masked_loss = huber_loss * valid_mask

        # Sum height, width and both coordinate channels while
        # preserving the batch dimension.
        loss_sum_per_image = tf.reduce_sum(
            masked_loss,
            axis=[1, 2, 3],
        )

        valid_pixels_per_image = tf.reduce_sum(
            valid_mask,
            axis=[1, 2, 3],
        )

        # Two supervised coordinates exist for every valid pixel.
        supervised_coordinates = (
            2.0 * valid_pixels_per_image
        )

        loss_per_image = tf.math.divide_no_nan(
            loss_sum_per_image,
            supervised_coordinates,
        )

        # Images without instances have no meaningful offset target.
        # Ignore them without allowing them to dilute the loss from
        # non-empty images.
        nonempty_images = tf.cast(
            valid_pixels_per_image > 0.0,
            tf.float32,
        )

        return tf.math.divide_no_nan(
            tf.reduce_sum(
                loss_per_image * nonempty_images
            ),
            tf.reduce_sum(nonempty_images),
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "delta_pixels": self.delta_pixels,
            }
        )

        return config

@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class BoundaryBCEDiceFromLogits(tf.keras.losses.Loss):
    """
    Boundary loss combining:

    - Class-balanced focal BCE for difficult boundary pixels
    - Per-image Dice loss for boundary overlap
    - Correct handling of images without boundaries
    - Stable float32 calculations during mixed precision
    """

    def __init__(
        self,
        positive_weight: float = BOUNDARY_POSITIVE_WEIGHT,
        gamma: float = 2.0,
        focal_weight: float = 0.50,
        dice_weight: float = 0.50,
        smooth: float = 1.0,
        name: str = "boundary_focal_bce_dice",
    ):
        super().__init__(name=name)

        self.positive_weight = float(positive_weight)
        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

        if self.positive_weight <= 0.0:
            raise ValueError(
                "positive_weight must be greater than zero"
            )

        if self.gamma < 0.0:
            raise ValueError(
                "gamma must be non-negative"
            )

        if self.focal_weight < 0.0:
            raise ValueError(
                "focal_weight must be non-negative"
            )

        if self.dice_weight < 0.0:
            raise ValueError(
                "dice_weight must be non-negative"
            )

        total_weight = (
            self.focal_weight + self.dice_weight
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "focal_weight + dice_weight must equal 1.0, "
                f"received {total_weight}"
            )

        if self.smooth <= 0.0:
            raise ValueError(
                "smooth must be greater than zero"
            )

    def call(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
    ) -> tf.Tensor:

        # Keep loss calculations stable during mixed-precision
        # training.
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)

        if (
            y_true.shape[-1] is not None
            and y_true.shape[-1] != 1
        ):
            raise ValueError(
                "Boundary target must contain one channel."
            )

        if (
            logits.shape[-1] is not None
            and logits.shape[-1] != 1
        ):
            raise ValueError(
                "Boundary prediction must contain one channel."
            )

        y_true = tf.clip_by_value(
            y_true,
            0.0,
            1.0,
        )

        probabilities = tf.nn.sigmoid(logits)

        reduction_axes = tf.range(
            1,
            tf.rank(logits),
        )

        # ---------------------------------------------------------
        # 1. WEIGHTED FOCAL BINARY CROSS-ENTROPY
        # ---------------------------------------------------------

        weighted_bce = (
            tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true,
                logits=logits,
                pos_weight=self.positive_weight,
            )
        )

        # Probability assigned to the correct binary class.
        correct_class_probability = (
            y_true * probabilities
            + (1.0 - y_true) * (1.0 - probabilities)
        )

        focal_modulation = tf.pow(
            1.0 - correct_class_probability,
            self.gamma,
        )

        focal_bce = (
            weighted_bce * focal_modulation
        )

        # Normalize by the effective pixel weights so the loss scale
        # remains stable when the boundary proportion changes.
        effective_pixel_weights = (
            1.0
            + (self.positive_weight - 1.0) * y_true
        )

        focal_loss_per_image = tf.math.divide_no_nan(
            tf.reduce_sum(
                focal_bce,
                axis=reduction_axes,
            ),
            tf.reduce_sum(
                effective_pixel_weights,
                axis=reduction_axes,
            ),
        )

        # ---------------------------------------------------------
        # 2. PER-IMAGE BOUNDARY DICE LOSS
        # ---------------------------------------------------------

        intersection = tf.reduce_sum(
            y_true * probabilities,
            axis=reduction_axes,
        )

        target_sum = tf.reduce_sum(
            y_true,
            axis=reduction_axes,
        )

        prediction_sum = tf.reduce_sum(
            probabilities,
            axis=reduction_axes,
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            target_sum
            + prediction_sum
            + self.smooth
        )

        normal_dice_loss = 1.0 - dice_score

        # Ordinary Dice becomes nearly constant when the target has
        # no boundary. In that case, directly penalize predicted
        # boundary probability.
        empty_boundary_loss = tf.reduce_mean(
            probabilities,
            axis=reduction_axes,
        )

        dice_loss_per_image = tf.where(
            target_sum > 0.0,
            normal_dice_loss,
            empty_boundary_loss,
        )

        # ---------------------------------------------------------
        # 3. FINAL COMBINED BOUNDARY LOSS
        # ---------------------------------------------------------

        total_loss_per_image = (
            self.focal_weight * focal_loss_per_image
            + self.dice_weight * dice_loss_per_image
        )

        return tf.reduce_mean(
            total_loss_per_image
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "positive_weight": self.positive_weight,
                "gamma": self.gamma,
                "focal_weight": self.focal_weight,
                "dice_weight": self.dice_weight,
                "smooth": self.smooth,
            }
        )

        return config
# =============================================================================
# 8. LOGIT-AWARE METRICS
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class SparseMeanIoUFromLogits(
    tf.keras.metrics.MeanIoU
):
    """
    Accurate mean IoU for sparse semantic labels and dense logits.

    Features:
    - Converts logits to class IDs internally.
    - Uses float64 confusion-matrix accumulation.
    - Supports an optional ignored/void label.
    - Supports sample weights.
    - Validates label and logit shapes.
    - Fully serializable with saved Keras models.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_class: int | None = None,
        axis: int = -1,
        name: str = "mean_iou",
        dtype=tf.float64,
        **kwargs,
    ):
        # Remove values that may appear when loading an older saved
        # metric configuration. This class defines them explicitly.
        kwargs.pop("sparse_y_true", None)
        kwargs.pop("sparse_y_pred", None)
        kwargs.pop("target_class_ids", None)

        self.configured_num_classes = int(num_classes)
        self.configured_ignore_class = (
            None
            if ignore_class is None
            else int(ignore_class)
        )
        self.logit_axis = int(axis)

        if self.configured_num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        super().__init__(
            num_classes=self.configured_num_classes,
            name=name,
            dtype=dtype,
            ignore_class=self.configured_ignore_class,
            sparse_y_true=True,
            # Predictions are dense logits. MeanIoU will apply
            # argmax using the configured class axis.
            sparse_y_pred=False,
            axis=self.logit_axis,
            **kwargs,
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
        sample_weight=None,
    ):
        y_true = tf.convert_to_tensor(y_true)
        logits = tf.convert_to_tensor(logits)

        # Semantic logits should contain one channel per class.
        logit_channels = logits.shape[self.logit_axis]

        if (
            logit_channels is not None
            and int(logit_channels)
            != self.configured_num_classes
        ):
            raise ValueError(
                "Semantic logit channel count does not match "
                f"num_classes: received {logit_channels}, "
                f"expected {self.configured_num_classes}."
            )

        # Accept both:
        #
        # [batch, height, width]
        # [batch, height, width, 1]
        labels_have_channel = (
            y_true.shape.rank is not None
            and logits.shape.rank is not None
            and y_true.shape.rank == logits.shape.rank
        )

        if labels_have_channel:
            if (
                y_true.shape[-1] is not None
                and y_true.shape[-1] != 1
            ):
                raise ValueError(
                    "Sparse semantic labels must have either no "
                    "channel dimension or one final channel."
                )

            y_true = tf.squeeze(
                y_true,
                axis=-1,
            )

            # If sample weights also contain a final singleton
            # channel, remove it to match the sparse labels.
            if sample_weight is not None:
                sample_weight = tf.convert_to_tensor(
                    sample_weight
                )

                if (
                    sample_weight.shape.rank
                    == logits.shape.rank
                ):
                    if (
                        sample_weight.shape[-1] is not None
                        and sample_weight.shape[-1] != 1
                    ):
                        raise ValueError(
                            "Pixel sample weights must have one "
                            "final channel."
                        )

                    sample_weight = tf.squeeze(
                        sample_weight,
                        axis=-1,
                    )

        y_true = tf.cast(
            y_true,
            tf.int32,
        )

        # MeanIoU applies argmax because sparse_y_pred=False.
        # Casting logits to float32 avoids mixed-precision rounding
        # affecting class selection.
        logits = tf.cast(
            logits,
            tf.float32,
        )

        if sample_weight is not None:
            sample_weight = tf.cast(
                sample_weight,
                tf.float64,
            )

        return super().update_state(
            y_true,
            logits,
            sample_weight=sample_weight,
        )

    def get_config(self):
        # Use a minimal configuration so sparse_y_true and
        # sparse_y_pred are not duplicated during deserialization.
        return {
            "num_classes": self.configured_num_classes,
            "ignore_class": self.configured_ignore_class,
            "axis": self.logit_axis,
            "name": self.name,
            "dtype": self.dtype,
        }
    
@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class ForegroundMeanIoUFromLogits(
    tf.keras.metrics.IoU
):
    """
    Mean IoU over foreground classes only.

    Background IoU is excluded from the final average, but background
    pixels remain in the confusion matrix. Therefore, foreground
    predictions made on background pixels still count as false
    positives.

    Sparse labels:
        [batch, height, width]
        or
        [batch, height, width, 1]

    Predictions:
        [batch, height, width, num_classes] logits
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        background_class: int = 0,
        ignore_class: int | None = None,
        axis: int = -1,
        name: str = "foreground_miou",
        dtype=tf.float64,
        **kwargs,
    ):
        # Remove inherited values that may appear when loading an
        # older saved configuration.
        kwargs.pop("target_class_ids", None)
        kwargs.pop("sparse_y_true", None)
        kwargs.pop("sparse_y_pred", None)

        self.configured_num_classes = int(num_classes)
        self.background_class = int(background_class)
        self.configured_ignore_class = (
            None
            if ignore_class is None
            else int(ignore_class)
        )
        self.logit_axis = int(axis)

        if self.configured_num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        if not (
            0
            <= self.background_class
            < self.configured_num_classes
        ):
            raise ValueError(
                f"Invalid background class: "
                f"{self.background_class}"
            )

        if (
            self.configured_ignore_class
            == self.background_class
        ):
            raise ValueError(
                "Background must not be ignored. Its IoU is excluded "
                "from the average, but its pixels are required to "
                "count foreground false positives correctly."
            )

        self.foreground_class_ids = tuple(
            class_id
            for class_id in range(
                self.configured_num_classes
            )
            if class_id != self.background_class
        )

        super().__init__(
            num_classes=self.configured_num_classes,
            target_class_ids=list(
                self.foreground_class_ids
            ),
            name=name,
            dtype=dtype,
            ignore_class=self.configured_ignore_class,
            sparse_y_true=True,
            # Predictions are dense semantic logits.
            sparse_y_pred=False,
            axis=self.logit_axis,
            **kwargs,
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
        sample_weight=None,
    ):
        y_true = tf.convert_to_tensor(y_true)
        logits = tf.convert_to_tensor(logits)

        logit_channels = logits.shape[
            self.logit_axis
        ]

        if (
            logit_channels is not None
            and int(logit_channels)
            != self.configured_num_classes
        ):
            raise ValueError(
                "Semantic logit channel count does not match "
                f"num_classes: received {logit_channels}, "
                f"expected {self.configured_num_classes}."
            )

        labels_have_channel = (
            y_true.shape.rank is not None
            and logits.shape.rank is not None
            and y_true.shape.rank == logits.shape.rank
        )

        if labels_have_channel:
            if (
                y_true.shape[-1] is not None
                and y_true.shape[-1] != 1
            ):
                raise ValueError(
                    "Sparse labels must have either no channel "
                    "dimension or one final channel."
                )

            y_true = tf.squeeze(
                y_true,
                axis=-1,
            )

            if sample_weight is not None:
                sample_weight = tf.convert_to_tensor(
                    sample_weight
                )

                if (
                    sample_weight.shape.rank
                    == logits.shape.rank
                ):
                    if (
                        sample_weight.shape[-1] is not None
                        and sample_weight.shape[-1] != 1
                    ):
                        raise ValueError(
                            "Pixel sample weights must have one "
                            "final channel."
                        )

                    sample_weight = tf.squeeze(
                        sample_weight,
                        axis=-1,
                    )

        y_true = tf.cast(
            y_true,
            tf.int32,
        )

        # Keras IoU applies argmax because sparse_y_pred=False.
        logits = tf.cast(
            logits,
            tf.float32,
        )

        if sample_weight is not None:
            sample_weight = tf.cast(
                sample_weight,
                tf.float64,
            )

        return super().update_state(
            y_true,
            logits,
            sample_weight=sample_weight,
        )

    def get_config(self):
        return {
            "num_classes": self.configured_num_classes,
            "background_class": self.background_class,
            "ignore_class": self.configured_ignore_class,
            "axis": self.logit_axis,
            "name": self.name,
            "dtype": self.dtype,
        }
@tf.keras.utils.register_keras_serializable(
    package="pcb_instance_v6"
)
class ClassIoUFromLogits(
    tf.keras.metrics.IoU
):
    """
    Exact IoU for one semantic class using sparse labels and logits.

    Background pixels remain part of the confusion matrix, ensuring
    foreground predictions on background count as false positives.
    """

    def __init__(
        self,
        class_id: int,
        num_classes: int = NUM_CLASSES,
        ignore_class: int | None = None,
        axis: int = -1,
        name: str | None = None,
        dtype=tf.float64,
        **kwargs,
    ):
        # Remove inherited values that can appear when loading a
        # previously saved metric configuration.
        kwargs.pop("target_class_ids", None)
        kwargs.pop("sparse_y_true", None)
        kwargs.pop("sparse_y_pred", None)

        self.class_id = int(class_id)
        self.configured_num_classes = int(num_classes)
        self.configured_ignore_class = (
            None
            if ignore_class is None
            else int(ignore_class)
        )
        self.logit_axis = int(axis)

        if self.configured_num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        if not (
            0
            <= self.class_id
            < self.configured_num_classes
        ):
            raise ValueError(
                f"class_id must be between 0 and "
                f"{self.configured_num_classes - 1}, "
                f"received {self.class_id}."
            )

        if (
            self.configured_ignore_class
            == self.class_id
        ):
            raise ValueError(
                f"Class {self.class_id} cannot be measured and "
                "ignored simultaneously."
            )

        # In your dataset class 0 is real background. Ignoring it
        # would remove foreground false positives on background.
        if self.configured_ignore_class == 0:
            raise ValueError(
                "Do not set ignore_class=0. Class 0 is real "
                "background and is required for correct IoU."
            )

        super().__init__(
            num_classes=self.configured_num_classes,
            target_class_ids=[self.class_id],
            name=name or f"iou_class_{self.class_id}",
            dtype=dtype,
            ignore_class=self.configured_ignore_class,
            sparse_y_true=True,
            # Predictions are dense semantic logits. Keras applies
            # argmax along the configured class axis.
            sparse_y_pred=False,
            axis=self.logit_axis,
            **kwargs,
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        logits: tf.Tensor,
        sample_weight=None,
    ):
        y_true = tf.convert_to_tensor(y_true)
        logits = tf.convert_to_tensor(logits)

        logit_channels = logits.shape[
            self.logit_axis
        ]

        if (
            logit_channels is not None
            and int(logit_channels)
            != self.configured_num_classes
        ):
            raise ValueError(
                "Semantic logit channel count does not match "
                f"num_classes: received {logit_channels}, "
                f"expected {self.configured_num_classes}."
            )

        labels_have_channel = (
            y_true.shape.rank is not None
            and logits.shape.rank is not None
            and y_true.shape.rank == logits.shape.rank
        )

        if labels_have_channel:
            if (
                y_true.shape[-1] is not None
                and y_true.shape[-1] != 1
            ):
                raise ValueError(
                    "Sparse semantic labels must have either no "
                    "channel dimension or one final channel."
                )

            y_true = tf.squeeze(
                y_true,
                axis=-1,
            )

            if sample_weight is not None:
                sample_weight = tf.convert_to_tensor(
                    sample_weight
                )

                if (
                    sample_weight.shape.rank
                    == logits.shape.rank
                ):
                    if (
                        sample_weight.shape[-1] is not None
                        and sample_weight.shape[-1] != 1
                    ):
                        raise ValueError(
                            "Pixel sample weights must have one "
                            "final channel."
                        )

                    sample_weight = tf.squeeze(
                        sample_weight,
                        axis=-1,
                    )

        y_true = tf.cast(
            y_true,
            tf.int32,
        )

        # Argmax is performed internally by Keras IoU because
        # sparse_y_pred=False.
        logits = tf.cast(
            logits,
            tf.float32,
        )

        if sample_weight is not None:
            sample_weight = tf.cast(
                sample_weight,
                tf.float64,
            )

        return super().update_state(
            y_true,
            logits,
            sample_weight=sample_weight,
        )

    def get_config(self):
        return {
            "class_id": self.class_id,
            "num_classes": self.configured_num_classes,
            "ignore_class": self.configured_ignore_class,
            "axis": self.logit_axis,
            "name": self.name,
            "dtype": self.dtype,
        }



@tf.keras.utils.register_keras_serializable(package="pcb_instance_v6")
class BoundaryF1FromLogits(tf.keras.metrics.Metric):
    """
    Micro-averaged, tolerance-aware boundary F1.

    A predicted boundary pixel is correct when a target boundary exists within
    `tolerance_pixels`. Recall is calculated symmetrically.

    Inputs:
        y_true: Binary boundary targets, [B, H, W, 1].
        logits: Raw boundary logits, [B, H, W, 1].
    """

    def __init__(
        self,
        probability_threshold: float = 0.5,
        target_threshold: float = 0.5,
        tolerance_pixels: int = 1,
        name: str = "boundary_f1",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if not 0.0 < probability_threshold < 1.0:
            raise ValueError(
                "probability_threshold must be strictly between 0 and 1."
            )
        if not 0.0 <= target_threshold <= 1.0:
            raise ValueError(
                "target_threshold must be between 0 and 1."
            )
        if tolerance_pixels < 0:
            raise ValueError("tolerance_pixels must be >= 0.")

        self.probability_threshold = float(probability_threshold)
        self.target_threshold = float(target_threshold)
        self.tolerance_pixels = int(tolerance_pixels)

        # Comparing logits directly avoids computing sigmoid for every pixel.
        self.logit_threshold = float(
            math.log(
                self.probability_threshold
                / (1.0 - self.probability_threshold)
            )
        )

        self.matched_predictions = self.add_weight(
            name="matched_predictions",
            initializer="zeros",
            dtype=tf.float64,
        )
        self.predicted_count = self.add_weight(
            name="predicted_count",
            initializer="zeros",
            dtype=tf.float64,
        )
        self.matched_targets = self.add_weight(
            name="matched_targets",
            initializer="zeros",
            dtype=tf.float64,
        )
        self.target_count = self.add_weight(
            name="target_count",
            initializer="zeros",
            dtype=tf.float64,
        )

    def _dilate(self, mask):
        """Dilate a binary mask by the configured pixel tolerance."""
        if self.tolerance_pixels == 0:
            return mask

        kernel_size = 2 * self.tolerance_pixels + 1

        dilated = tf.nn.max_pool2d(
            tf.cast(mask, tf.float32),
            ksize=kernel_size,
            strides=1,
            padding="SAME",
            data_format="NHWC",
        )
        return dilated > 0.0

    @staticmethod
    def _prepare_sample_weight(sample_weight, reference):
        """Broadcast common Keras sample-weight shapes to [B,H,W,1]."""
        if sample_weight is None:
            return tf.ones_like(reference, dtype=tf.float64)

        weights = tf.cast(sample_weight, tf.float64)
        rank = weights.shape.rank

        if rank == 1:
            # [B] -> [B,1,1,1]
            weights = weights[:, None, None, None]
        elif rank == 2:
            # [B,1] -> [B,1,1,1]
            if weights.shape[-1] not in (None, 1):
                raise ValueError(
                    "Rank-2 sample_weight must have shape [batch, 1]."
                )
            weights = weights[:, None, None, :]
        elif rank == 3:
            # [B,H,W] -> [B,H,W,1]
            weights = weights[..., None]
        elif rank not in (None, 0, 4):
            raise ValueError(
                "sample_weight must be scalar, [B], [B,1], "
                "[B,H,W], or [B,H,W,1]."
            )

        weights = tf.broadcast_to(weights, tf.shape(reference))
        return tf.maximum(weights, 0.0)

    def update_state(self, y_true, logits, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)

        # Also accept unchannelled [B,H,W] inputs defensively.
        if y_true.shape.rank == 3:
            y_true = y_true[..., None]
        if logits.shape.rank == 3:
            logits = logits[..., None]

        if y_true.shape.rank not in (None, 4):
            raise ValueError(
                f"y_true must have rank 4; received rank {y_true.shape.rank}."
            )
        if logits.shape.rank not in (None, 4):
            raise ValueError(
                f"logits must have rank 4; received rank {logits.shape.rank}."
            )
        if y_true.shape[-1] not in (None, 1):
            raise ValueError("y_true must contain exactly one boundary channel.")
        if logits.shape[-1] not in (None, 1):
            raise ValueError("logits must contain exactly one boundary channel.")

        tf.debugging.assert_equal(
            tf.shape(y_true),
            tf.shape(logits),
            message="Boundary targets and logits must have identical shapes.",
        )

        target = y_true >= self.target_threshold
        predicted = logits >= self.logit_threshold

        # Precision: predicted pixels close to any target boundary.
        dilated_target = self._dilate(target)
        matched_predictions = predicted & dilated_target

        # Recall: target pixels close to any predicted boundary.
        dilated_prediction = self._dilate(predicted)
        matched_targets = target & dilated_prediction

        weights = self._prepare_sample_weight(sample_weight, y_true)

        self.matched_predictions.assign_add(
            tf.reduce_sum(
                tf.cast(matched_predictions, tf.float64) * weights
            )
        )
        self.predicted_count.assign_add(
            tf.reduce_sum(tf.cast(predicted, tf.float64) * weights)
        )
        self.matched_targets.assign_add(
            tf.reduce_sum(tf.cast(matched_targets, tf.float64) * weights)
        )
        self.target_count.assign_add(
            tf.reduce_sum(tf.cast(target, tf.float64) * weights)
        )

    def result(self):
        precision = tf.math.divide_no_nan(
            self.matched_predictions,
            self.predicted_count,
        )
        recall = tf.math.divide_no_nan(
            self.matched_targets,
            self.target_count,
        )

        f1 = tf.math.divide_no_nan(
            2.0 * precision * recall,
            precision + recall,
        )
        return tf.cast(f1, tf.float32)

    def reset_state(self):
        self.matched_predictions.assign(0.0)
        self.predicted_count.assign(0.0)
        self.matched_targets.assign(0.0)
        self.target_count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "probability_threshold": self.probability_threshold,
                "target_threshold": self.target_threshold,
                "tolerance_pixels": self.tolerance_pixels,
            }
        )
        return config


# =============================================================================
# 9. MODEL: EDGE INPUT + YOLO-STYLE ENCODER + FULL-RESOLUTION DECODER
# =============================================================================
"""Final scratch-built Model V7 architecture for PCB instance segmentation.

Paste this section over the existing architecture section in the complete
V6.1 trainer. It deliberately keeps the existing external configuration and
target contract:

    IMG_SIZE
    NUM_CLASSES
    INSTANCE_HEAD_SIZE == IMG_SIZE // 2

Outputs remain:

    semantic : [B, IMG_SIZE, IMG_SIZE, NUM_CLASSES] raw logits
    boundary : [B, IMG_SIZE, IMG_SIZE, 1] raw logits
    center   : [B, INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE, NUM_CLASSES - 1]
    offset   : [B, INSTANCE_HEAD_SIZE, INSTANCE_HEAD_SIZE, 2]

The encoder and every decoder/head layer are initialized from scratch. This
module does not import a Keras Applications model and cannot silently download
or load pretrained weights. It does not alter data loading, target generation,
augmentation, losses, metrics, decoding, or evaluation.
"""



# =============================================================================
# EDGE AND FEATURE-FUSION LAYERS
# =============================================================================


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v6")
class FixedSobelFeatures(layers.Layer):
    """Return normalized Sobel dx, dy, and magnitude feature channels."""

    def call(self, inputs):
        values = tf.cast(inputs, tf.float32)
        gray = tf.image.rgb_to_grayscale(values)
        sobel = tf.image.sobel_edges(gray)

        dy = tf.clip_by_value(sobel[..., 0] / 4.0, -1.0, 1.0)
        dx = tf.clip_by_value(sobel[..., 1] / 4.0, -1.0, 1.0)
        magnitude = tf.sqrt(tf.square(dx) + tf.square(dy) + 1e-6)
        magnitude = tf.clip_by_value(magnitude / math.sqrt(2.0), 0.0, 1.0)

        edges = tf.concat([dx, dy, magnitude], axis=-1)
        return tf.cast(edges, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (3,)


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v6")
class WeightedFeatureFusion(layers.Layer):
    """Fast normalized learnable fusion used by the bidirectional pyramid."""

    def __init__(self, epsilon: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.raw_weights = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                "WeightedFeatureFusion requires at least two input tensors."
            )

        reference = tuple(input_shape[0][1:])
        for index, shape in enumerate(input_shape[1:], start=1):
            if tuple(shape[1:]) != reference:
                raise ValueError(
                    "All fusion tensors must have identical non-batch shapes; "
                    f"input 0 is {reference}, input {index} is {tuple(shape[1:])}."
                )

        self.raw_weights = self.add_weight(
            name="fusion_weights",
            shape=(len(input_shape),),
            initializer="ones",
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.nn.relu(tf.cast(self.raw_weights, tf.float32))
        weights = weights / (tf.reduce_sum(weights) + self.epsilon)

        fused = tf.add_n(
            [
                tf.cast(tensor, tf.float32) * weights[index]
                for index, tensor in enumerate(inputs)
            ]
        )
        return tf.cast(fused, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


# =============================================================================
# CORE CONVOLUTION BLOCKS
# =============================================================================


def group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to `maximum`."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def Conv(
    x,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    name: str | None = None,
):
    """Conv2D + GroupNorm + SiLU, stable with batches of one or two."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.GroupNormalization(
        groups=group_count(filters),
        axis=-1,
        epsilon=1e-5,
        name=None if name is None else f"{name}_gn",
    )(x)
    return layers.Activation(
        "swish",
        name=None if name is None else f"{name}_silu",
    )(x)


def SeparableConv(
    x,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation_rate: int = 1,
    name: str | None = None,
):
    """Memory-efficient spatial refinement for the pyramid and decoders."""
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name=None if name is None else f"{name}_sepconv",
    )(x)
    x = layers.GroupNormalization(
        groups=group_count(filters),
        axis=-1,
        epsilon=1e-5,
        name=None if name is None else f"{name}_gn",
    )(x)
    return layers.Activation(
        "swish",
        name=None if name is None else f"{name}_silu",
    )(x)


def Bottleneck(x, filters: int, name: str | None = None):
    """Residual two-convolution bottleneck."""
    shortcut = x
    y = Conv(x, filters, 3, 1, None if name is None else f"{name}_1")
    y = Conv(y, filters, 3, 1, None if name is None else f"{name}_2")

    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv(
            shortcut,
            filters,
            1,
            1,
            None if name is None else f"{name}_shortcut",
        )

    return layers.Add(name=None if name is None else f"{name}_add")(
        [shortcut, y]
    )


def C3k2(
    x,
    filters: int,
    repeats: int = 2,
    name: str | None = None,
):
    """YOLO-inspired CSP/C3k2 feature block."""
    if filters % 2 != 0:
        raise ValueError(f"C3k2 filters must be even; received {filters}.")

    branch_1 = Conv(
        x,
        filters // 2,
        1,
        1,
        None if name is None else f"{name}_branch1",
    )
    branch_2 = Conv(
        x,
        filters // 2,
        1,
        1,
        None if name is None else f"{name}_branch2",
    )

    for index in range(repeats):
        branch_2 = Bottleneck(
            branch_2,
            filters // 2,
            None if name is None else f"{name}_bottleneck{index + 1}",
        )

    merged = layers.Concatenate(
        name=None if name is None else f"{name}_concat"
    )([branch_1, branch_2])

    return Conv(
        merged,
        filters,
        1,
        1,
        None if name is None else f"{name}_out",
    )


def SPPF(x, filters: int, name: str = "sppf"):
    """Fast spatial pyramid pooling at the deepest encoder level."""
    hidden = Conv(x, filters // 2, 1, 1, f"{name}_reduce")
    pooled_1 = layers.MaxPooling2D(
        5, 1, padding="same", name=f"{name}_pool1"
    )(hidden)
    pooled_2 = layers.MaxPooling2D(
        5, 1, padding="same", name=f"{name}_pool2"
    )(pooled_1)
    pooled_3 = layers.MaxPooling2D(
        5, 1, padding="same", name=f"{name}_pool3"
    )(pooled_2)

    merged = layers.Concatenate(name=f"{name}_concat")(
        [hidden, pooled_1, pooled_2, pooled_3]
    )
    return Conv(merged, filters, 1, 1, f"{name}_out")


def ChannelAttention(
    x,
    reduction: int = 8,
    name: str = "channel_attention",
):
    """Shared-MLP channel attention using average and maximum statistics."""
    channels = int(x.shape[-1])
    hidden = max(channels // reduction, 8)

    dense_1 = layers.Dense(
        hidden,
        activation="swish",
        kernel_initializer="he_normal",
        name=f"{name}_dense1",
    )
    dense_2 = layers.Dense(
        channels,
        kernel_initializer="glorot_uniform",
        name=f"{name}_dense2",
    )

    average = dense_2(
        dense_1(layers.GlobalAveragePooling2D(name=f"{name}_gap")(x))
    )
    maximum = dense_2(
        dense_1(layers.GlobalMaxPooling2D(name=f"{name}_gmp")(x))
    )
    attention = layers.Activation("sigmoid", name=f"{name}_sigmoid")(
        layers.Add(name=f"{name}_add")([average, maximum])
    )
    attention = layers.Reshape(
        (1, 1, channels), name=f"{name}_reshape"
    )(attention)
    return layers.Multiply(name=f"{name}_scale")([x, attention])


def upsample(x, factor: int = 2, name: str | None = None):
    return layers.UpSampling2D(
        factor,
        interpolation="bilinear",
        name=name,
    )(x)


# =============================================================================
# MULTI-SCALE NECK AND CONTEXT
# =============================================================================


def BiFPNCell(
    p3,
    p4,
    p5,
    channels: int,
    name: str,
):
    """One learnable top-down and bottom-up feature-pyramid pass."""
    p4_top_down = WeightedFeatureFusion(name=f"{name}_p4_td_fusion")(
        [p4, upsample(p5, 2, f"{name}_p5_up")]
    )
    p4_top_down = SeparableConv(
        p4_top_down, channels, name=f"{name}_p4_td_refine"
    )

    p3_output = WeightedFeatureFusion(name=f"{name}_p3_out_fusion")(
        [p3, upsample(p4_top_down, 2, f"{name}_p4_td_up")]
    )
    p3_output = SeparableConv(
        p3_output, channels, name=f"{name}_p3_out_refine"
    )

    p3_down = Conv(
        p3_output,
        channels,
        3,
        2,
        f"{name}_p3_down",
    )
    p4_output = WeightedFeatureFusion(name=f"{name}_p4_out_fusion")(
        [p4, p4_top_down, p3_down]
    )
    p4_output = SeparableConv(
        p4_output, channels, name=f"{name}_p4_out_refine"
    )

    p4_down = Conv(
        p4_output,
        channels,
        3,
        2,
        f"{name}_p4_down",
    )
    p5_output = WeightedFeatureFusion(name=f"{name}_p5_out_fusion")(
        [p5, p4_down]
    )
    p5_output = SeparableConv(
        p5_output, channels, name=f"{name}_p5_out_refine"
    )

    return p3_output, p4_output, p5_output


def AtrousContext(
    x,
    filters: int = 176,
    name: str = "atrous_context",
):
    """Fuse local and dilated context without reducing feature resolution."""
    if filters % 4 != 0:
        raise ValueError("AtrousContext filters must be divisible by four.")

    branch_filters = filters // 4
    branch_1 = Conv(x, branch_filters, 1, 1, f"{name}_1x1")
    branch_2 = SeparableConv(
        x,
        branch_filters,
        3,
        1,
        2,
        f"{name}_rate2",
    )
    branch_3 = SeparableConv(
        x,
        branch_filters,
        3,
        1,
        4,
        f"{name}_rate4",
    )
    branch_4 = SeparableConv(
        x,
        branch_filters,
        3,
        1,
        6,
        f"{name}_rate6",
    )

    merged = layers.Concatenate(name=f"{name}_concat")(
        [branch_1, branch_2, branch_3, branch_4]
    )
    merged = Conv(merged, filters, 1, 1, f"{name}_project")
    return ChannelAttention(merged, name=f"{name}_attention")


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================


def validate_model_output_shapes(model: Model) -> None:
    """Fail immediately if architecture and target dimensions disagree."""
    if IMG_SIZE <= 0 or IMG_SIZE % 32 != 0:
        raise ValueError(
            f"IMG_SIZE must be a positive multiple of 32; received {IMG_SIZE}."
        )
    if INSTANCE_HEAD_SIZE != IMG_SIZE // 2:
        raise ValueError(
            "This architecture produces half-resolution instance heads: "
            f"INSTANCE_HEAD_SIZE must equal IMG_SIZE // 2 ({IMG_SIZE // 2}), "
            f"but received {INSTANCE_HEAD_SIZE}."
        )
    if NUM_CLASSES < 2:
        raise ValueError("NUM_CLASSES must include background and foreground.")

    expected_input = (None, IMG_SIZE, IMG_SIZE, 3)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"Unexpected model input shape {model.input_shape}; "
            f"expected {expected_input}."
        )

    expected_outputs = {
        "semantic": (None, IMG_SIZE, IMG_SIZE, NUM_CLASSES),
        "inner_distance": (
            None,
            INSTANCE_HEAD_SIZE,
            INSTANCE_HEAD_SIZE,
            1,
        ),
        "boundary": (None, IMG_SIZE, IMG_SIZE, 1),
    }

    for layer_name, expected_shape in expected_outputs.items():
        observed_shape = tuple(model.get_layer(layer_name).output.shape)
        if observed_shape != expected_shape:
            raise ValueError(
                f"Output layer '{layer_name}' has shape {observed_shape}; "
                f"expected {expected_shape}."
            )


# =============================================================================
# MODEL V7
# =============================================================================


def build_model_v7_instance(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
) -> Model:
    """Build the edge-aware dual-decoder PCB instance model from scratch."""
    inputs = layers.Input(shape=input_shape, name="image")

    # Process RGB and fixed edges separately so edge magnitude cannot dominate
    # the colour stream. The following learned fusion can retain or suppress it.
    edges = FixedSobelFeatures(name="fixed_sobel")(inputs)
    rgb_detail = Conv(inputs, 20, 3, 1, "rgb_detail")
    edge_detail = Conv(edges, 12, 3, 1, "edge_detail")
    detail = C3k2(
        layers.Concatenate(name="rgb_edge_stem_concat")(
            [rgb_detail, edge_detail]
        ),
        32,
        1,
        "detail_fusion",
    )

    # Encoder P1/2 through P5/32.
    p1 = Conv(detail, 40, 3, 2, "backbone_p1")
    p2 = C3k2(
        Conv(p1, 72, 3, 2, "backbone_p2_down"),
        72,
        2,
        "backbone_p2",
    )
    p3 = C3k2(
        Conv(p2, 144, 3, 2, "backbone_p3_down"),
        144,
        3,
        "backbone_p3",
    )
    p4 = C3k2(
        Conv(p3, 256, 3, 2, "backbone_p4_down"),
        256,
        3,
        "backbone_p4",
    )
    p5 = C3k2(
        Conv(p4, 384, 3, 2, "backbone_p5_down"),
        384,
        2,
        "backbone_p5",
    )
    p5 = SPPF(p5, 384, "sppf")
    p5 = ChannelAttention(p5, name="deep_attention")

    # Two lightweight weighted bidirectional fusion passes at 1/8, 1/16, 1/32.
    pyramid_channels = 160
    pyramid_p3 = Conv(p3, pyramid_channels, 1, 1, "pyramid_p3_project")
    pyramid_p4 = Conv(p4, pyramid_channels, 1, 1, "pyramid_p4_project")
    pyramid_p5 = Conv(p5, pyramid_channels, 1, 1, "pyramid_p5_project")

    pyramid_p3, pyramid_p4, pyramid_p5 = BiFPNCell(
        pyramid_p3,
        pyramid_p4,
        pyramid_p5,
        pyramid_channels,
        "bifpn_1",
    )
    pyramid_p3, pyramid_p4, pyramid_p5 = BiFPNCell(
        pyramid_p3,
        pyramid_p4,
        pyramid_p5,
        pyramid_channels,
        "bifpn_2",
    )

    # Task-specific context prevents semantic classification gradients from
    # forcing the centre/offset representation to use exactly the same deep
    # feature mixture. Both branches still benefit from the shared encoder and
    # weighted pyramid, but each learns its own receptive-field balance.
    semantic_context = AtrousContext(
        pyramid_p3,
        filters=176,
        name="semantic_context",
    )
    instance_context = AtrousContext(
        pyramid_p3,
        filters=160,
        name="instance_context",
    )

    # Dedicated semantic/boundary decoder: 1/8 -> 1/4 -> 1/2. The two logit
    # maps are bilinearly upsampled to full resolution at the very end.
    # 41% of this model's MACs used to sit in the 512x512 stage; measured
    # object diameters here are 16-36 px, so half-resolution features plus a
    # smooth logit upsample cost far less accuracy than they buy in speed.
    semantic_p2 = C3k2(
        layers.Concatenate(name="semantic_p2_skip_concat")(
            [upsample(semantic_context, 2, "semantic_context_up"), p2]
        ),
        112,
        2,
        "semantic_p2",
    )
    semantic_p1 = C3k2(
        layers.Concatenate(name="semantic_p1_skip_concat")(
            [upsample(semantic_p2, 2, "semantic_p2_up"), p1]
        ),
        80,
        2,
        "semantic_p1",
    )
    detail_half = Conv(detail, 24, 3, 2, "detail_half")
    semantic_full = C3k2(
        layers.Concatenate(name="semantic_full_skip_concat")(
            [semantic_p1, detail_half]
        ),
        56,
        2,
        "semantic_full",
    )
    semantic_full = layers.SpatialDropout2D(
        0.10,
        name="semantic_spatial_dropout",
    )(semantic_full)

    # Boundary-supervised features refine semantic edges and are also passed
    # into the half-resolution instance decoder.
    boundary_features = SeparableConv(
        semantic_full,
        48,
        name="boundary_refinement",
    )
    boundary_features = Conv(
        boundary_features,
        40,
        3,
        1,
        "boundary_features",
    )
    boundary_logits = layers.Conv2D(
        1,
        1,
        padding="same",
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.keras.initializers.Constant(-2.94),
        dtype="float32",
        name="boundary_logits_half",
    )(boundary_features)
    boundary_logits = layers.UpSampling2D(
        2,
        interpolation="bilinear",
        dtype="float32",
        name="boundary",
    )(boundary_logits)

    semantic_features = C3k2(
        layers.Concatenate(name="semantic_boundary_concat")(
            [semantic_full, boundary_features]
        ),
        64,
        1,
        "semantic_boundary_fusion",
    )
    semantic_logits = layers.Conv2D(
        NUM_CLASSES,
        1,
        padding="same",
        kernel_initializer="glorot_uniform",
        dtype="float32",
        name="semantic_logits_half",
    )(semantic_features)
    semantic_logits = layers.UpSampling2D(
        2,
        interpolation="bilinear",
        dtype="float32",
        name="semantic",
    )(semantic_logits)

    # Dedicated instance decoder: 1/8 -> 1/4 -> 1/2.
    instance_p2 = C3k2(
        layers.Concatenate(name="instance_p2_skip_concat")(
            [upsample(instance_context, 2, "instance_context_up"), p2]
        ),
        104,
        2,
        "instance_p2",
    )
    instance_p1 = C3k2(
        layers.Concatenate(name="instance_p1_skip_concat")(
            [upsample(instance_p2, 2, "instance_p2_up"), p1]
        ),
        72,
        2,
        "instance_p1",
    )

    boundary_half = Conv(
        boundary_features,
        24,
        1,
        1,
        "boundary_half",
    )
    instance_features = C3k2(
        layers.Concatenate(name="instance_boundary_concat")(
            [instance_p1, boundary_half]
        ),
        80,
        2,
        "instance_boundary_fusion",
    )
    instance_features = layers.SpatialDropout2D(
        0.05,
        name="instance_spatial_dropout",
    )(instance_features)

    # One stack where V7 had two. The centre head (4 channels of sparse
    # Gaussian peaks) and the offset head (2 channels of a vector field) are
    # both replaced by a single scalar inner-distance field, from which
    # `watershed_instances` recovers the seeds the centre head used to supply.
    #
    # The MAC accounting is worth stating precisely, because the plan's "22.6%
    # freed" assumes both stacks disappear and this adds one back: the saving
    # here is roughly half of that, and it is what pays for tiling running the
    # network several times per board.
    distance_features = Conv(
        instance_features,
        64,
        3,
        1,
        "distance_features_1",
    )
    distance_features = Conv(
        distance_features,
        48,
        3,
        1,
        "distance_features_2",
    )
    inner_distance = layers.Conv2D(
        1,
        1,
        padding="same",
        # The target is a normalised distance in [0,1], so a sigmoid matches
        # its range exactly. tanh, which the offset head used, would waste half
        # its output range on values the target can never take.
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        dtype="float32",
        name="inner_distance",
    )(distance_features)

    model = Model(
        inputs=inputs,
        outputs={
            "semantic": semantic_logits,
            "inner_distance": inner_distance,
            "boundary": boundary_logits,
        },
        name="Model_v7_PCB_Scratch_BiFPN_DualContext_DualDecoder",
    )
    validate_model_output_shapes(model)
    return model


def _transfer_variable_path(variable) -> str:
    """Return a stable Keras variable identifier for gradient grouping."""
    raw_path = getattr(variable, "path", None)
    if raw_path in (None, ""):
        raw_path = getattr(variable, "name", "")
    path = str(raw_path).split(":", 1)[0]
    if not path:
        raise ValueError("Encountered a model variable without a stable path.")
    return path


def _static_tensor_shape(tensor) -> tuple[int | None, ...]:
    """Convert one TensorShape-like value into a comparable tuple."""
    return tuple(
        None if dimension is None else int(dimension)
        for dimension in tuple(tensor.shape)
    )


def _layer_weight_shapes(layer) -> list[list[int]]:
    """Return fully defined serialized weight shapes for one built layer."""
    shapes: list[list[int]] = []
    for variable in layer.weights:
        shape = []
        for dimension in tuple(variable.shape):
            if dimension is None:
                raise ValueError(
                    f"Layer {layer.name!r} has an undefined weight shape."
                )
            shape.append(int(dimension))
        shapes.append(shape)
    return shapes


def _validate_model_v5_transfer_source(source_model: Model) -> None:
    """Reject a checkpoint that is not the expected full-resolution V5 model."""
    if not isinstance(source_model, tf.keras.Model):
        raise TypeError(
            "TRANSFER_SOURCE_MODEL did not deserialize to a Keras Model."
        )

    expected_input = (None, IMG_SIZE, IMG_SIZE, 3)
    observed_input = tuple(source_model.input_shape)
    if observed_input != expected_input:
        raise ValueError(
            "Model_v5 transfer source has the wrong input shape: "
            f"{observed_input}; expected {expected_input}."
        )

    # V5 predicts semantic and boundary at 512x512. Only those two are checked:
    # a rename once swept `inner_distance` into this dict, which made the check
    # demand a head no V5 model has ever had, so the transfer path could not
    # run at all. V5's centre/offset heads have no counterpart in V7 and are not
    # transferred -- this path copies compatible encoder blocks, not heads.
    expected_outputs = {
        "semantic": (None, IMG_SIZE, IMG_SIZE, NUM_CLASSES),
        "boundary": (None, IMG_SIZE, IMG_SIZE, 1),
    }
    for layer_name, expected_shape in expected_outputs.items():
        try:
            output_layer = source_model.get_layer(layer_name)
        except ValueError as error:
            raise ValueError(
                "Model_v5 transfer source is missing output layer "
                f"{layer_name!r}."
            ) from error
        observed_shape = _static_tensor_shape(output_layer.output)
        if observed_shape != expected_shape:
            raise ValueError(
                f"Model_v5 output {layer_name!r} has shape {observed_shape}; "
                f"expected {expected_shape}. Do not use a V6 checkpoint as "
                "TRANSFER_SOURCE_MODEL."
            )


def initialize_model_v7_from_v5(
    source_model_path: Path,
) -> tuple[Model, dict[str, tuple[str, ...]], dict[str, object]]:
    """Build V7 and transplant only explicitly homologous V5 layers.

    A layer is copied only when its explicit architectural prefix rule matches,
    its Keras class matches, and every weight tensor shape matches. The method
    never performs partial tensor slicing and never copies any task head.
    """
    import hashlib

    source_path = Path(source_model_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Model_v5 transfer checkpoint was not found: {source_path}"
        )
    if source_path.stat().st_size <= 0:
        raise OSError(f"Model_v5 transfer checkpoint is empty: {source_path}")

    target_model = build_model_v7_instance()
    validate_model_output_shapes(target_model)
    source_model = tf.keras.models.load_model(source_path, compile=False)

    try:
        _validate_model_v5_transfer_source(source_model)
        source_layers = {layer.name: layer for layer in source_model.layers}
        target_layers = {layer.name: layer for layer in target_model.layers}
        if len(source_layers) != len(source_model.layers):
            raise ValueError("Model_v5 contains duplicate layer names.")
        if len(target_layers) != len(target_model.layers):
            raise ValueError("Model_v7 contains duplicate layer names.")

        copied_target_names: set[str] = set()
        copied_source_names: set[str] = set()
        records: list[dict[str, object]] = []
        variable_paths_by_group: dict[str, set[str]] = {
            group_name: set() for group_name in TRANSFER_GROUP_NAMES
        }

        for source_layer in source_model.layers:
            if not source_layer.weights:
                continue

            matching_rules = [
                rule
                for rule in TRANSFER_LAYER_PREFIX_RULES
                if str(source_layer.name).startswith(rule[0])
            ]
            if not matching_rules:
                continue
            if len(matching_rules) != 1:
                raise ValueError(
                    f"Ambiguous transfer rules for source layer "
                    f"{source_layer.name!r}: {matching_rules}."
                )

            source_prefix, target_prefix, group_name = matching_rules[0]
            target_name = (
                target_prefix + str(source_layer.name)[len(source_prefix) :]
            )
            if target_name not in target_layers:
                raise ValueError(
                    f"Audited V5 layer {source_layer.name!r} maps to missing "
                    f"V7 layer {target_name!r}."
                )
            if target_name in copied_target_names:
                raise ValueError(
                    f"More than one V5 layer maps to {target_name!r}."
                )

            target_layer = target_layers[target_name]
            source_class = source_layer.__class__.__name__
            target_class = target_layer.__class__.__name__
            if source_class != target_class:
                raise TypeError(
                    f"Layer class mismatch for {source_layer.name!r} -> "
                    f"{target_name!r}: {source_class} versus {target_class}."
                )

            source_shapes = _layer_weight_shapes(source_layer)
            target_shapes = _layer_weight_shapes(target_layer)
            if source_shapes != target_shapes:
                raise ValueError(
                    f"Weight-shape mismatch for {source_layer.name!r} -> "
                    f"{target_name!r}: {source_shapes} versus {target_shapes}. "
                    "No partial transfer was attempted."
                )

            source_weights = source_layer.get_weights()
            initial_target_weights = target_layer.get_weights()
            if len(source_weights) != len(initial_target_weights):
                raise ValueError(
                    f"Weight-count mismatch for {source_layer.name!r} -> "
                    f"{target_name!r}."
                )
            target_layer.set_weights(source_weights)
            copied_weights = target_layer.get_weights()
            if len(copied_weights) != len(source_weights) or any(
                not np.array_equal(source_array, copied_array)
                for source_array, copied_array in zip(
                    source_weights, copied_weights
                )
            ):
                raise RuntimeError(
                    f"Post-copy verification failed for target layer "
                    f"{target_name!r}."
                )

            target_changed = any(
                not np.array_equal(before, after)
                for before, after in zip(
                    initial_target_weights, copied_weights
                )
            )
            parameter_count = int(
                sum(np.prod(shape, dtype=np.int64) for shape in source_shapes)
            )
            records.append(
                {
                    "source_layer": str(source_layer.name),
                    "target_layer": str(target_name),
                    "layer_class": source_class,
                    "unfreeze_group": str(group_name),
                    "weight_shapes": source_shapes,
                    "parameters": parameter_count,
                    "different_from_target_initialization": bool(
                        target_changed
                    ),
                    "post_copy_exact_match": True,
                }
            )
            copied_source_names.add(str(source_layer.name))
            copied_target_names.add(str(target_name))
            for variable in target_layer.trainable_variables:
                variable_paths_by_group[group_name].add(
                    _transfer_variable_path(variable)
                )

        missing_required = sorted(
            set(TRANSFER_REQUIRED_TARGET_LAYERS) - copied_target_names
        )
        if missing_required:
            raise RuntimeError(
                "The transfer audit did not copy required V7 layers: "
                f"{missing_required}."
            )
        if not records:
            raise RuntimeError("No compatible Model_v5 layers were copied.")

        path_owner: dict[str, str] = {}
        for group_name, paths in variable_paths_by_group.items():
            if not paths:
                raise RuntimeError(
                    f"Transfer group {group_name!r} has no trainable variables."
                )
            for path in paths:
                previous_group = path_owner.get(path)
                if previous_group is not None:
                    raise RuntimeError(
                        f"Variable {path!r} belongs to both {previous_group!r} "
                        f"and {group_name!r}."
                    )
                path_owner[path] = group_name

        transferred_parameters = int(
            sum(int(record["parameters"]) for record in records)
        )
        group_parameter_counts = {
            group_name: int(
                sum(
                    int(record["parameters"])
                    for record in records
                    if record["unfreeze_group"] == group_name
                )
            )
            for group_name in TRANSFER_GROUP_NAMES
        }
        if len(records) != int(TRANSFER_EXPECTED_WEIGHTED_LAYER_COUNT):
            raise RuntimeError(
                "Audited transfer layer count changed: "
                f"observed={len(records)}, "
                f"expected={TRANSFER_EXPECTED_WEIGHTED_LAYER_COUNT}."
            )
        if transferred_parameters != int(TRANSFER_EXPECTED_PARAMETER_COUNT):
            raise RuntimeError(
                "Audited transfer parameter count changed: "
                f"observed={transferred_parameters:,}, "
                f"expected={TRANSFER_EXPECTED_PARAMETER_COUNT:,}."
            )
        if group_parameter_counts != dict(
            TRANSFER_EXPECTED_GROUP_PARAMETER_COUNTS
        ):
            raise RuntimeError(
                "Audited transfer group parameter counts changed: "
                f"observed={group_parameter_counts}, expected="
                f"{TRANSFER_EXPECTED_GROUP_PARAMETER_COUNTS}."
            )
        total_parameters = int(target_model.count_params())
        source_digest = hashlib.sha256()
        with source_path.open("rb") as source_handle:
            while True:
                chunk = source_handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                source_digest.update(chunk)
        source_stat = source_path.stat()

        manifest: dict[str, object] = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": (
                "explicit homologous layer mapping plus exact class/shape "
                "verification; no partial tensors"
            ),
            "source_checkpoint": str(source_path),
            "source_checkpoint_size_bytes": int(source_stat.st_size),
            "source_checkpoint_mtime_ns": int(source_stat.st_mtime_ns),
            "source_checkpoint_sha256": source_digest.hexdigest(),
            "source_model_name": str(source_model.name),
            "target_model_name": str(target_model.name),
            "source_preprocessing": "512x512 direct stretch",
            "target_preprocessing": "512x512 aspect-ratio-preserving letterbox",
            "excluded_from_transfer": [
                "rgb/edge stem and p1-p3 because channel dimensions differ",
                "all semantic, boundary, center, and offset heads",
                "all V5 FPN, PAN, context, and decoder layers",
                "source optimizer, EMA, and training state",
            ],
            "copied_layer_count": int(len(records)),
            "copied_parameter_count": transferred_parameters,
            "group_parameter_counts": group_parameter_counts,
            "target_parameter_count": total_parameters,
            "target_parameter_coverage": float(
                transferred_parameters / max(total_parameters, 1)
            ),
            "copied_source_layers": sorted(copied_source_names),
            "copied_target_layers": sorted(copied_target_names),
            "gradient_groups": {
                group_name: sorted(paths)
                for group_name, paths in variable_paths_by_group.items()
            },
            "layer_records": records,
        }
        frozen_groups = {
            group_name: tuple(sorted(paths))
            for group_name, paths in variable_paths_by_group.items()
        }
        return target_model, frozen_groups, manifest
    finally:
        del source_model
        gc.collect()

# =============================================================================
# 10. INSTANCE DECODING THAT USES ALL FOUR HEADS
# =============================================================================


def resize_channels(
    array: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize an HWC float tensor without changing its channel semantics."""
    array = np.asarray(array, dtype=np.float32)
    width = int(width)
    height = int(height)

    if array.ndim != 3:
        raise ValueError(
            f"resize_channels expects an HWC tensor; received {array.shape}."
        )
    if array.shape[-1] < 1:
        raise ValueError("resize_channels received an empty channel dimension.")
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Resize dimensions must be positive; received {width}x{height}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("resize_channels received NaN or infinite values.")

    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.shape[1] == width and array.shape[0] == height:
        return array.copy()

    # OpenCV resizes every channel independently internally. Linear
    # interpolation is intentionally retained for both probabilities and
    # normalized offset vectors: cubic interpolation can overshoot their valid
    # ranges, while nearest-neighbour interpolation creates blocky offsets.
    resized = cv2.resize(
        array,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCV removes the final axis when an input has exactly one channel.
    if resized.ndim == 2:
        resized = resized[..., None]

    expected_shape = (height, width, array.shape[-1])
    if resized.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected resized shape {resized.shape}; expected {expected_shape}."
        )
    return np.ascontiguousarray(resized, dtype=np.float32)


def find_center_peaks(
    heatmap: np.ndarray,
    nms_radius: int | None = None,
    confidence_threshold: float | None = None,
    maximum_centers: int | None = None,
) -> np.ndarray:
    """Find deterministic, plateau-safe, subpixel-refined centre peaks.

    Returns an ``N x 3`` float32 array containing ``[x, y, score]``. Spatial
    non-maximum suppression uses a circular neighbourhood and every connected
    flat maximum is represented by only one candidate.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(
            f"find_center_peaks expects a 2-D heatmap; received {heatmap.shape}."
        )
    if heatmap.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    radius = int(
        CENTER_NMS_RADIUS
        if nms_radius is None
        else nms_radius
    )
    threshold = float(
        CENTER_CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else confidence_threshold
    )
    maximum_centers = int(
        MAX_CENTERS_PER_CLASS
        if maximum_centers is None
        else maximum_centers
    )
    if radius < 0:
        raise ValueError("CENTER_NMS_RADIUS must be >= 0.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("CENTER_CONFIDENCE_THRESHOLD must be in [0, 1].")
    if maximum_centers < 1:
        raise ValueError("MAX_CENTERS_PER_CLASS must be >= 1.")

    # Model outputs are probabilities. Invalid values are rejected from peak
    # candidacy rather than being allowed to poison OpenCV morphology.
    heatmap = np.nan_to_num(
        heatmap,
        copy=True,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    heatmap = np.clip(heatmap, 0.0, 1.0)

    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    neighbourhood_maximum = cv2.dilate(heatmap, kernel)

    # The tolerance prevents insignificant floating-point differences from
    # breaking one flat maximum into many candidates.
    peak_mask = (
        (heatmap >= threshold)
        & (heatmap >= neighbourhood_maximum - 1e-7)
    ).astype(np.uint8)
    if not np.any(peak_mask):
        return np.empty((0, 3), dtype=np.float32)

    component_count, component_labels, _, centroids = (
        cv2.connectedComponentsWithStats(peak_mask, connectivity=8)
    )

    candidates: list[tuple[float, float, float]] = []
    refinement_radius = max(1, radius // 2)
    image_height, image_width = heatmap.shape

    for component_id in range(1, component_count):
        component_y, component_x = np.where(
            component_labels == component_id
        )
        if len(component_x) == 0:
            continue

        component_scores = heatmap[component_y, component_x]
        score = float(np.max(component_scores))

        # On a plateau, select the highest pixel nearest the component centroid
        # so the result is deterministic and spatially central.
        highest = np.flatnonzero(component_scores >= score - 1e-7)
        centroid_x, centroid_y = centroids[component_id]
        highest_x = component_x[highest].astype(np.float32)
        highest_y = component_y[highest].astype(np.float32)
        closest = int(
            np.argmin(
                np.square(highest_x - centroid_x)
                + np.square(highest_y - centroid_y)
            )
        )
        integer_x = int(highest_x[closest])
        integer_y = int(highest_y[closest])

        # Refine the integer maximum with only the high-confidence part of its
        # local response. This improves offset assignment without moving a peak
        # toward a nearby lower response.
        x0 = max(0, integer_x - refinement_radius)
        x1 = min(image_width, integer_x + refinement_radius + 1)
        y0 = max(0, integer_y - refinement_radius)
        y1 = min(image_height, integer_y + refinement_radius + 1)
        patch = heatmap[y0:y1, x0:x1]
        weight_floor = max(threshold, 0.5 * score)
        weights = np.maximum(patch - weight_floor, 0.0).astype(np.float64)

        weight_sum = float(weights.sum())
        if weight_sum > 1e-12:
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            refined_x = float(np.sum(grid_x * weights) / weight_sum)
            refined_y = float(np.sum(grid_y * weights) / weight_sum)
        else:
            refined_x = float(integer_x)
            refined_y = float(integer_y)

        candidates.append((score, refined_x, refined_y))

    # Stable deterministic ordering: strongest score, then top-to-bottom and
    # left-to-right when scores are equal.
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))

    selected: list[tuple[float, float, float]] = []
    minimum_distance_squared = float(radius * radius)
    for score, x, y in candidates:
        if all(
            (x - old_x) ** 2 + (y - old_y) ** 2
            > minimum_distance_squared
            for _, old_x, old_y in selected
        ):
            selected.append((score, x, y))
        if len(selected) >= maximum_centers:
            break

    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(
        [(x, y, score) for score, x, y in selected],
        dtype=np.float32,
    )


def boundary_partition(
    candidate_mask: np.ndarray,
    boundary_probability: np.ndarray,
) -> list[np.ndarray]:
    """Conservatively split a candidate using the predicted boundary ridge.

    Unlike the previous implementation, this function never opens/erodes the
    object cores. It closes only one-pixel gaps in confident boundary ridges,
    finds sufficiently large non-boundary cores, and then restores every
    boundary pixel to its nearest valid core. If a valid lossless split cannot
    be guaranteed, the original candidate is returned unchanged.
    """
    candidate = np.asarray(candidate_mask, dtype=bool)
    boundary = np.asarray(boundary_probability, dtype=np.float32)

    if candidate.ndim != 2:
        raise ValueError(
            f"candidate_mask must be 2-D; received {candidate.shape}."
        )
    if boundary.ndim == 3 and boundary.shape[-1] == 1:
        boundary = boundary[..., 0]
    if boundary.ndim != 2:
        raise ValueError(
            "boundary_probability must be [H,W] or [H,W,1]; "
            f"received {boundary.shape}."
        )
    if boundary.shape != candidate.shape:
        raise ValueError(
            "Candidate and boundary shapes must match; "
            f"received {candidate.shape} and {boundary.shape}."
        )

    candidate_area = int(candidate.sum())
    minimum_instance_area = int(MIN_INSTANCE_AREA)
    minimum_core_area = int(MIN_BOUNDARY_CORE_AREA)
    threshold = float(BOUNDARY_CONFIDENCE_THRESHOLD)

    if minimum_instance_area < 1:
        raise ValueError("MIN_INSTANCE_AREA must be >= 1.")
    if minimum_core_area < 1:
        raise ValueError("MIN_BOUNDARY_CORE_AREA must be >= 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("BOUNDARY_CONFIDENCE_THRESHOLD must be in [0, 1].")
    if candidate_area < minimum_instance_area:
        return []

    boundary = np.nan_to_num(
        boundary,
        copy=True,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    boundary = np.clip(boundary, 0.0, 1.0)

    # Light smoothing suppresses isolated probability spikes. A 3x3 closing is
    # applied to the boundary ridge—not the object core—to bridge only tiny
    # prediction gaps that would otherwise reconnect two touching objects.
    smoothed_boundary = cv2.GaussianBlur(
        boundary,
        (3, 3),
        sigmaX=0.65,
        sigmaY=0.65,
        borderType=cv2.BORDER_REPLICATE,
    )
    boundary_barrier = (
        candidate & (smoothed_boundary >= threshold)
    ).astype(np.uint8)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_barrier = cv2.morphologyEx(
        boundary_barrier,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1,
    )
    boundary_barrier = (
        (boundary_barrier > 0) & candidate
    )

    # Crucially, no opening or erosion is applied to the remaining cores.
    core = candidate & (~boundary_barrier)
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            core.astype(np.uint8),
            connectivity=8,
        )
    )
    valid_component_ids = [
        component_id
        for component_id in range(1, component_count)
        if int(component_stats[component_id, cv2.CC_STAT_AREA])
        >= minimum_core_area
    ]
    if len(valid_component_ids) <= 1:
        return [candidate]

    valid_core = np.isin(component_labels, valid_component_ids)
    if not np.any(valid_core):
        return [candidate]

    # Assign the removed ridge and all remaining candidate pixels to the
    # nearest connected core. DIST_LABEL_CCOMP assigns one label per connected
    # zero-valued core component.
    distance_source = np.ones(candidate.shape, dtype=np.uint8)
    distance_source[valid_core] = 0
    _, nearest_core = cv2.distanceTransformWithLabels(
        distance_source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )

    nearest_ids = sorted(
        int(value)
        for value in np.unique(nearest_core[valid_core])
        if int(value) > 0
    )
    if len(nearest_ids) <= 1:
        return [candidate]

    partitions = [
        candidate & (nearest_core == nearest_id)
        for nearest_id in nearest_ids
    ]

    # A split must be lossless and every resulting object must satisfy the
    # instance-size requirement. Otherwise retaining one candidate is safer
    # than creating false fragments or silently dropping pixels.
    if any(int(part.sum()) < minimum_instance_area for part in partitions):
        return [candidate]

    covered = np.zeros_like(candidate, dtype=bool)
    for part in partitions:
        if np.any(covered & part):
            return [candidate]
        covered |= part
    if not np.array_equal(covered, candidate):
        return [candidate]

    # Deterministic order makes instance IDs reproducible between runs.
    partitions.sort(
        key=lambda part: (
            int(np.nonzero(part)[0].min()),
            int(np.nonzero(part)[1].min()),
        )
    )
    return [np.asarray(part, dtype=bool) for part in partitions]



def tile_grid(
    height: int,
    width: int,
    tile: int = IMG_SIZE,
    overlap: int = TILE_OVERLAP,
) -> list[tuple[int, int]]:
    """Top-left corners covering [height, width] with overlapping tiles.

    The last row and column are pulled back flush with the edge rather than
    padded, so no tile contains manufactured background. Padding here would
    reintroduce the very thing tiling exists to remove.
    """
    tile = int(tile)
    overlap = int(overlap)
    if tile <= 0:
        raise ValueError("tile must be positive.")
    if not 0 <= overlap < tile:
        raise ValueError(
            f"overlap must lie in [0, tile); received {overlap} for tile {tile}."
        )

    stride = tile - overlap
    positions: list[tuple[int, int]] = []
    tops = list(range(0, max(int(height) - tile, 0) + 1, stride))
    lefts = list(range(0, max(int(width) - tile, 0) + 1, stride))
    if not tops or tops[-1] != max(int(height) - tile, 0):
        tops.append(max(int(height) - tile, 0))
    if not lefts or lefts[-1] != max(int(width) - tile, 0):
        lefts.append(max(int(width) - tile, 0))
    for top in tops:
        for left in lefts:
            positions.append((int(top), int(left)))
    return positions


def _tile_blend_weight(
    tile: int = IMG_SIZE,
    margin: int = TILE_BLEND_MARGIN,
) -> np.ndarray:
    """Cosine taper, 0 at the tile edge rising to 1 over `margin` pixels."""
    tile = int(tile)
    margin = int(min(max(margin, 1), tile // 2))
    ramp = 0.5 * (
        1.0 - np.cos(np.pi * (np.arange(margin, dtype=np.float32) + 0.5) / margin)
    )
    profile = np.ones(tile, dtype=np.float32)
    profile[:margin] = ramp
    profile[-margin:] = ramp[::-1]
    return np.outer(profile, profile).astype(np.float32)


def predict_fields_tiled(
    model,
    image: np.ndarray,
    tile: int = IMG_SIZE,
    overlap: int = TILE_OVERLAP,
    batch_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over overlapping tiles and stitch the predicted fields.

    Returns ``(semantic_probabilities, inner_distance, boundary_probability)``
    at the *source* resolution, ready for `decode_instances`.

    Fields are stitched and the decoder runs once over the result, rather than
    decoding each tile and merging instances afterwards. Merging instance maps
    across a seam means deciding whether two partial masks are one object,
    which is guesswork; stitching the fields leaves that decision to the same
    grouping rule that handles every other pixel.

    The inner-distance field is normalised per instance, so an object split
    across a seam would be normalised against a partial view of itself in each
    tile. `TILE_OVERLAP` is sized so that objects smaller than the overlap are
    always wholly inside at least one tile, which is what keeps that from
    mattering in practice.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be [H,W,3]; received {image.shape}.")
    height, width = image.shape[:2]
    tile = int(min(tile, height, width))

    positions = tile_grid(height, width, tile, min(overlap, tile - 1))
    weight_tile = _tile_blend_weight(tile)

    semantic_sum = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    distance_sum = np.zeros((height, width), dtype=np.float32)
    boundary_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(positions), max(int(batch_size), 1)):
        chunk = positions[start:start + max(int(batch_size), 1)]
        batch = np.stack([
            image[top:top + tile, left:left + tile] for top, left in chunk
        ])
        outputs = model(batch, training=False)
        for position, (top, left) in enumerate(chunk):
            semantic, distance, boundary = output_probabilities(
                outputs, position
            )
            # The distance head runs at half resolution; bring it up to the
            # tile grid before stitching so every field shares one geometry.
            distance_full = resize_channels(distance, tile, tile)[..., 0]

            semantic_sum[top:top + tile, left:left + tile] += (
                semantic * weight_tile[..., None]
            )
            distance_sum[top:top + tile, left:left + tile] += (
                distance_full * weight_tile
            )
            boundary_sum[top:top + tile, left:left + tile] += (
                boundary * weight_tile
            )
            weight_sum[top:top + tile, left:left + tile] += weight_tile

    # The taper is zero at the outer frame edge, where only one tile
    # contributes, so guard the division rather than assuming coverage.
    safe = np.maximum(weight_sum, 1e-6)[..., None]
    semantic = semantic_sum / safe
    semantic /= np.maximum(semantic.sum(axis=-1, keepdims=True), 1e-6)
    distance = np.clip(distance_sum / safe[..., 0], 0.0, 1.0)
    boundary = np.clip(boundary_sum / safe[..., 0], 0.0, 1.0)

    return (
        np.ascontiguousarray(semantic.astype(np.float32)),
        np.ascontiguousarray(distance.astype(np.float32)[..., None]),
        np.ascontiguousarray(boundary.astype(np.float32)),
    )


def predict_fields(
    model,
    image: np.ndarray,
    tile: int = IMG_SIZE,
    overlap: int = TILE_OVERLAP,
    batch_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One entry point for inference, tiled or not, ready for `decode_instances`.

    `TILE_INFERENCE_ENABLED` had no reader: `predict_fields_tiled` was called
    only from its own test, so the flag read True while every real prediction
    went through a single pass. This is the dispatch that makes it mean
    something.

    **Tiling helps only on a source larger than one tile**, and this returns a
    plain single pass otherwise -- tiling a 512x512 letterboxed frame would run
    the network once on the whole image and call it a grid. That matters because
    the prepared arrays *are* 512x512, so evaluation on the benchmark splits is
    unaffected by this flag; only prediction from original-resolution boards is.

    **And the gain it can offer is bounded.** Tiling exists because a thin ring
    cannot be resolved at head resolution -- but measured on the test split, only
    3.1% of `circle`, 3.9% of `Rectangle` and 2.3% of `circle_full` instances
    have a sub-floor core at INSTANCE_HEAD_SIZE, and 1-2% are degenerate. The
    synthetic 1.5 px annulus that motivated tiling is a worst case, not the
    common one, so expect a few per cent of objects to change, not a step
    improvement.

    The distinction from zoom-in augmentation is worth keeping in view: a tile is
    a *native-resolution crop*, carrying detail the letterboxed frame threw away.
    Zoom-in upsamples, which manufactures detail and measurably cost the Tier-1
    run its mask precision. They are not the same operation.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be [H,W,3]; received {image.shape}.")
    height, width = image.shape[:2]

    if TILE_INFERENCE_ENABLED and (height > int(tile) or width > int(tile)):
        return predict_fields_tiled(
            model, image, tile=tile, overlap=overlap, batch_size=batch_size
        )

    outputs = model(image[None, ...], training=False)
    semantic, distance, boundary = output_probabilities(outputs, 0)
    return (
        np.ascontiguousarray(semantic.astype(np.float32)),
        np.ascontiguousarray(np.asarray(distance, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(boundary, dtype=np.float32)),
    )


def watershed_instances(
    foreground_mask: np.ndarray,
    inner_distance: np.ndarray,
    core_threshold: float = INNER_DISTANCE_CORE_THRESHOLD,
    minimum_instance_area: int = MIN_INSTANCE_AREA,
    core_smoothing: float = INNER_DISTANCE_CORE_SMOOTHING,
    minimum_core_area: int = MIN_INNER_DISTANCE_CORE_AREA,
    core_closing: int = INNER_DISTANCE_CORE_CLOSING,
) -> np.ndarray:
    """Group one class's foreground into instances by inner-distance cores.

    Returns an int32 label map over ``foreground_mask``: 0 for background,
    1..n for instances. Replaces `nearest_center_assignments`, and removes
    three of its failure modes by construction rather than by threshold:

    * **No distance cap.** Assignment was gated on the projected offset landing
      within 51 head pixels of a centre, so a large object's outer pixels were
      unassignable no matter how good the prediction. Cores here are found
      inside the object itself; there is no distance for them to exceed.
    * **No unassigned residue, so no fallback.** Every foreground pixel has a
      nearest core, so nothing is left over for the fallback path to mint
      instances from. That path produced 75 false positives against 2 true
      positives on validation and is deleted rather than gated.
    * **No centre peaks.** Seeds come from the field's own high ground, so an
      object needs no separate centre channel to be found, and a weak spurious
      peak cannot invent one.

    * **No cross-object leakage.** Cores are found, filtered and expanded within
      each connected foreground blob, so a blob always keeps its own pixels even
      when its core is too small to clear the area floor.

    A pixel joins the nearest core *inside its own blob*, exactly as
    `boundary_partition` restores ridge pixels -- a distance transform with
    DIST_LABEL_CCOMP, which labels by connected zero-component, so every core
    keeps one identity.
    """
    foreground = np.asarray(foreground_mask, dtype=bool)
    distance = np.asarray(inner_distance, dtype=np.float32)
    if distance.ndim == 3 and distance.shape[-1] == 1:
        distance = distance[..., 0]
    if foreground.ndim != 2:
        raise ValueError(
            f"foreground_mask must be 2-D; received {foreground.shape}."
        )
    if distance.shape != foreground.shape:
        raise ValueError(
            "inner_distance and foreground_mask shapes differ: "
            f"{distance.shape} versus {foreground.shape}."
        )
    threshold = float(core_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"core_threshold must lie in (0,1); received {threshold}."
        )

    labels = np.zeros(foreground.shape, dtype=np.int32)
    if not foreground.any():
        return labels

    # Smooth before thresholding. A true inner-distance field has no high
    # frequencies -- it is a distance transform -- so smoothing encodes that
    # prior rather than hiding a problem. Without it, thresholding a noisy
    # prediction yields speckled cores, and because every core expands to
    # claim territory by nearest-distance, each speck becomes a full-sized
    # instance that the final area filter cannot catch. Measured on the
    # synthetic scenes, sigma 1.0 blur plus 0.05 noise turned 2 objects into
    # 15 before this was added.
    field = distance
    if float(core_smoothing) > 0.0:
        field = cv2.GaussianBlur(distance, (0, 0), float(core_smoothing))

    # Everything below runs **per connected foreground blob**, cropped to its
    # own bounding box. The first version worked on the whole class mask at
    # once, and that leaked across objects twice:
    #
    #   * the minimum-core-area filter was global. If any core anywhere cleared
    #     the floor, every sub-floor core was dropped -- so a small object beside
    #     a large one lost its core while the large one kept its own.
    #   * expansion was global and Euclidean, with no connectivity constraint
    #     and no distance cap. A blob with no core of its own did not become its
    #     own instance; its pixels went to the nearest *other* blob's core,
    #     producing a spatially disconnected mask.
    #
    # Together those are the "16 px disc vanishes into the 130 px rectangle"
    # failure that relative thresholding was added to fix, returning through a
    # different door. Measured on the test split, 3.9% of Rectangles, 3.1% of
    # circles and 2.3% of circle_fulls have a sub-floor core at head resolution
    # -- roughly two objects per image at 77.5 objects/image, so this was not a
    # corner case.
    blob_count, blobs = cv2.connectedComponents(
        foreground.astype(np.uint8), connectivity=8
    )
    if blob_count <= 1:
        return labels

    next_label = 1
    for blob_id in range(1, blob_count):
        rows, columns = np.nonzero(blobs == blob_id)
        if rows.size == 0:
            continue
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        blob = blobs[y0:y1, x0:x1] == blob_id
        field_crop = field[y0:y1, x0:x1]

        # Threshold relative to this blob's own maximum, not against an
        # absolute 0.5. The target peaks at exactly 1.0 per instance, but a
        # prediction need not: a small or thin object is the kind a model
        # under-predicts, and under an absolute threshold it yields no core at
        # all. Relative thresholding drops the assumption that prediction
        # magnitude is calibrated, and changes nothing on a well-formed field,
        # where every blob maximum is 1.0 and this reduces to the absolute rule.
        peak = float(field_crop[blob].max()) if blob.any() else 0.0
        if peak <= 0.0:
            peak = 1.0
        cores = blob & (field_crop >= threshold * peak)

        # Bridge one-pixel gaps, the same conservative closing
        # `boundary_partition` applies to boundary ridges and for the same
        # reason. An annulus is the case that needs it: at head resolution a
        # `circle` is a ring a few pixels thick, so its core is a one-pixel-wide
        # centreline with no redundancy, and a single noisy pixel splits the
        # ring into arcs that each clear the area floor. The kernel stays at one
        # pixel so it can only rejoin what noise separated, never bridge two
        # genuinely distinct cores, which sit half an object apart.
        if cores.any() and int(core_closing) > 0:
            radius = int(core_closing)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            cores = (
                cv2.morphologyEx(
                    cores.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                ).astype(bool)
                & blob
            )

        if cores.any() and int(minimum_core_area) > 1:
            # A core below this area is noise, not an object's inner half, and
            # the filter has to be applied to the *cores* before expansion --
            # afterwards every core has grown into a plausible-looking instance.
            # Scoped to this blob, and if the filter would empty it the largest
            # core is kept: an object whose core is genuinely tiny is still an
            # object, and must not be handed to a neighbour.
            core_count, core_labels = cv2.connectedComponents(
                cores.astype(np.uint8), connectivity=8
            )
            if core_count > 1:
                areas = np.bincount(core_labels.ravel(), minlength=core_count)
                keep = np.zeros(core_count, dtype=bool)
                keep[1:] = areas[1:] >= int(minimum_core_area)
                if not keep[1:].any():
                    keep[int(np.argmax(areas[1:])) + 1] = True
                cores = keep[core_labels]

        if not cores.any():
            # No evidence for a split anywhere in this blob. Treating it as one
            # instance is the honest degenerate answer: it under-segments
            # touching objects rather than inventing splits the field gives no
            # evidence for, and -- unlike handing the pixels to another blob's
            # core -- it keeps the object.
            labels[y0:y1, x0:x1][blob] = next_label
            next_label += 1
            continue

        core_count, _ = cv2.connectedComponents(
            cores.astype(np.uint8), connectivity=8
        )
        if core_count <= 1:
            labels[y0:y1, x0:x1][blob] = next_label
            next_label += 1
            continue

        # Expand every core over this blob by nearest distance, exactly as
        # `boundary_partition` restores ridge pixels. DIST_LABEL_CCOMP labels by
        # connected zero-component, so each core keeps one identity. Confining
        # it to the blob's own bounding box is what prevents a core from
        # claiming pixels across a gap.
        distance_source = np.ones(blob.shape, dtype=np.uint8)
        distance_source[cores] = 0
        _, nearest_core = cv2.distanceTransformWithLabels(
            distance_source,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        grouped_blob = np.where(blob, nearest_core, 0).astype(np.int32)
        present = [int(v) for v in np.unique(grouped_blob) if int(v) > 0]
        for value in present:
            piece = grouped_blob == value
            labels[y0:y1, x0:x1][piece] = next_label
            next_label += 1

    return _relabel_by_area(labels, minimum_instance_area)


def inner_distance_instance_score(
    mask_crop: np.ndarray,
    predicted_crop: np.ndarray,
    semantic_crop: np.ndarray | None = None,
    upsample_factor: int = max(1, int(round(IMG_SIZE / INSTANCE_HEAD_SIZE))),
) -> float:
    """Detection confidence: how well the predicted field fits this mask.

    **Why not the peak predicted distance.** That was the first ranking signal
    here, and it does not work -- the target normalises every instance to peak
    at exactly 1.0, and `watershed_instances` guarantees every returned instance
    contains a core, so the peak is ~1.0 for a perfect object, a merged pair and
    a rim sliver alike. Fed the *oracle* field, a four-object scene decoded to
    scores 0.8593, 0.9863, 1.0000, 1.0000 -- and that 0.14 of spread came from
    upsampling smoothing small peaks, not from instance quality, so it ranked by
    size and ranked small objects worst. Mask AP is computed from that ordering.

    **What this measures instead.** A normalised inner-distance field is fully
    determined by the shape it describes: given a mask, the field a correct
    prediction *must* produce can be computed exactly. So decode the instance,
    compute the field its own shape implies, and score the agreement. That is a
    quantity a wrong grouping cannot fake:

    * a correct object -- prediction equals its own implied field, agreement 1;
    * a **fragment** -- its pixels carry a field normalised over the whole
      object, which does not reach 1.0 inside the fragment and has the wrong
      profile for the fragment's shape;
    * a **merged pair** -- two humps where one broad hump is implied;
    * a **rim sliver** -- a field near zero where its own shape implies a full
      0-to-1 ramp.

    Multiplying by mean semantic confidence keeps class certainty in the score.
    That term saturates near 1.0 on its own -- which is exactly why it cannot be
    the ranking signal -- but as a factor it still demotes a well-formed blob the
    classifier is unsure about, and costs nothing.

    Works on crops, not full frames: the caller passes the component's bounding
    box, so cost is proportional to object area rather than image area.
    """
    mask_crop = np.asarray(mask_crop, dtype=bool)
    predicted_crop = np.asarray(predicted_crop, dtype=np.float32)
    if mask_crop.shape != predicted_crop.shape:
        raise ValueError(
            "mask and prediction crops differ: "
            f"{mask_crop.shape} versus {predicted_crop.shape}."
        )
    if not mask_crop.any():
        return 0.0

    # The field this shape implies. Zero-padded so the crop border counts as
    # background -- true by construction, since the crop is the component's own
    # bounding box -- which is the same convention the target generator uses.
    padded = np.pad(mask_crop.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    peak = float(distance.max())
    if peak > 0.0:
        implied = distance / peak
    else:
        # One pixel wide: every pixel is on the boundary. The target generator
        # assigns 1.0 rather than dividing by zero, so match it.
        implied = np.ones_like(distance, dtype=np.float32)

    # Put the implied field through the prediction's own resolution round trip.
    # The head runs at INSTANCE_HEAD_SIZE and its output is bilinearly upsampled,
    # so a crisp analytic transform is not what a *correct* prediction looks
    # like -- and the mismatch falls hardest on small objects, exactly the ones
    # that must not be under-ranked. Measured on the synthetic scenes this lifts
    # a 16 px disc from 0.890 to 0.901 while leaving every wrong grouping where
    # it was, widening the correct-versus-wrong margin from 0.025 to 0.037.
    factor = max(1, int(upsample_factor))
    if factor > 1 and min(implied.shape) >= factor:
        height_i, width_i = implied.shape
        small = cv2.resize(
            implied,
            (max(1, width_i // factor), max(1, height_i // factor)),
            interpolation=cv2.INTER_AREA,
        )
        implied = cv2.resize(
            small, (width_i, height_i), interpolation=cv2.INTER_LINEAR
        )

    error = float(np.mean(np.abs(predicted_crop[mask_crop] - implied[mask_crop])))
    agreement = float(min(max(1.0 - error, 0.0), 1.0))

    if semantic_crop is None:
        return agreement
    semantic_crop = np.asarray(semantic_crop, dtype=np.float32)
    confidence = float(np.mean(semantic_crop[mask_crop]))
    confidence = float(min(max(confidence, 0.0), 1.0))
    return float(min(max(agreement * confidence, 0.0), 1.0))


def _relabel_by_area(
    labelled: np.ndarray,
    minimum_instance_area: int,
) -> np.ndarray:
    """Drop undersized components and renumber the rest 1..n, deterministically.

    Ordering is by first pixel in row-major order, the same rule
    `boundary_partition` sorts its partitions by, so instance ids are
    reproducible between runs rather than dependent on label allocation order.
    """
    labelled = np.asarray(labelled, dtype=np.int32)
    result = np.zeros(labelled.shape, dtype=np.int32)
    present = [int(value) for value in np.unique(labelled) if int(value) > 0]
    if not present:
        return result

    kept = []
    for value in present:
        mask = labelled == value
        if int(mask.sum()) < int(minimum_instance_area):
            continue
        rows, columns = np.nonzero(mask)
        kept.append((int(rows[0]), int(columns[0]), value))
    kept.sort()
    for new_id, (_, _, value) in enumerate(kept, start=1):
        result[labelled == value] = new_id
    return result


def nearest_center_assignments(
    xs: np.ndarray,
    ys: np.ndarray,
    projected_x: np.ndarray,
    projected_y: np.ndarray,
    centers: np.ndarray,
    maximum_working_memory_mb: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign projected pixels to centres with bounded temporary memory.

    The return values are the zero-based index of the nearest centre and its
    Euclidean distance for every input pixel. Equal-distance ties select the
    first centre; ``find_center_peaks`` orders centres by confidence, so this
    deterministically favours the stronger centre.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    projected_x = np.asarray(projected_x, dtype=np.float32)
    projected_y = np.asarray(projected_y, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)

    one_dimensional = {
        "xs": xs,
        "ys": ys,
        "projected_x": projected_x,
        "projected_y": projected_y,
    }
    for name, values in one_dimensional.items():
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional; received {values.shape}."
            )

    pixel_count = len(xs)
    lengths = {name: len(values) for name, values in one_dimensional.items()}
    if any(length != pixel_count for length in lengths.values()):
        raise ValueError(
            "Pixel-coordinate arrays must have equal lengths; "
            f"received {lengths}."
        )
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError(
            "centers must have shape [N,2] or [N,3]; "
            f"received {centers.shape}."
        )

    center_count = int(centers.shape[0])
    if pixel_count == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    if center_count == 0:
        raise ValueError("Cannot assign foreground pixels without any centres.")
    if not np.all(np.isfinite(projected_x)):
        raise ValueError("projected_x contains NaN or infinite values.")
    if not np.all(np.isfinite(projected_y)):
        raise ValueError("projected_y contains NaN or infinite values.")
    if not np.all(np.isfinite(centers[:, :2])):
        raise ValueError("Centre coordinates contain NaN or infinite values.")

    configured_chunk_size = int(ASSIGNMENT_CHUNK_SIZE)
    if configured_chunk_size < 1:
        raise ValueError("ASSIGNMENT_CHUNK_SIZE must be >= 1.")
    maximum_working_memory_mb = float(maximum_working_memory_mb)
    if not np.isfinite(maximum_working_memory_mb) or maximum_working_memory_mb <= 0:
        raise ValueError("maximum_working_memory_mb must be finite and positive.")

    # Each pixel-centre pair temporarily needs two float32 arrays (dx² and
    # dy²). Twelve bytes per pair includes a conservative margin for NumPy's
    # reduction and indexing workspaces.
    memory_budget_bytes = int(maximum_working_memory_mb * 1024 * 1024)
    memory_limited_chunk = max(
        1,
        memory_budget_bytes // max(12 * center_count, 1),
    )
    chunk_size = min(
        pixel_count,
        configured_chunk_size,
        memory_limited_chunk,
    )

    nearest = np.empty(pixel_count, dtype=np.int32)
    distance = np.empty(pixel_count, dtype=np.float32)
    center_x = np.ascontiguousarray(centers[:, 0], dtype=np.float32)
    center_y = np.ascontiguousarray(centers[:, 1], dtype=np.float32)

    for start in range(0, pixel_count, chunk_size):
        end = min(pixel_count, start + chunk_size)

        # Square in place to avoid creating four full distance temporaries.
        distances_squared = (
            projected_x[start:end, None] - center_x[None, :]
        )
        np.square(distances_squared, out=distances_squared)

        dy_squared = projected_y[start:end, None] - center_y[None, :]
        np.square(dy_squared, out=dy_squared)
        distances_squared += dy_squared
        del dy_squared

        local_nearest = np.argmin(distances_squared, axis=1).astype(
            np.int32,
            copy=False,
        )
        row_indices = np.arange(end - start, dtype=np.intp)
        chosen_squared = distances_squared[
            row_indices,
            local_nearest,
        ].astype(np.float32, copy=True)
        np.sqrt(chosen_squared, out=chosen_squared)

        nearest[start:end] = local_nearest
        distance[start:end] = chosen_squared

    return nearest, distance


def add_partitioned_instances(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
    instance_scores: dict[int, float],
    candidate: np.ndarray,
    boundary_probability: np.ndarray,
    class_id: int,
    next_instance_id: int,
    center_score: float,
    score_fn=None,
) -> int:
    """Write valid connected partitions using deterministic, overflow-safe IDs.

    ``center_score`` is the fallback detection confidence for every instance
    written here. When ``score_fn`` is given it is called per component as
    ``score_fn(component_mask_crop, y0, x0)`` and its value is used instead.

    Scoring per component rather than per candidate matters: `boundary_partition`
    can split one candidate into several instances, and under the old contract
    all of them inherited a single confidence -- so a rim fragment was emitted
    carrying exactly the confidence of the real object beside it, which is
    precisely the ordering mask AP is computed from.

    Components are processed inside their bounding boxes instead of repeatedly
    scanning the complete image. Any overlap with an existing instance is
    treated as a decoder error rather than silently overwriting earlier output.
    """
    instance_map = np.asarray(instance_map)
    candidate = np.asarray(candidate, dtype=bool)
    class_id = int(class_id)
    next_instance_id = int(next_instance_id)

    if instance_map.ndim != 2:
        raise ValueError(
            f"instance_map must be two-dimensional; received {instance_map.shape}."
        )
    if not np.issubdtype(instance_map.dtype, np.integer):
        raise TypeError(
            f"instance_map must use an integer dtype; received {instance_map.dtype}."
        )
    if candidate.ndim != 2 or candidate.shape != instance_map.shape:
        raise ValueError(
            "candidate must be a 2-D mask matching instance_map; "
            f"received {candidate.shape} and {instance_map.shape}."
        )
    if not isinstance(class_by_instance, dict):
        raise TypeError("class_by_instance must be a dictionary.")
    if not isinstance(instance_scores, dict):
        raise TypeError("instance_scores must be a dictionary.")
    center_score = float(center_score)
    if not np.isfinite(center_score) or not 0.0 <= center_score <= 1.0:
        raise ValueError(
            f"center_score must be a finite value in [0,1]; got {center_score}."
        )
    if class_id <= 0 or class_id >= int(NUM_CLASSES):
        raise ValueError(
            f"class_id must be in [1, {int(NUM_CLASSES) - 1}]; "
            f"received {class_id}."
        )
    if next_instance_id < 1:
        raise ValueError("next_instance_id must be >= 1.")

    dtype_maximum = int(np.iinfo(instance_map.dtype).max)
    maximum_instance_id = min(int(MAX_INSTANCE_ID), dtype_maximum)
    if maximum_instance_id < 1:
        raise ValueError("The configured instance-ID range is invalid.")

    existing_map_maximum = (
        int(np.max(instance_map)) if instance_map.size else 0
    )
    existing_dictionary_maximum = max(
        (int(key) for key in class_by_instance),
        default=0,
    )
    existing_maximum = max(existing_map_maximum, existing_dictionary_maximum)
    if next_instance_id <= existing_maximum:
        raise ValueError(
            f"next_instance_id={next_instance_id} is not greater than the "
            f"existing maximum ID {existing_maximum}."
        )

    if np.any(candidate & (instance_map != 0)):
        raise RuntimeError(
            "Candidate pixels overlap an instance that was already written."
        )

    minimum_area = int(MIN_INSTANCE_AREA)
    if minimum_area < 1:
        raise ValueError("MIN_INSTANCE_AREA must be >= 1.")

    partitions = boundary_partition(candidate, boundary_probability)
    component_records: list[tuple] = []

    for partition_index, partition in enumerate(partitions):
        partition = np.asarray(partition, dtype=bool)
        if partition.shape != candidate.shape:
            raise RuntimeError(
                "boundary_partition returned an incorrectly shaped mask: "
                f"{partition.shape}; expected {candidate.shape}."
            )
        part_y, part_x = np.nonzero(partition)
        if len(part_x) == 0:
            continue

        crop_x0 = int(part_x.min())
        crop_x1 = int(part_x.max()) + 1
        crop_y0 = int(part_y.min())
        crop_y1 = int(part_y.max()) + 1
        partition_crop = partition[crop_y0:crop_y1, crop_x0:crop_x1]

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            partition_crop.astype(np.uint8),
            connectivity=8,
        )

        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue

            local_x = int(stats[component_id, cv2.CC_STAT_LEFT])
            local_y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            global_x = crop_x0 + local_x
            global_y = crop_y0 + local_y

            # Records sort top-to-bottom, left-to-right, then prefer the larger
            # component for exact positional ties.
            component_records.append(
                (
                    global_y,
                    global_x,
                    -area,
                    partition_index,
                    component_id,
                    crop_y0,
                    crop_x0,
                    local_y,
                    local_x,
                    height,
                    width,
                    labels,
                )
            )

    # A candidate is one object, or several touching objects that the
    # boundary ridge separated into comparable pieces. A component far smaller
    # than the largest one in the same candidate is a rim fragment, and
    # emitting it would create a false positive carrying exactly the
    # confidence of the real detection beside it. The floor spans the whole
    # candidate, not one partition: boundary_partition puts a disconnected
    # fragment in a partition of its own, where a per-partition floor could
    # never see how small it is.
    if component_records:
        largest_component_area = max(
            -record[2] for record in component_records
        )
        area_floor = max(
            minimum_area,
            int(
                math.ceil(
                    float(MIN_COMPONENT_AREA_FRACTION)
                    * largest_component_area
                )
            ),
        )
        component_records = [
            record
            for record in component_records
            if -record[2] >= area_floor
        ]

    component_records.sort(key=lambda record: record[:5])

    for (
        global_y,
        global_x,
        negative_area,
        _,
        component_id,
        crop_y0,
        crop_x0,
        local_y,
        local_x,
        height,
        width,
        labels,
    ) in component_records:
        del global_y, global_x, negative_area

        local_x0 = local_x
        local_x1 = local_x + width
        local_y0 = local_y
        local_y1 = local_y + height
        component_local = (
            labels[local_y0:local_y1, local_x0:local_x1] == component_id
        )

        target_y0 = crop_y0 + local_y0
        target_y1 = crop_y0 + local_y1
        target_x0 = crop_x0 + local_x0
        target_x1 = crop_x0 + local_x1
        target_view = instance_map[target_y0:target_y1, target_x0:target_x1]

        if np.any(target_view[component_local] != 0):
            raise RuntimeError(
                "A decoded component overlaps an existing instance ID."
            )
        if next_instance_id > maximum_instance_id:
            raise OverflowError(
                "Decoded more than "
                f"{maximum_instance_id:,} instances in one image; "
                f"dtype {instance_map.dtype} cannot represent another ID."
            )
        if next_instance_id in class_by_instance:
            raise RuntimeError(
                f"Instance ID {next_instance_id} already exists in "
                "class_by_instance."
            )
        if next_instance_id in instance_scores:
            raise RuntimeError(
                f"Instance ID {next_instance_id} already exists in "
                "instance_scores."
            )

        target_view[component_local] = next_instance_id
        class_by_instance[next_instance_id] = class_id
        if score_fn is None:
            instance_scores[next_instance_id] = center_score
        else:
            component_score = float(
                score_fn(component_local, target_y0, target_x0)
            )
            if not np.isfinite(component_score):
                component_score = 0.0
            instance_scores[next_instance_id] = float(
                min(max(component_score, 0.0), 1.0)
            )
        next_instance_id += 1

    return next_instance_id

def decode_instances(
    semantic_probabilities: np.ndarray,
    inner_distance_small: np.ndarray,
    boundary_probability: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray, dict[int, float]]:
    """Decode instances from the semantic map and the inner-distance field.

    Replaces the centre/offset decoder. The return signature is unchanged --
    ``(instance_map, class_by_instance, semantic_confidence,
    instance_scores)`` -- so every call site, the mask-AP record builder and
    `summarise_instances` keep working.

    What disappears with the offset field:

    * the 51-head-pixel assignment cap, which made a large object's outer
      pixels unassignable however good the prediction;
    * the unassigned residue and therefore the fallback path, measured at 75
      false positives against 2 true positives on validation;
    * centre peaks, and with them `CENTER_CONFIDENCE_THRESHOLD`, whose 0.10
      default admitted a spurious low-confidence duplicate on top of most real
      pads.

    The fourth return value is the per-instance detection score. It comes from
    `inner_distance_instance_score`: the agreement between the predicted field
    over an instance and the field that instance's own shape implies, times mean
    semantic confidence. Peak predicted distance was tried first and does not
    work -- the target normalises every instance to peak at 1.0, so the peak is
    ~1.0 for a perfect object and a rim sliver alike. See that function for the
    measurement.

    Grouping happens at full resolution. The field is predicted at
    INSTANCE_HEAD_SIZE and bilinearly upsampled, which is sound because a
    distance transform is smooth by construction -- unlike a vector field,
    where interpolating between two pixels pointing at different centres is
    meaningless.
    """
    semantic_probabilities = np.asarray(semantic_probabilities, dtype=np.float32)
    inner_distance_small = np.asarray(inner_distance_small, dtype=np.float32)
    boundary_probability = np.asarray(boundary_probability, dtype=np.float32)

    if boundary_probability.ndim == 3 and boundary_probability.shape[-1] == 1:
        boundary_probability = boundary_probability[..., 0]
    if inner_distance_small.ndim == 2:
        inner_distance_small = inner_distance_small[..., None]
    if semantic_probabilities.ndim != 3:
        raise ValueError(
            "semantic_probabilities must be [H,W,C]; received "
            f"{semantic_probabilities.shape}."
        )
    if semantic_probabilities.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"semantic_probabilities must have {NUM_CLASSES} channels; "
            f"received {semantic_probabilities.shape[-1]}."
        )
    if inner_distance_small.shape[-1] != 1:
        raise ValueError(
            "inner_distance must have one channel; received "
            f"{inner_distance_small.shape[-1]}."
        )

    height, width = semantic_probabilities.shape[:2]
    if boundary_probability.shape != (height, width):
        raise ValueError(
            "boundary_probability does not match the semantic resolution: "
            f"{boundary_probability.shape} versus {(height, width)}."
        )

    distance_full = resize_channels(inner_distance_small, width, height)[..., 0]
    semantic_labels = np.argmax(semantic_probabilities, axis=-1).astype(np.int32)
    semantic_confidence = np.max(semantic_probabilities, axis=-1).astype(np.float32)

    # A pixel must be confidently foreground to take part at all, the same
    # gate the previous decoder applied before grouping.
    confident = semantic_confidence >= float(SEMANTIC_CONFIDENCE_THRESHOLD)

    instance_map = np.zeros((height, width), dtype=INSTANCE_ID_DTYPE)
    class_by_instance: dict[int, int] = {}
    instance_scores: dict[int, float] = {}
    next_instance_id = 1

    for class_id in range(1, NUM_CLASSES):
        foreground = (semantic_labels == class_id) & confident
        if not foreground.any():
            continue

        grouped = watershed_instances(foreground, distance_full)

        def score_component(component_crop, y0, x0):
            """Confidence for one decoded component, from its own bounding box."""
            height_c, width_c = component_crop.shape
            return inner_distance_instance_score(
                component_crop,
                distance_full[y0:y0 + height_c, x0:x0 + width_c],
                semantic_confidence[y0:y0 + height_c, x0:x0 + width_c],
            )

        for local_id in np.unique(grouped):
            if local_id <= 0:
                continue
            candidate = grouped == local_id
            next_instance_id = add_partitioned_instances(
                instance_map,
                class_by_instance,
                instance_scores,
                candidate,
                boundary_probability,
                class_id,
                next_instance_id,
                0.0,
                score_fn=score_component,
            )

    return (
        instance_map,
        class_by_instance,
        semantic_confidence,
        instance_scores,
    )


def decode_instances_centre_offset(
    semantic_probabilities: np.ndarray,
    center_probabilities_small: np.ndarray,
    offset_vectors_small: np.ndarray,
    boundary_probability: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray, dict[int, float]]:
    """
    Decode instances the V7 way, from centre peaks and an offset field.

    Retained unused so a V7 checkpoint can still be decoded and compared
    against; `decode_instances` below is what V7 runs. Not deleted because
    the published 0.7711 came from this function, and reproducing that number
    is the only way to show the replacement is an improvement rather than a
    different measurement.

    Centre peaks are detected at the native instance-head resolution and then
    transformed accurately to the full semantic resolution.

    Returns ``(instance_map, class_by_instance, semantic_confidence,
    instance_center_scores)``. The fourth value is the fused centre-peak score
    each instance was decoded from; it is the only per-instance quantity that
    separates a confident detection from a weak one, because the mean semantic
    probability saturates near 1.0 for real objects and fragments alike.
    """
    semantic_probabilities = np.asarray(
        semantic_probabilities,
        dtype=np.float32,
    )
    center_probabilities_small = np.asarray(
        center_probabilities_small,
        dtype=np.float32,
    )
    offset_vectors_small = np.asarray(
        offset_vectors_small,
        dtype=np.float32,
    )
    boundary_probability = np.asarray(
        boundary_probability,
        dtype=np.float32,
    )

    # Accept either [H,W] or [H,W,1].
    if (
        boundary_probability.ndim == 3
        and boundary_probability.shape[-1] == 1
    ):
        boundary_probability = boundary_probability[..., 0]

    expected_shapes = {
        "semantic": (
            IMG_SIZE,
            IMG_SIZE,
            NUM_CLASSES,
        ),
        "center": (
            INSTANCE_HEAD_SIZE,
            INSTANCE_HEAD_SIZE,
            NUM_CLASSES - 1,
        ),
        "offset": (
            INSTANCE_HEAD_SIZE,
            INSTANCE_HEAD_SIZE,
            2,
        ),
        "boundary": (
            IMG_SIZE,
            IMG_SIZE,
        ),
    }

    observed_shapes = {
        "semantic": semantic_probabilities.shape,
        "center": center_probabilities_small.shape,
        "offset": offset_vectors_small.shape,
        "boundary": boundary_probability.shape,
    }

    mismatches = {
        name: {
            "observed": observed_shapes[name],
            "expected": expected_shape,
        }
        for name, expected_shape in expected_shapes.items()
        if observed_shapes[name] != expected_shape
    }

    if mismatches:
        raise ValueError(
            "Model outputs do not match the decoder configuration: "
            f"{mismatches}"
        )

    tensors = {
        "semantic_probabilities": semantic_probabilities,
        "center_probabilities_small": center_probabilities_small,
        "offset_vectors_small": offset_vectors_small,
        "boundary_probability": boundary_probability,
    }

    invalid_tensors = [
        name
        for name, values in tensors.items()
        if not np.all(np.isfinite(values))
    ]

    if invalid_tensors:
        raise ValueError(
            "Model outputs contain NaN or infinite values: "
            f"{invalid_tensors}."
        )

    probability_tensors = {
        "semantic_probabilities": semantic_probabilities,
        "center_probabilities_small": center_probabilities_small,
        "boundary_probability": boundary_probability,
    }

    invalid_probability_ranges = {
        name: (
            float(values.min()),
            float(values.max()),
        )
        for name, values in probability_tensors.items()
        if (
            float(values.min()) < -1e-4
            or float(values.max()) > 1.0001
        )
    }

    if invalid_probability_ranges:
        raise ValueError(
            "Decoder expects probabilities in [0,1]. Apply softmax to "
            "semantic logits and sigmoid to centre/boundary logits first. "
            f"Invalid ranges: {invalid_probability_ranges}"
        )

    if (
        float(offset_vectors_small.min()) < -1.0001
        or float(offset_vectors_small.max()) > 1.0001
    ):
        raise ValueError(
            "Offset predictions must be in [-1,1]. Ensure the offset "
            "head uses tanh activation."
        )

    semantic_probabilities = np.clip(
        semantic_probabilities,
        0.0,
        1.0,
    )

    center_probabilities_small = np.clip(
        center_probabilities_small,
        0.0,
        1.0,
    )

    boundary_probability = np.clip(
        boundary_probability,
        0.0,
        1.0,
    )

    offset_vectors_small = np.clip(
        offset_vectors_small,
        -1.0,
        1.0,
    )

    # Check that semantic probabilities came from softmax.
    semantic_sum = np.sum(
        semantic_probabilities,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    if float(semantic_sum.min()) <= 1e-6:
        raise ValueError(
            "At least one semantic probability vector sums to zero."
        )

    maximum_sum_error = float(
        np.max(np.abs(semantic_sum - 1.0))
    )

    if maximum_sum_error > 0.05:
        raise ValueError(
            "Semantic channels do not sum to one. Apply softmax before "
            f"decoding. Maximum sum error: {maximum_sum_error:.6f}."
        )

    # Correct small floating-point normalization errors.
    semantic_probabilities = (
        semantic_probabilities / semantic_sum
    )

    height, width = semantic_probabilities.shape[:2]
    head_height, head_width = center_probabilities_small.shape[:2]

    if head_height <= 0 or head_width <= 0:
        raise ValueError(
            "Instance-head dimensions must be positive."
        )

    # Offset vectors are dense normalized vector fields and may be
    # interpolated safely.
    offset_vectors = resize_channels(
        offset_vectors_small,
        width,
        height,
    )

    semantic_labels = np.argmax(
        semantic_probabilities,
        axis=-1,
    ).astype(np.uint8)

    semantic_confidence = np.max(
        semantic_probabilities,
        axis=-1,
    ).astype(np.float32)

    foreground = (
        (semantic_labels > 0)
        & (
            semantic_confidence
            >= float(SEMANTIC_CONFIDENCE_THRESHOLD)
        )
    )

    instance_dtype = np.dtype(INSTANCE_ID_DTYPE)

    if not np.issubdtype(instance_dtype, np.integer):
        raise TypeError(
            "INSTANCE_ID_DTYPE must be an integer dtype; "
            f"received {instance_dtype}."
        )

    instance_map = np.zeros(
        (height, width),
        dtype=instance_dtype,
    )

    class_by_instance: dict[int, int] = {}
    instance_center_scores: dict[int, float] = {}
    next_instance_id = 1

    # CENTER_NMS_RADIUS is expressed in instance-head pixels, which is the
    # resolution the peaks are actually detected at, so it is used directly.
    native_nms_radius = max(1, int(CENTER_NMS_RADIUS))

    for class_id in range(1, NUM_CLASSES):
        class_pixels = (
            foreground
            & (semantic_labels == class_id)
        )

        if not np.any(class_pixels):
            continue

        # Downsample semantic support to the native centre-head size.
        semantic_class_small = cv2.resize(
            semantic_probabilities[..., class_id],
            (head_width, head_height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        # Combine class-specific centre confidence with semantic support.
        fused_heatmap_small = (
            center_probabilities_small[..., class_id - 1]
            * np.sqrt(
                np.clip(
                    semantic_class_small,
                    0.0,
                    1.0,
                )
            )
        )

        # Preserve small semantic objects while determining where centres
        # are allowed.
        class_support_small = cv2.resize(
            class_pixels.astype(np.float32),
            (head_width, head_height),
            interpolation=cv2.INTER_AREA,
        ) > 0.01

        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        allowed_centers_small = cv2.dilate(
            class_support_small.astype(np.uint8),
            support_kernel,
            iterations=1,
        ).astype(bool)

        fused_heatmap_small = np.where(
            allowed_centers_small,
            fused_heatmap_small,
            0.0,
        ).astype(np.float32)

        # Detect centres directly at their trained resolution.
        centers_small = find_center_peaks(
            fused_heatmap_small,
            nms_radius=native_nms_radius,
            confidence_threshold=float(
                CENTER_CONFIDENCE_THRESHOLD
            ),
            maximum_centers=int(
                MAX_CENTERS_PER_CLASS
            ),
        )

        # If the centre head finds nothing, fall back to boundaries and
        # connected components.
        if len(centers_small) == 0:
            next_instance_id = add_partitioned_instances(
                instance_map,
                class_by_instance,
                instance_center_scores,
                class_pixels,
                boundary_probability,
                class_id,
                next_instance_id,
                center_score=float(FALLBACK_INSTANCE_SCORE),
            )
            continue

        # Convert centre-head coordinates to full-resolution coordinates
        # using OpenCV's pixel-centre convention.
        centers = centers_small.copy()

        centers[:, 0] = (
            (centers_small[:, 0] + 0.5)
            * width
            / float(head_width)
            - 0.5
        )

        centers[:, 1] = (
            (centers_small[:, 1] + 0.5)
            * height
            / float(head_height)
            - 0.5
        )

        centers[:, 0] = np.clip(
            centers[:, 0],
            0.0,
            width - 1.0,
        )

        centers[:, 1] = np.clip(
            centers[:, 1],
            0.0,
            height - 1.0,
        )

        ys, xs = np.nonzero(class_pixels)

        xs_float = xs.astype(np.float32)
        ys_float = ys.astype(np.float32)

        # Convert normalized offsets into full-resolution pixel offsets.
        projected_x = (
            xs_float
            + offset_vectors[ys, xs, 0] * float(width)
        )

        projected_y = (
            ys_float
            + offset_vectors[ys, xs, 1] * float(height)
        )

        nearest_center, nearest_distance = (
            nearest_center_assignments(
                xs,
                ys,
                projected_x,
                projected_y,
                centers,
            )
        )

        assigned = np.zeros(
            len(xs),
            dtype=bool,
        )

        maximum_assignment_distance = float(
            MAX_CENTER_ASSIGNMENT_DISTANCE
        )

        if maximum_assignment_distance <= 0:
            raise ValueError(
                "MAX_CENTER_ASSIGNMENT_DISTANCE must be positive."
            )

        for center_index in range(len(centers)):
            selected = (
                (nearest_center == center_index)
                & (
                    nearest_distance
                    <= maximum_assignment_distance
                )
            )

            if not np.any(selected):
                continue

            candidate = np.zeros(
                (height, width),
                dtype=bool,
            )

            candidate[
                ys[selected],
                xs[selected],
            ] = True

            next_instance_id = add_partitioned_instances(
                instance_map,
                class_by_instance,
                instance_center_scores,
                candidate,
                boundary_probability,
                class_id,
                next_instance_id,
                center_score=float(
                    np.clip(centers[center_index, 2], 0.0, 1.0)
                ),
            )

            assigned[selected] = True

        # Pixels whose predicted offset lands nowhere near a centre. Minting an
        # instance from them costs 75 false positives per 2 true positives
        # (measured, see UNASSIGNED_PIXEL_POLICY), because the residue is
        # object rims and specular highlights rather than missed objects.
        if UNASSIGNED_PIXEL_POLICY == "new_instances" and np.any(~assigned):
            fallback = np.zeros(
                (height, width),
                dtype=bool,
            )

            fallback[
                ys[~assigned],
                xs[~assigned],
            ] = True

            next_instance_id = add_partitioned_instances(
                instance_map,
                class_by_instance,
                instance_center_scores,
                fallback,
                boundary_probability,
                class_id,
                next_instance_id,
                center_score=float(FALLBACK_INSTANCE_SCORE),
            )
        elif UNASSIGNED_PIXEL_POLICY not in ("drop", "new_instances"):
            raise ValueError(
                "UNASSIGNED_PIXEL_POLICY must be 'drop' or 'new_instances'; "
                f"received {UNASSIGNED_PIXEL_POLICY!r}."
            )

    return (
        instance_map,
        class_by_instance,
        semantic_confidence,
        instance_center_scores,
    )


# =============================================================================
# 11. SEMANTIC AND INSTANCE EVALUATION
# =============================================================================
def unpack_model_outputs(outputs) -> dict[str, np.ndarray]:
    """Validate and convert the three model outputs to float32 NumPy arrays."""
    required_names = (
        "semantic",
        "inner_distance",
        "boundary",
    )

    if not isinstance(outputs, dict):
        if isinstance(outputs, (list, tuple)):
            raise TypeError(
                "The model returned a list/tuple, whose output order is "
                "ambiguous. Build the model with dictionary outputs named: "
                "semantic, inner_distance, boundary."
            )
        raise TypeError(
            f"Expected dictionary model outputs, received {type(outputs)}."
        )

    observed_names = set(outputs)
    required_set = set(required_names)

    missing = sorted(required_set - observed_names)
    unexpected = sorted(observed_names - required_set)

    if missing or unexpected:
        raise KeyError(
            "Incorrect model output names: "
            f"missing={missing}, unexpected={unexpected}."
        )

    arrays: dict[str, np.ndarray] = {}

    for name in required_names:
        value = outputs[name]

        if hasattr(value, "numpy"):
            value = value.numpy()

        array = np.asarray(value, dtype=np.float32)

        if array.ndim != 4:
            raise ValueError(
                f"Output {name!r} must have shape [B,H,W,C]; "
                f"received {array.shape}."
            )

        if array.shape[0] <= 0:
            raise ValueError(
                f"Output {name!r} has an empty batch."
            )

        if not np.all(np.isfinite(array)):
            raise FloatingPointError(
                f"Output {name!r} contains NaN or infinity."
            )

        arrays[name] = np.ascontiguousarray(array)

    expected_channels = {
        "semantic": NUM_CLASSES,
        "inner_distance": 1,
        "boundary": 1,
    }

    for name, channels in expected_channels.items():
        observed_channels = arrays[name].shape[-1]

        if observed_channels != channels:
            raise ValueError(
                f"Output {name!r} has {observed_channels} channels; "
                f"expected {channels}. Shape: {arrays[name].shape}."
            )

    batch_sizes = {
        name: array.shape[0]
        for name, array in arrays.items()
    }

    if len(set(batch_sizes.values())) != 1:
        raise ValueError(
            f"Model output batch sizes disagree: {batch_sizes}."
        )

    semantic_shape = arrays["semantic"].shape[1:3]
    boundary_shape = arrays["boundary"].shape[1:3]
    distance_shape = arrays["inner_distance"].shape[1:3]

    if semantic_shape != boundary_shape:
        raise ValueError(
            "Semantic and boundary spatial shapes differ: "
            f"{semantic_shape} versus {boundary_shape}."
        )

    # The distance head runs at half the semantic resolution by design. Assert
    # the ratio rather than leave it implied: a head accidentally built at full
    # resolution would still decode, just with the core threshold acting on a
    # differently-scaled field, and nothing downstream would complain.
    if (
        distance_shape[0] * 2 != semantic_shape[0]
        or distance_shape[1] * 2 != semantic_shape[1]
    ):
        raise ValueError(
            "inner_distance must be half the semantic resolution: "
            f"{distance_shape} versus {semantic_shape}."
        )

    return arrays


def output_probabilities(
    outputs: dict[str, np.ndarray],
    batch_position: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Convert one batch position from logits into stable probabilities."""
    arrays = unpack_model_outputs(outputs)

    if (
        isinstance(batch_position, bool)
        or not isinstance(batch_position, (int, np.integer))
    ):
        raise TypeError(
            "batch_position must be an integer."
        )

    batch_position = int(batch_position)
    batch_size = arrays["semantic"].shape[0]

    if not 0 <= batch_position < batch_size:
        raise IndexError(
            f"batch_position={batch_position} is outside "
            f"the available batch range 0..{batch_size - 1}."
        )

    semantic_logits = arrays["semantic"][batch_position]
    # The head already applies a sigmoid, so this is a probability in
    # [0,1] rather than a logit and needs no further activation.
    inner_distance = arrays["inner_distance"][batch_position].copy()
    boundary_logits = arrays[
        "boundary"
    ][batch_position, ..., 0]

    # Float64 is used temporarily to prevent overflow when logits
    # have a large numerical range.
    semantic_logits_64 = semantic_logits.astype(
        np.float64,
        copy=False,
    )

    shifted_logits = (
        semantic_logits_64
        - np.max(
            semantic_logits_64,
            axis=-1,
            keepdims=True,
        )
    )

    semantic_exponentials = np.exp(shifted_logits)

    semantic = (
        semantic_exponentials
        / np.sum(
            semantic_exponentials,
            axis=-1,
            keepdims=True,
        )
    ).astype(np.float32)

    def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
        logits_64 = np.asarray(
            logits,
            dtype=np.float64,
        )

        probabilities = np.empty_like(
            logits_64,
            dtype=np.float64,
        )

        positive = logits_64 >= 0.0
        negative = ~positive

        probabilities[positive] = (
            1.0
            / (
                1.0
                + np.exp(-logits_64[positive])
            )
        )

        negative_exponential = np.exp(
            logits_64[negative]
        )

        probabilities[negative] = (
            negative_exponential
            / (1.0 + negative_exponential)
        )

        return probabilities.astype(np.float32)

    boundary = stable_sigmoid(boundary_logits)

    if not np.all(np.isfinite(inner_distance)):
        raise FloatingPointError(
            "Inner-distance output contains NaN or infinity."
        )

    distance_minimum = float(inner_distance.min())
    distance_maximum = float(inner_distance.max())

    # The V7 inner-distance head uses sigmoid, so its output is
    # already a normalised distance in [0,1] and is not passed through
    # another activation here. A value outside that range means the head
    # was rebuilt with a different activation, which would silently
    # change what the core threshold means.
    if distance_minimum < -0.0001 or distance_maximum > 1.0001:
        raise ValueError(
            "Inner-distance predictions must be in [0,1]. "
            f"Observed [{distance_minimum:.6g}, "
            f"{distance_maximum:.6g}]. Verify that the inner_distance "
            "head uses sigmoid activation."
        )

    inner_distance = np.clip(
        inner_distance,
        0.0,
        1.0,
    ).astype(np.float32)

    semantic_sums = np.sum(
        semantic,
        axis=-1,
    )

    if not np.allclose(
        semantic_sums,
        1.0,
        atol=2e-5,
        rtol=2e-5,
    ):
        raise FloatingPointError(
            "Semantic softmax probabilities do not sum to one."
        )

    return (
        np.ascontiguousarray(semantic),
        np.ascontiguousarray(inner_distance),
        np.ascontiguousarray(boundary),
    )


def ground_truth_instance_classes(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    minimum_class_purity: float = 0.95,
) -> dict[int, int]:
    """Assign each ground-truth instance its dominant foreground class."""
    instances = np.asarray(instance_map)
    semantic = np.asarray(semantic_map)

    if instances.ndim == 3 and instances.shape[-1] == 1:
        instances = instances[..., 0]

    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]

    if instances.ndim != 2:
        raise ValueError(
            "instance_map must have shape [H,W] or [H,W,1]; "
            f"received {instances.shape}."
        )

    if semantic.ndim != 2:
        raise ValueError(
            "semantic_map must have shape [H,W] or [H,W,1]; "
            f"received {semantic.shape}."
        )

    if instances.shape != semantic.shape:
        raise ValueError(
            "Instance and semantic map shapes differ: "
            f"{instances.shape} versus {semantic.shape}."
        )

    if not np.issubdtype(
        instances.dtype,
        np.integer,
    ):
        raise TypeError(
            f"instance_map must use an integer dtype, "
            f"received {instances.dtype}."
        )

    if not np.issubdtype(
        semantic.dtype,
        np.integer,
    ):
        raise TypeError(
            f"semantic_map must use an integer dtype, "
            f"received {semantic.dtype}."
        )

    if not 0.0 <= minimum_class_purity <= 1.0:
        raise ValueError(
            "minimum_class_purity must be in [0,1]."
        )

    instances = instances.astype(
        np.int64,
        copy=False,
    )

    semantic = semantic.astype(
        np.int32,
        copy=False,
    )

    if instances.size == 0:
        return {}

    minimum_instance_id = int(instances.min())
    maximum_instance_id = int(instances.max())

    if minimum_instance_id < 0:
        raise ValueError(
            "instance_map contains a negative instance ID."
        )

    if maximum_instance_id > MAX_INSTANCE_ID:
        raise OverflowError(
            f"instance_map contains ID {maximum_instance_id:,}, "
            f"which exceeds MAX_INSTANCE_ID={MAX_INSTANCE_ID:,}."
        )

    minimum_class_id = int(semantic.min())
    maximum_class_id = int(semantic.max())

    if (
        minimum_class_id < 0
        or maximum_class_id >= NUM_CLASSES
    ):
        invalid_classes = np.unique(
            semantic[
                (semantic < 0)
                | (semantic >= NUM_CLASSES)
            ]
        )

        raise ValueError(
            f"Semantic IDs must be in 0..{NUM_CLASSES - 1}; "
            f"found {invalid_classes[:16].tolist()}."
        )

    instance_foreground = instances > 0
    semantic_foreground = semantic > 0

    background_inside_instances = (
        instance_foreground
        & (~semantic_foreground)
    )

    if np.any(background_inside_instances):
        raise ValueError(
            "Ground-truth data is inconsistent: "
            f"{int(np.count_nonzero(background_inside_instances)):,} "
            "instance pixels are labelled as semantic background."
        )

    semantic_without_instance = (
        semantic_foreground
        & (~instance_foreground)
    )

    if np.any(semantic_without_instance):
        raise ValueError(
            "Ground-truth data is inconsistent: "
            f"{int(np.count_nonzero(semantic_without_instance)):,} "
            "foreground semantic pixels have no instance ID."
        )

    present_ids = np.unique(
        instances[instance_foreground]
    )

    if present_ids.size == 0:
        return {}

    encoded_pairs = (
        instances[instance_foreground]
        * NUM_CLASSES
        + semantic[instance_foreground]
    )

    pair_counts = np.bincount(
        encoded_pairs,
        minlength=(
            (maximum_instance_id + 1)
            * NUM_CLASSES
        ),
    ).reshape(
        maximum_instance_id + 1,
        NUM_CLASSES,
    )

    result: dict[int, int] = {}

    for raw_instance_id in present_ids:
        instance_id = int(raw_instance_id)

        foreground_counts = pair_counts[
            instance_id,
            1:,
        ]

        foreground_pixel_count = int(
            foreground_counts.sum()
        )

        if foreground_pixel_count <= 0:
            raise ValueError(
                f"Instance {instance_id} has no foreground "
                "semantic pixels."
            )

        class_id = (
            int(np.argmax(foreground_counts))
            + 1
        )

        dominant_count = int(
            foreground_counts[class_id - 1]
        )

        purity = (
            dominant_count
            / foreground_pixel_count
        )

        if purity < minimum_class_purity:
            raise ValueError(
                f"Instance {instance_id} mixes semantic classes. "
                f"Dominant class={class_id}, purity={purity:.4f}, "
                f"required={minimum_class_purity:.4f}, "
                f"class counts={foreground_counts.tolist()}."
            )

        result[instance_id] = class_id

    return result


def match_instances_at_iou(
    predicted_map: np.ndarray,
    predicted_classes: dict[int, int],
    target_map: np.ndarray,
    target_classes: dict[int, int],
    threshold: float,
) -> dict[int, dict[str, int]]:
    """Maximum-cardinality same-class matching at an IoU threshold."""

    if (
        not np.isfinite(threshold)
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError(
            f"threshold must be in (0,1], received {threshold}."
        )

    threshold = float(threshold)

    def validate_map(
        array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(array)

        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]

        if values.ndim != 2:
            raise ValueError(
                f"{name} must have shape [H,W] or [H,W,1]; "
                f"received {values.shape}."
            )

        if not np.issubdtype(
            values.dtype,
            np.integer,
        ):
            raise TypeError(
                f"{name} must use an integer dtype, "
                f"received {values.dtype}."
            )

        values = values.astype(
            np.int64,
            copy=False,
        )

        if values.size:
            minimum = int(values.min())
            maximum = int(values.max())

            if minimum < 0:
                raise ValueError(
                    f"{name} contains a negative instance ID."
                )

            if maximum > MAX_INSTANCE_ID:
                raise OverflowError(
                    f"{name} contains ID {maximum:,}, exceeding "
                    f"MAX_INSTANCE_ID={MAX_INSTANCE_ID:,}."
                )

        return values

    def validate_mapping(
        mapping: dict[int, int],
        present_ids: set[int],
        name: str,
    ) -> dict[int, int]:
        normalized: dict[int, int] = {}

        for raw_instance_id, raw_class_id in mapping.items():
            instance_id = int(raw_instance_id)
            class_id = int(raw_class_id)

            if instance_id <= 0:
                raise ValueError(
                    f"{name} contains invalid instance ID "
                    f"{instance_id}."
                )

            if instance_id > MAX_INSTANCE_ID:
                raise OverflowError(
                    f"{name} instance ID {instance_id:,} exceeds "
                    f"MAX_INSTANCE_ID={MAX_INSTANCE_ID:,}."
                )

            if not 1 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"{name} instance {instance_id} has invalid "
                    f"class ID {class_id}."
                )

            if instance_id in normalized:
                raise ValueError(
                    f"{name} contains duplicate normalized "
                    f"instance ID {instance_id}."
                )

            normalized[instance_id] = class_id

        mapped_ids = set(normalized)

        missing = sorted(
            present_ids - mapped_ids
        )

        stale = sorted(
            mapped_ids - present_ids
        )

        if missing or stale:
            raise ValueError(
                f"{name} disagrees with its instance map: "
                f"unmapped IDs={missing[:16]}, "
                f"absent mapped IDs={stale[:16]}."
            )

        return normalized

    def maximum_matching(
        adjacency: list[list[tuple[int, float]]],
        right_count: int,
    ) -> int:
        """
        Find a maximum-cardinality bipartite matching.

        Neighbours are ordered by descending IoU so equally sized
        maximum matchings prefer stronger overlaps.
        """
        left_count = len(adjacency)

        left_match = np.full(
            left_count,
            -1,
            dtype=np.int32,
        )

        right_match = np.full(
            right_count,
            -1,
            dtype=np.int32,
        )

        left_order = sorted(
            range(left_count),
            key=lambda left: (
                -max(
                    (
                        iou
                        for _, iou
                        in adjacency[left]
                    ),
                    default=-1.0,
                ),
                left,
            ),
        )

        for starting_left in left_order:
            if left_match[starting_left] >= 0:
                continue

            seen_left = np.zeros(
                left_count,
                dtype=bool,
            )

            parent_right = np.full(
                right_count,
                -1,
                dtype=np.int32,
            )

            queue = [starting_left]
            seen_left[starting_left] = True
            queue_position = 0
            unmatched_right = -1

            while (
                queue_position < len(queue)
                and unmatched_right < 0
            ):
                left = queue[queue_position]
                queue_position += 1

                for right, _ in adjacency[left]:
                    if parent_right[right] >= 0:
                        continue

                    parent_right[right] = left
                    matched_left = int(
                        right_match[right]
                    )

                    if matched_left < 0:
                        unmatched_right = right
                        break

                    if not seen_left[matched_left]:
                        seen_left[matched_left] = True
                        queue.append(matched_left)

            if unmatched_right < 0:
                continue

            current_right = unmatched_right

            while current_right >= 0:
                current_left = int(
                    parent_right[current_right]
                )

                previous_right = int(
                    left_match[current_left]
                )

                left_match[current_left] = current_right
                right_match[current_right] = current_left

                current_right = previous_right

        return int(
            np.count_nonzero(left_match >= 0)
        )

    predicted = validate_map(
        predicted_map,
        "predicted_map",
    )

    target = validate_map(
        target_map,
        "target_map",
    )

    if predicted.shape != target.shape:
        raise ValueError(
            "Predicted and target instance-map shapes differ: "
            f"{predicted.shape} versus {target.shape}."
        )

    predicted_present_ids = {
        int(value)
        for value in np.unique(predicted)
        if int(value) > 0
    }

    target_present_ids = {
        int(value)
        for value in np.unique(target)
        if int(value) > 0
    }

    predicted_mapping = validate_mapping(
        predicted_classes,
        predicted_present_ids,
        "predicted_classes",
    )

    target_mapping = validate_mapping(
        target_classes,
        target_present_ids,
        "target_classes",
    )

    result = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for class_id in range(1, NUM_CLASSES)
    }

    maximum_predicted_id = (
        int(predicted.max())
        if predicted.size
        else 0
    )

    maximum_target_id = (
        int(target.max())
        if target.size
        else 0
    )

    predicted_areas = np.bincount(
        predicted.reshape(-1),
        minlength=maximum_predicted_id + 1,
    )

    target_areas = np.bincount(
        target.reshape(-1),
        minlength=maximum_target_id + 1,
    )

    overlapping = (
        (predicted > 0)
        & (target > 0)
    )

    intersections: dict[
        tuple[int, int],
        int,
    ] = {}

    if np.any(overlapping):
        pair_multiplier = (
            maximum_target_id + 1
        )

        pair_codes = (
            predicted[overlapping]
            * pair_multiplier
            + target[overlapping]
        )

        unique_codes, counts = np.unique(
            pair_codes,
            return_counts=True,
        )

        for code, count in zip(
            unique_codes.tolist(),
            counts.tolist(),
        ):
            predicted_id = int(
                code // pair_multiplier
            )

            target_id = int(
                code % pair_multiplier
            )

            intersections[
                (predicted_id, target_id)
            ] = int(count)

    for class_id in range(1, NUM_CLASSES):
        predicted_ids = sorted(
            instance_id
            for instance_id, mapped_class
            in predicted_mapping.items()
            if mapped_class == class_id
        )

        target_ids = sorted(
            instance_id
            for instance_id, mapped_class
            in target_mapping.items()
            if mapped_class == class_id
        )

        predicted_index = {
            instance_id: index
            for index, instance_id
            in enumerate(predicted_ids)
        }

        target_index = {
            instance_id: index
            for index, instance_id
            in enumerate(target_ids)
        }

        adjacency: list[
            list[tuple[int, float]]
        ] = [
            []
            for _ in predicted_ids
        ]

        for (
            predicted_id,
            target_id,
        ), intersection in intersections.items():
            if (
                predicted_id not in predicted_index
                or target_id not in target_index
            ):
                continue

            union = (
                int(predicted_areas[predicted_id])
                + int(target_areas[target_id])
                - intersection
            )

            if union <= 0:
                continue

            iou = intersection / union

            if iou >= threshold:
                adjacency[
                    predicted_index[predicted_id]
                ].append(
                    (
                        target_index[target_id],
                        float(iou),
                    )
                )

        for neighbours in adjacency:
            neighbours.sort(
                key=lambda item: (
                    -item[1],
                    target_ids[item[0]],
                )
            )

        true_positives = maximum_matching(
            adjacency,
            len(target_ids),
        )

        result[class_id]["tp"] = (
            true_positives
        )

        result[class_id]["fp"] = (
            len(predicted_ids)
            - true_positives
        )

        result[class_id]["fn"] = (
            len(target_ids)
            - true_positives
        )

    return result


def mask_ap_records_for_image(
    predicted_map: np.ndarray,
    predicted_classes: dict[int, int],
    predicted_scores: dict[int, float],
    target_map: np.ndarray,
    target_classes: dict[int, int],
    image_index: int,
) -> tuple[
    dict[int, list[dict[str, object]]],
    dict[int, int],
]:
    """Create compact confidence/IoU records for mask average precision.

    Only IoUs for same-class, overlapping prediction/target pairs are stored.
    This avoids retaining a full-resolution mask for every decoded instance
    while still allowing dataset-level confidence ranking at every AP IoU.
    """
    if (
        isinstance(image_index, bool)
        or not isinstance(image_index, (int, np.integer))
    ):
        raise TypeError("image_index must be an integer.")

    image_index = int(image_index)
    if image_index < 0:
        raise ValueError("image_index cannot be negative.")

    (
        predicted,
        normalized_predicted_classes,
        predicted_ids,
    ) = _validated_instance_inputs(
        predicted_map,
        predicted_classes,
    )
    (
        target,
        normalized_target_classes,
        target_ids,
    ) = _validated_instance_inputs(
        target_map,
        target_classes,
    )

    if predicted.shape != target.shape:
        raise ValueError(
            "Predicted and target instance-map shapes differ: "
            f"{predicted.shape} versus {target.shape}."
        )

    if not isinstance(predicted_scores, dict):
        raise TypeError("predicted_scores must be a dictionary.")

    normalized_scores: dict[int, float] = {}
    for raw_instance_id, raw_score in predicted_scores.items():
        if (
            isinstance(raw_instance_id, bool)
            or not isinstance(raw_instance_id, (int, np.integer))
        ):
            raise TypeError(
                "predicted_scores keys must be integer instance IDs."
            )
        instance_id = int(raw_instance_id)
        score = float(raw_score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Instance {instance_id} has invalid confidence {score}; "
                "expected a finite value in [0,1]."
            )
        if instance_id in normalized_scores:
            raise ValueError(
                f"predicted_scores contains duplicate instance ID "
                f"{instance_id}."
            )
        normalized_scores[instance_id] = score

    present_predicted_ids = {
        int(value)
        for value in predicted_ids.tolist()
    }
    score_ids = set(normalized_scores)
    missing_scores = sorted(present_predicted_ids - score_ids)
    stale_scores = sorted(score_ids - present_predicted_ids)
    if missing_scores or stale_scores:
        raise ValueError(
            "predicted_scores disagrees with predicted_map: "
            f"missing IDs={missing_scores[:16]}, "
            f"absent IDs={stale_scores[:16]}."
        )

    maximum_detections = int(
        MASK_AP_MAX_DETECTIONS_PER_IMAGE
    )
    if maximum_detections <= 0:
        raise ValueError(
            "MASK_AP_MAX_DETECTIONS_PER_IMAGE must be positive."
        )

    # Match Ultralytics' usual max_det behaviour: retain the strongest
    # predictions across all classes in this image before calculating AP.
    retained_predicted_ids = sorted(
        present_predicted_ids,
        key=lambda instance_id: (
            -normalized_scores[instance_id],
            instance_id,
        ),
    )[:maximum_detections]

    maximum_predicted_id = (
        int(predicted_ids.max())
        if predicted_ids.size
        else 0
    )
    maximum_target_id = (
        int(target_ids.max())
        if target_ids.size
        else 0
    )

    predicted_areas = np.bincount(
        predicted.reshape(-1),
        minlength=maximum_predicted_id + 1,
    ).astype(np.int64, copy=False)
    target_areas = np.bincount(
        target.reshape(-1),
        minlength=maximum_target_id + 1,
    ).astype(np.int64, copy=False)

    intersections: dict[tuple[int, int], int] = {}
    overlapping = (predicted > 0) & (target > 0)
    if np.any(overlapping):
        pair_multiplier = maximum_target_id + 1
        pair_codes = (
            predicted[overlapping] * pair_multiplier
            + target[overlapping]
        )
        unique_codes, counts = np.unique(
            pair_codes,
            return_counts=True,
        )
        for raw_code, raw_count in zip(
            unique_codes.tolist(),
            counts.tolist(),
        ):
            code = int(raw_code)
            predicted_id = code // pair_multiplier
            target_id = code % pair_multiplier
            intersections[(predicted_id, target_id)] = int(raw_count)

    target_ids_by_class = {
        class_id: sorted(
            instance_id
            for instance_id, mapped_class
            in normalized_target_classes.items()
            if mapped_class == class_id
        )
        for class_id in range(1, NUM_CLASSES)
    }
    target_counts = {
        class_id: len(target_ids_by_class[class_id])
        for class_id in range(1, NUM_CLASSES)
    }
    records: dict[int, list[dict[str, object]]] = {
        class_id: []
        for class_id in range(1, NUM_CLASSES)
    }

    for predicted_id in retained_predicted_ids:
        class_id = int(
            normalized_predicted_classes[predicted_id]
        )
        target_ious: dict[int, float] = {}

        for target_id in target_ids_by_class[class_id]:
            intersection = intersections.get(
                (predicted_id, target_id),
                0,
            )
            if intersection <= 0:
                continue
            union = (
                int(predicted_areas[predicted_id])
                + int(target_areas[target_id])
                - intersection
            )
            if union <= 0:
                raise RuntimeError(
                    "Mask union must be positive for an overlapping pair."
                )
            target_ious[target_id] = float(
                intersection / union
            )

        records[class_id].append(
            {
                "image_index": image_index,
                "instance_id": predicted_id,
                "confidence": normalized_scores[predicted_id],
                "target_ious": target_ious,
            }
        )

    return records, target_counts


def _interpolated_average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
    recall_points: int = MASK_AP_RECALL_POINTS,
) -> float:
    """Integrate the monotonic precision envelope over 101 recall samples."""
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)

    if recall.ndim != 1 or precision.ndim != 1:
        raise ValueError("recall and precision must be one-dimensional.")
    if recall.shape != precision.shape:
        raise ValueError(
            "recall and precision must have identical shapes."
        )
    if (
        isinstance(recall_points, bool)
        or not isinstance(recall_points, (int, np.integer))
    ):
        raise TypeError("recall_points must be an integer.")
    recall_points = int(recall_points)
    if recall_points < 2:
        raise ValueError("recall_points must be at least two.")
    if recall.size:
        if not np.all(np.isfinite(recall)):
            raise ValueError("recall contains NaN or infinity.")
        if not np.all(np.isfinite(precision)):
            raise ValueError("precision contains NaN or infinity.")
        if np.any(recall < 0.0) or np.any(recall > 1.0):
            raise ValueError("recall must be in [0,1].")
        if np.any(precision < 0.0) or np.any(precision > 1.0):
            raise ValueError("precision must be in [0,1].")
        if recall.size > 1 and np.any(recall[1:] < recall[:-1]):
            raise ValueError("recall must be non-decreasing.")

    modified_recall = np.concatenate(
        ([0.0], recall, [1.0])
    )
    modified_precision = np.concatenate(
        ([1.0], precision, [0.0])
    )
    modified_precision = np.maximum.accumulate(
        modified_precision[::-1]
    )[::-1]

    recall_grid = np.linspace(
        0.0,
        1.0,
        recall_points,
        dtype=np.float64,
    )
    precision_grid = np.interp(
        recall_grid,
        modified_recall,
        modified_precision,
    )

    if hasattr(np, "trapezoid"):
        average_precision = np.trapezoid(
            precision_grid,
            recall_grid,
        )
    else:
        average_precision = np.trapz(
            precision_grid,
            recall_grid,
        )

    return float(
        np.clip(average_precision, 0.0, 1.0)
    )


def mask_average_precision_report(
    prediction_records: dict[int, list[dict[str, object]]],
    target_counts: dict[int, int],
) -> dict[str, object]:
    """Calculate confidence-ranked same-class mask mAP50 and mAP50-95."""
    thresholds = np.asarray(
        MASK_AP_IOU_THRESHOLDS,
        dtype=np.float64,
    )
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError(
            "MASK_AP_IOU_THRESHOLDS must be a non-empty sequence."
        )
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Mask AP IoU thresholds must be finite.")
    if np.any(thresholds <= 0.0) or np.any(thresholds > 1.0):
        raise ValueError("Mask AP IoU thresholds must be in (0,1].")
    if thresholds.size > 1 and np.any(
        thresholds[1:] <= thresholds[:-1]
    ):
        raise ValueError(
            "Mask AP IoU thresholds must be strictly increasing."
        )

    threshold_keys = [
        f"{float(threshold):.2f}"
        for threshold in thresholds
    ]
    if "0.50" not in threshold_keys:
        raise ValueError(
            "MASK_AP_IOU_THRESHOLDS must contain 0.50."
        )

    expected_class_ids = set(range(1, NUM_CLASSES))
    if set(prediction_records) != expected_class_ids:
        raise ValueError(
            "prediction_records must contain every foreground class ID."
        )
    if set(target_counts) != expected_class_ids:
        raise ValueError(
            "target_counts must contain every foreground class ID."
        )

    per_class: dict[str, dict[str, object]] = {}
    class_ap_values: dict[int, dict[str, float]] = {}

    for class_id in range(1, NUM_CLASSES):
        ground_truth_count = int(target_counts[class_id])
        if ground_truth_count < 0:
            raise ValueError(
                f"Class {class_id} has negative target count."
            )

        raw_records = prediction_records[class_id]
        if not isinstance(raw_records, list):
            raise TypeError(
                f"prediction_records[{class_id}] must be a list."
            )

        records = sorted(
            raw_records,
            key=lambda record: (
                -float(record["confidence"]),
                int(record["image_index"]),
                int(record["instance_id"]),
            ),
        )

        ap_by_iou: dict[str, float | None] = {}
        if ground_truth_count > 0:
            class_ap_values[class_id] = {}

        for threshold, threshold_key in zip(
            thresholds.tolist(),
            threshold_keys,
        ):
            if ground_truth_count == 0:
                ap_by_iou[threshold_key] = None
                continue

            true_positive = np.zeros(
                len(records),
                dtype=np.float64,
            )
            false_positive = np.ones(
                len(records),
                dtype=np.float64,
            )
            matched_targets: set[tuple[int, int]] = set()

            for position, record in enumerate(records):
                image_index = int(record["image_index"])
                confidence = float(record["confidence"])
                target_ious = record["target_ious"]

                if image_index < 0:
                    raise ValueError(
                        "AP record contains a negative image index."
                    )
                if (
                    not np.isfinite(confidence)
                    or not 0.0 <= confidence <= 1.0
                ):
                    raise ValueError(
                        "AP record confidence must be in [0,1]."
                    )
                if not isinstance(target_ious, dict):
                    raise TypeError(
                        "AP record target_ious must be a dictionary."
                    )

                best_target_id: int | None = None
                best_iou = -1.0
                for raw_target_id, raw_iou in target_ious.items():
                    target_id = int(raw_target_id)
                    iou = float(raw_iou)
                    if (
                        target_id <= 0
                        or not np.isfinite(iou)
                        or not 0.0 <= iou <= 1.0
                    ):
                        raise ValueError(
                            "AP record contains an invalid target ID or IoU."
                        )
                    target_key = (image_index, target_id)
                    if target_key in matched_targets or iou < threshold:
                        continue
                    if (
                        iou > best_iou
                        or (
                            np.isclose(iou, best_iou)
                            and (
                                best_target_id is None
                                or target_id < best_target_id
                            )
                        )
                    ):
                        best_iou = iou
                        best_target_id = target_id

                if best_target_id is not None:
                    matched_targets.add(
                        (image_index, best_target_id)
                    )
                    true_positive[position] = 1.0
                    false_positive[position] = 0.0

            if len(records) == 0:
                average_precision = 0.0
            else:
                cumulative_tp = np.cumsum(true_positive)
                cumulative_fp = np.cumsum(false_positive)
                recall = cumulative_tp / ground_truth_count
                precision = np.divide(
                    cumulative_tp,
                    cumulative_tp + cumulative_fp,
                    out=np.zeros_like(cumulative_tp),
                    where=(cumulative_tp + cumulative_fp) > 0.0,
                )
                average_precision = _interpolated_average_precision(
                    recall,
                    precision,
                    recall_points=MASK_AP_RECALL_POINTS,
                )

            ap_by_iou[threshold_key] = float(
                average_precision
            )
            class_ap_values[class_id][threshold_key] = float(
                average_precision
            )

        if ground_truth_count > 0:
            class_map50 = class_ap_values[class_id]["0.50"]
            class_map50_95 = float(
                np.mean(
                    list(class_ap_values[class_id].values())
                )
            )
            class_map75 = class_ap_values[class_id].get("0.75")
        else:
            class_map50 = None
            class_map50_95 = None
            class_map75 = None

        per_class[str(CLASS_NAMES[class_id])] = {
            "class_id": class_id,
            "ground_truth_instances": ground_truth_count,
            "predicted_instances": len(records),
            "ap50": class_map50,
            "ap75": class_map75,
            "ap50_95": class_map50_95,
            "ap_by_iou": ap_by_iou,
        }

    active_class_ids = sorted(class_ap_values)
    mean_ap_by_iou: dict[str, float] = {}
    for threshold_key in threshold_keys:
        if active_class_ids:
            value = float(
                np.mean(
                    [
                        class_ap_values[class_id][threshold_key]
                        for class_id in active_class_ids
                    ]
                )
            )
        else:
            value = 0.0
        mean_ap_by_iou[threshold_key] = value

    map50 = mean_ap_by_iou["0.50"]
    map75 = mean_ap_by_iou.get("0.75", 0.0)
    map50_95 = float(
        np.mean(list(mean_ap_by_iou.values()))
    )

    return {
        "map50": float(map50),
        "map75": float(map75),
        "map50_95": float(map50_95),
        "map_by_iou": mean_ap_by_iou,
        "iou_thresholds": [
            float(value)
            for value in thresholds.tolist()
        ],
        "recall_interpolation_points": int(
            MASK_AP_RECALL_POINTS
        ),
        "active_foreground_classes": len(active_class_ids),
        "max_detections_per_image": int(
            MASK_AP_MAX_DETECTIONS_PER_IMAGE
        ),
        "confidence_source": (
            "mean predicted probability of the decoded instance's class "
            "over all pixels in that instance"
        ),
        "matching_method": (
            "confidence-ranked greedy one-to-one same-image same-class "
            "mask matching at each IoU threshold"
        ),
        "integration_method": (
            "monotonic precision envelope integrated on 101 recall points"
        ),
        "per_class": per_class,
    }


def metric_from_counts(
    counts: dict[str, int],
) -> dict[str, float | int]:
    """Calculate instance precision, recall and F1 from validated counts."""
    required = ("tp", "fp", "fn")
    missing = [key for key in required if key not in counts]

    if missing:
        raise KeyError(
            f"Missing instance-count keys: {missing}."
        )

    validated: dict[str, int] = {}

    for key in required:
        value = counts[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
        ):
            raise TypeError(
                f"{key} must be an integer, received "
                f"{type(value).__name__}."
            )

        value = int(value)

        if value < 0:
            raise ValueError(
                f"{key} cannot be negative: {value}."
            )

        validated[key] = value

    tp = validated["tp"]
    fp = validated["fp"]
    fn = validated["fn"]

    predicted_instances = tp + fp
    target_instances = tp + fn

    precision = (
        tp / predicted_instances
        if predicted_instances > 0
        else 0.0
    )

    recall = (
        tp / target_instances
        if target_instances > 0
        else 0.0
    )

    # Direct count formula is more stable than calculating F1
    # from already-divided precision and recall.
    f1_denominator = 2 * tp + fp + fn

    f1 = (
        2.0 * tp / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_instances": predicted_instances,
        "target_instances": target_instances,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def semantic_report_from_confusion(
    confusion: np.ndarray,
) -> dict[str, object]:
    """Create a complete semantic report from a validated confusion matrix."""
    matrix = np.asarray(confusion)

    expected_shape = (
        NUM_CLASSES,
        NUM_CLASSES,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Confusion matrix must have shape {expected_shape}; "
            f"received {matrix.shape}."
        )

    if not np.issubdtype(
        matrix.dtype,
        np.integer,
    ):
        raise TypeError(
            "Confusion matrix must contain integer counts; "
            f"received dtype {matrix.dtype}."
        )

    if matrix.size and np.any(matrix < 0):
        raise ValueError(
            "Confusion matrix contains negative counts."
        )

    if (
        np.issubdtype(matrix.dtype, np.unsignedinteger)
        and matrix.size
        and int(matrix.max()) > np.iinfo(np.int64).max
    ):
        raise OverflowError(
            "Confusion-matrix value exceeds int64 capacity."
        )

    matrix = matrix.astype(
        np.int64,
        copy=False,
    )

    true_positive = np.diag(
        matrix
    ).astype(np.float64)

    target_pixels = np.sum(
        matrix,
        axis=1,
        dtype=np.int64,
    ).astype(np.float64)

    predicted_pixels = np.sum(
        matrix,
        axis=0,
        dtype=np.int64,
    ).astype(np.float64)

    false_positive = (
        predicted_pixels - true_positive
    )

    false_negative = (
        target_pixels - true_positive
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    dice_denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )

    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive),
        where=precision_denominator > 0,
    )

    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive),
        where=recall_denominator > 0,
    )

    f1_dice = np.divide(
        2.0 * true_positive,
        dice_denominator,
        out=np.zeros_like(true_positive),
        where=dice_denominator > 0,
    )

    valid_union = union > 0
    valid_foreground = valid_union.copy()
    valid_foreground[0] = False

    total_pixels = int(matrix.sum())
    total_correct = int(true_positive.sum())

    overall_pixel_accuracy = (
        total_correct / total_pixels
        if total_pixels > 0
        else 0.0
    )

    mean_iou_including_background = (
        float(np.mean(iou[valid_union]))
        if np.any(valid_union)
        else 0.0
    )

    foreground_mean_iou = (
        float(np.mean(iou[valid_foreground]))
        if np.any(valid_foreground)
        else 0.0
    )

    foreground_mean_precision = (
        float(np.mean(precision[valid_foreground]))
        if np.any(valid_foreground)
        else 0.0
    )

    foreground_mean_recall = (
        float(np.mean(recall[valid_foreground]))
        if np.any(valid_foreground)
        else 0.0
    )

    foreground_mean_f1_dice = (
        float(np.mean(f1_dice[valid_foreground]))
        if np.any(valid_foreground)
        else 0.0
    )

    frequency_weighted_iou = (
        float(
            np.sum(
                (target_pixels / target_pixels.sum())
                * iou
            )
        )
        if target_pixels.sum() > 0
        else 0.0
    )

    foreground_tp = int(
        true_positive[1:].sum()
    )

    foreground_fp = int(
        false_positive[1:].sum()
    )

    foreground_fn = int(
        false_negative[1:].sum()
    )

    foreground_micro = metric_from_counts(
        {
            "tp": foreground_tp,
            "fp": foreground_fp,
            "fn": foreground_fn,
        }
    )

    per_class: dict[str, dict[str, object]] = {}

    observed_class_names: set[str] = set()

    for class_id in range(NUM_CLASSES):
        if class_id not in CLASS_NAMES:
            raise KeyError(
                f"CLASS_NAMES has no entry for class {class_id}."
            )

        class_name = str(
            CLASS_NAMES[class_id]
        )

        if class_name in observed_class_names:
            raise ValueError(
                f"Duplicate CLASS_NAMES value: {class_name!r}."
            )

        observed_class_names.add(class_name)

        per_class[class_name] = {
            "class_id": class_id,
            "support_pixels": int(
                target_pixels[class_id]
            ),
            "predicted_pixels": int(
                predicted_pixels[class_id]
            ),
            "tp_pixels": int(
                true_positive[class_id]
            ),
            "fp_pixels": int(
                false_positive[class_id]
            ),
            "fn_pixels": int(
                false_negative[class_id]
            ),
            "present_in_target": bool(
                target_pixels[class_id] > 0
            ),
            "present_in_prediction": bool(
                predicted_pixels[class_id] > 0
            ),
            "iou": float(
                iou[class_id]
            ),
            "precision": float(
                precision[class_id]
            ),
            "recall": float(
                recall[class_id]
            ),
            "f1_dice": float(
                f1_dice[class_id]
            ),
        }

    return {
        "total_pixels": total_pixels,
        "correct_pixels": total_correct,
        "overall_pixel_accuracy": float(
            overall_pixel_accuracy
        ),
        "mean_iou_including_background": (
            mean_iou_including_background
        ),
        "foreground_mean_iou": (
            foreground_mean_iou
        ),
        "foreground_mean_precision": (
            foreground_mean_precision
        ),
        "foreground_mean_recall": (
            foreground_mean_recall
        ),
        "foreground_mean_f1_dice": (
            foreground_mean_f1_dice
        ),
        "frequency_weighted_iou": (
            frequency_weighted_iou
        ),
        "foreground_micro_class_aware": (
            foreground_micro
        ),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": (
            matrix.tolist()
        ),
    }


def selected_indices(
    total: int,
    maximum: int,
) -> np.ndarray:
    """Select exact, deterministic midpoints from equal dataset intervals."""
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, np.integer))
    ):
        raise TypeError(
            "total must be an integer."
        )

    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, np.integer))
    ):
        raise TypeError(
            "maximum must be an integer."
        )

    total = int(total)
    maximum = int(maximum)

    if total < 0:
        raise ValueError(
            f"total cannot be negative: {total}."
        )

    if total == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    # maximum <= 0 means evaluate the complete split.
    if maximum <= 0 or maximum >= total:
        return np.arange(
            total,
            dtype=np.int64,
        )

    # Select the midpoint of each equally sized interval.
    # This avoids bias towards the first or last dataset items.
    indices = np.floor(
        (
            np.arange(
                maximum,
                dtype=np.float64,
            )
            + 0.5
        )
        * total
        / maximum
    ).astype(np.int64)

    indices = np.clip(
        indices,
        0,
        total - 1,
    )

    if (
        len(indices) != maximum
        or len(np.unique(indices)) != maximum
    ):
        raise RuntimeError(
            "Could not produce the requested number of "
            "unique evaluation indices."
        )

    return indices


def evaluate_model_arrays(
    model: Model,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    show_progress: bool = False,
    inference_batch_size: int | None = None,
    iou_threshold: float = INSTANCE_EVALUATION_IOU,
) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    """Evaluate semantic and instance segmentation on explicit indices."""
    required_array_names = {
        "images",
        "semantic",
        "instance",
    }

    missing_arrays = sorted(
        required_array_names - set(arrays)
    )

    if missing_arrays:
        raise KeyError(
            f"Evaluation arrays are missing: {missing_arrays}."
        )

    total_images = len(
        arrays["images"]
    )

    if (
        len(arrays["semantic"]) != total_images
        or len(arrays["instance"]) != total_images
    ):
        raise ValueError(
            "Image, semantic and instance arrays have "
            "different lengths."
        )

    selected = np.asarray(indices)

    if selected.ndim != 1:
        raise ValueError(
            "indices must be a one-dimensional array; "
            f"received shape {selected.shape}."
        )

    if not np.issubdtype(
        selected.dtype,
        np.integer,
    ):
        raise TypeError(
            f"indices must use an integer dtype, "
            f"received {selected.dtype}."
        )

    selected = selected.astype(
        np.int64,
        copy=False,
    )

    if len(selected) == 0:
        raise ValueError(
            "No evaluation indices were provided."
        )

    if (
        int(selected.min()) < 0
        or int(selected.max()) >= total_images
    ):
        raise IndexError(
            "Evaluation indices are outside the dataset range "
            f"0..{total_images - 1}."
        )

    if len(np.unique(selected)) != len(selected):
        raise ValueError(
            "Evaluation indices contain duplicates, which would "
            "count some images more than once."
        )

    if inference_batch_size is None:
        inference_batch_size = BATCH_SIZE

    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(
            inference_batch_size,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "inference_batch_size must be an integer."
        )

    inference_batch_size = int(
        inference_batch_size
    )

    if inference_batch_size <= 0:
        raise ValueError(
            "inference_batch_size must be positive."
        )

    if (
        not np.isfinite(iou_threshold)
        or not 0.0 < float(iou_threshold) <= 1.0
    ):
        raise ValueError(
            "iou_threshold must be in (0,1]."
        )

    iou_threshold = float(
        iou_threshold
    )

    validate_model_output_shapes(model)

    aggregate = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for class_id in range(1, NUM_CLASSES)
    }

    mask_ap_prediction_records: dict[
        int,
        list[dict[str, object]],
    ] = {
        class_id: []
        for class_id in range(1, NUM_CLASSES)
    }
    mask_ap_target_counts = {
        class_id: 0
        for class_id in range(1, NUM_CLASSES)
    }

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    evaluation_start = time.perf_counter()
    next_progress_report = 50

    for batch_start in range(
        0,
        len(selected),
        inference_batch_size,
    ):
        batch_indices = selected[
            batch_start:
            batch_start + inference_batch_size
        ]

        current_batch_size = len(
            batch_indices
        )

        image_batch = np.empty(
            (
                current_batch_size,
                IMG_SIZE,
                IMG_SIZE,
                3,
            ),
            dtype=np.float32,
        )

        for position, array_index in enumerate(
            batch_indices
        ):
            normalized = np.asarray(
                normalize_image(
                    arrays["images"][array_index]
                ),
                dtype=np.float32,
            )

            expected_image_shape = (
                IMG_SIZE,
                IMG_SIZE,
                3,
            )

            if normalized.shape != expected_image_shape:
                raise ValueError(
                    f"Image {int(array_index)} has shape "
                    f"{normalized.shape}; expected "
                    f"{expected_image_shape}."
                )

            if not np.all(
                np.isfinite(normalized)
            ):
                raise FloatingPointError(
                    f"Image {int(array_index)} contains "
                    "NaN or infinity after normalization."
                )

            image_batch[position] = normalized

        outputs = unpack_model_outputs(
            model(
                image_batch,
                training=False,
            )
        )

        observed_output_batch = outputs[
            "semantic"
        ].shape[0]

        if observed_output_batch != current_batch_size:
            raise ValueError(
                "Model output batch size differs from input: "
                f"{observed_output_batch} versus "
                f"{current_batch_size}."
            )

        for position, array_index in enumerate(
            batch_indices
        ):
            if USE_TTA_AT_EVALUATION:
                # Imported here rather than at module scope: tta.py consumes
                # this file, so an eager import would be circular.
                import tta_v7 as _tta

                (
                    semantic_probabilities,
                    inner_distance,
                    boundary_probability,
                ) = _tta.predict_with_tta(
                    model,
                    image_batch[position],
                    unpack_model_outputs,
                    transforms=TTA_TRANSFORMS,
                    batch_size=int(TTA_BATCH_SIZE),
                )
            else:
                (
                    semantic_probabilities,
                    inner_distance,
                    boundary_probability,
                ) = output_probabilities(
                    outputs,
                    position,
                )

            predicted_semantic = np.argmax(
                semantic_probabilities,
                axis=-1,
            ).astype(np.int32)

            target_semantic_raw = np.asarray(
                arrays["semantic"][array_index]
            )

            if (
                target_semantic_raw.ndim == 3
                and target_semantic_raw.shape[-1] == 1
            ):
                target_semantic_raw = (
                    target_semantic_raw[..., 0]
                )

            if not np.issubdtype(
                target_semantic_raw.dtype,
                np.integer,
            ):
                raise TypeError(
                    f"Semantic target {int(array_index)} must "
                    f"use an integer dtype; received "
                    f"{target_semantic_raw.dtype}."
                )

            target_semantic = (
                target_semantic_raw.astype(
                    np.int32,
                    copy=False,
                )
            )

            if (
                target_semantic.shape
                != predicted_semantic.shape
            ):
                raise ValueError(
                    f"Semantic target/prediction shape mismatch "
                    f"for image {int(array_index)}: "
                    f"{target_semantic.shape} versus "
                    f"{predicted_semantic.shape}."
                )

            if (
                int(target_semantic.min()) < 0
                or int(target_semantic.max())
                >= NUM_CLASSES
            ):
                invalid = np.unique(
                    target_semantic[
                        (target_semantic < 0)
                        | (
                            target_semantic
                            >= NUM_CLASSES
                        )
                    ]
                )

                raise ValueError(
                    f"Image {int(array_index)} contains invalid "
                    f"semantic IDs: {invalid[:16].tolist()}."
                )

            # Training targets pass through sanitize_semantic_and_instances,
            # which drops sub-minimum-area fragments. Scoring against the raw
            # arrays would count objects the model was never trained to
            # produce as guaranteed false negatives.
            (
                target_semantic,
                target_instance_map,
            ) = sanitize_semantic_and_instances(
                target_semantic,
                np.asarray(
                    arrays["instance"][array_index]
                ),
            )
            target_semantic = target_semantic.astype(
                np.int32,
                copy=False,
            )

            encoded_semantic_pairs = (
                target_semantic.reshape(-1).astype(
                    np.int64
                )
                * NUM_CLASSES
                + predicted_semantic.reshape(-1).astype(
                    np.int64
                )
            )

            confusion += np.bincount(
                encoded_semantic_pairs,
                minlength=(
                    NUM_CLASSES
                    * NUM_CLASSES
                ),
            ).reshape(
                NUM_CLASSES,
                NUM_CLASSES,
            )

            (
                predicted_instance_map,
                predicted_classes,
                _,
                predicted_center_scores,
            ) = decode_instances(
                semantic_probabilities,
                inner_distance,
                boundary_probability,
            )

            target_classes = (
                ground_truth_instance_classes(
                    target_instance_map,
                    target_semantic,
                )
            )

            predicted_summaries = summarise_instances(
                predicted_instance_map,
                predicted_classes,
                semantic_probabilities,
                predicted_center_scores,
            )
            predicted_scores = {
                int(summary["instance_id"]): float(
                    summary["confidence"]
                )
                for summary in predicted_summaries
            }
            (
                image_ap_records,
                image_ap_target_counts,
            ) = mask_ap_records_for_image(
                predicted_instance_map,
                predicted_classes,
                predicted_scores,
                target_instance_map,
                target_classes,
                image_index=int(array_index),
            )

            for class_id in range(1, NUM_CLASSES):
                mask_ap_prediction_records[class_id].extend(
                    image_ap_records[class_id]
                )
                mask_ap_target_counts[class_id] += int(
                    image_ap_target_counts[class_id]
                )

            image_counts = (
                match_instances_at_iou(
                    predicted_instance_map,
                    predicted_classes,
                    target_instance_map,
                    target_classes,
                    iou_threshold,
                )
            )

            for class_id in range(
                1,
                NUM_CLASSES,
            ):
                for key in (
                    "tp",
                    "fp",
                    "fn",
                ):
                    aggregate[class_id][key] += int(
                        image_counts[
                            class_id
                        ][key]
                    )

        completed = min(
            batch_start + current_batch_size,
            len(selected),
        )

        if show_progress and (
            completed >= next_progress_report
            or completed == len(selected)
        ):
            elapsed = max(
                time.perf_counter()
                - evaluation_start,
                1e-9,
            )

            rate = completed / elapsed

            print(
                f"Evaluated {completed}/{len(selected)} images "
                f"({rate:.2f} images/second)"
            )

            while (
                next_progress_report
                <= completed
            ):
                next_progress_report += 50

    elapsed_seconds = max(
        time.perf_counter()
        - evaluation_start,
        0.0,
    )

    semantic_report = (
        semantic_report_from_confusion(
            confusion
        )
    )

    semantic_report[
        "images_evaluated"
    ] = int(len(selected))

    semantic_report[
        "elapsed_seconds"
    ] = float(elapsed_seconds)

    per_class: dict[
        str,
        dict[str, float | int],
    ] = {}

    for class_id in range(
        1,
        NUM_CLASSES,
    ):
        class_metrics = metric_from_counts(
            aggregate[class_id]
        )

        per_class[
            str(CLASS_NAMES[class_id])
        ] = {
            "class_id": class_id,
            **class_metrics,
        }

    total_counts = {
        key: sum(
            aggregate[class_id][key]
            for class_id in aggregate
        )
        for key in (
            "tp",
            "fp",
            "fn",
        )
    }

    overall = metric_from_counts(
        total_counts
    )

    active_class_metrics = [
        metrics
        for metrics in per_class.values()
        if (
            int(metrics["tp"])
            + int(metrics["fp"])
            + int(metrics["fn"])
        ) > 0
    ]

    if active_class_metrics:
        macro_precision = float(
            np.mean(
                [
                    float(metrics["precision"])
                    for metrics in active_class_metrics
                ]
            )
        )

        macro_recall = float(
            np.mean(
                [
                    float(metrics["recall"])
                    for metrics in active_class_metrics
                ]
            )
        )

        macro_f1 = float(
            np.mean(
                [
                    float(metrics["f1"])
                    for metrics in active_class_metrics
                ]
            )
        )
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    images_per_second = (
        len(selected) / elapsed_seconds
        if elapsed_seconds > 0.0
        else 0.0
    )

    mask_ap_report = mask_average_precision_report(
        mask_ap_prediction_records,
        mask_ap_target_counts,
    )

    instance_report: dict[str, object] = {
        "images_evaluated": int(
            len(selected)
        ),
        "inference_batch_size": (
            inference_batch_size
        ),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "images_per_second": float(
            images_per_second
        ),
        "iou_threshold": (
            iou_threshold
        ),
        "matching_method": (
            "maximum-cardinality one-to-one same-class matching"
        ),
        # Convenient headline aliases plus the complete AP breakdown.
        "mask_map50": float(
            mask_ap_report["map50"]
        ),
        "mask_map75": float(
            mask_ap_report["map75"]
        ),
        "mask_map50_95": float(
            mask_ap_report["map50_95"]
        ),
        "mask_average_precision": mask_ap_report,
        "overall": overall,
        "macro_foreground": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "active_classes": len(
                active_class_metrics
            ),
        },
        "per_class": per_class,
        "decoder_thresholds": {
            "semantic_confidence": (
                SEMANTIC_CONFIDENCE_THRESHOLD
            ),
            "center_confidence": (
                CENTER_CONFIDENCE_THRESHOLD
            ),
            "center_nms_radius": (
                CENTER_NMS_RADIUS
            ),
            "maximum_assignment_distance": (
                MAX_CENTER_ASSIGNMENT_DISTANCE
            ),
            "boundary_confidence": (
                BOUNDARY_CONFIDENCE_THRESHOLD
            ),
            "minimum_instance_area": (
                MIN_INSTANCE_AREA
            ),
            "minimum_component_area_fraction": (
                MIN_COMPONENT_AREA_FRACTION
            ),
            "fallback_instance_score": (
                FALLBACK_INSTANCE_SCORE
            ),
            "minimum_boundary_core_area": (
                MIN_BOUNDARY_CORE_AREA
            ),
            "maximum_centers_per_class": (
                MAX_CENTERS_PER_CLASS
            ),
        },
    }

    return (
        semantic_report,
        instance_report,
    )
def save_evaluation_reports(
    split: str,
    model_path: Path,
    semantic_report: dict[str, object],
    instance_report: dict[str, object],
    output_dir: Path,
) -> None:
    """Safely save complete semantic and instance evaluation reports."""

    if not isinstance(split, str):
        raise TypeError("split must be a string.")

    split = split.strip().lower()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", split):
        raise ValueError(
            "split may contain only letters, numbers, "
            "underscores and hyphens."
        )

    if not isinstance(semantic_report, dict):
        raise TypeError(
            "semantic_report must be a dictionary."
        )

    if not isinstance(instance_report, dict):
        raise TypeError(
            "instance_report must be a dictionary."
        )

    if not semantic_report:
        raise ValueError(
            "semantic_report cannot be empty."
        )

    if not instance_report:
        raise ValueError(
            "instance_report cannot be empty."
        )

    model_path = Path(model_path).expanduser()
    output_dir = Path(output_dir).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Evaluated model does not exist: {model_path}"
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    metadata = {
        "split": split,
        "model": str(model_path),
        "generated_at_utc": generated_at,
        "evaluation_report_schema_version": 3,
    }

    semantic_payload = {
        **dict(semantic_report),
        **metadata,
        "report_type": "semantic_segmentation",
    }

    instance_payload = {
        **dict(instance_report),
        **metadata,
        "report_type": "instance_segmentation",
    }

    def json_converter(value):
        """Convert common NumPy and Path values for strict JSON."""
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            converted = float(value)

            if not np.isfinite(converted):
                raise ValueError(
                    "Evaluation report contains NaN or infinity."
                )

            return converted

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "numpy"):
            return json_converter(value.numpy())

        raise TypeError(
            f"Cannot serialize value of type "
            f"{type(value).__name__}."
        )

    # Serialize both reports before modifying either output file.
    semantic_json = json.dumps(
        semantic_payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=json_converter,
    ) + "\n"

    instance_json = json.dumps(
        instance_payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=json_converter,
    ) + "\n"

    semantic_path = (
        output_dir
        / f"{split}_semantic_evaluation.json"
    )

    instance_path = (
        output_dir
        / f"{split}_instance_evaluation.json"
    )

    def atomic_write(
        destination: Path,
        contents: str,
    ) -> None:
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                temporary_file.write(contents)
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    atomic_write(
        semantic_path,
        semantic_json,
    )

    atomic_write(
        instance_path,
        instance_json,
    )

    print(
        f"\n{split.upper()} SEMANTIC RESULTS"
    )
    print(semantic_json.rstrip())

    print(
        f"\n{split.upper()} INSTANCE RESULTS"
    )
    print(instance_json.rstrip())

    print("\nSemantic report:", semantic_path)
    print("Instance report:", instance_path)

class InstanceF1Checkpoint(tf.keras.callbacks.Callback):
    """
    Periodically evaluate instance segmentation and atomically save the
    checkpoint with the best validation instance F1.

    Place this callback after SwapEMAWeights(swap_on_epoch=True).
    """

    def __init__(
        self,
        val_arrays: dict[str, np.ndarray],
        output_path: Path,
        history_path: Path,
        every_n_epochs: int = INSTANCE_CHECKPOINT_EVERY_N_EPOCHS,
        maximum_images: int = INSTANCE_CHECKPOINT_MAX_IMAGES,
        inference_batch_size: int = 1,
        minimum_improvement: float = 1e-5,
        resume_history: bool = True,
        early_stopping_patience_evaluations: int | None = None,
        early_stopping_start_epoch: int = 0,
    ):
        super().__init__()

        required_arrays = {
            "images",
            "semantic",
            "instance",
        }

        missing = sorted(
            required_arrays - set(val_arrays)
        )

        if missing:
            raise KeyError(
                f"Validation arrays are missing: {missing}."
            )

        validation_length = len(
            val_arrays["images"]
        )

        if validation_length <= 0:
            raise ValueError(
                "Validation dataset is empty."
            )

        if (
            len(val_arrays["semantic"])
            != validation_length
            or len(val_arrays["instance"])
            != validation_length
        ):
            raise ValueError(
                "Validation image, semantic and instance "
                "arrays have different lengths."
            )

        self.every_n_epochs = int(
            every_n_epochs
        )

        self.maximum_images = int(
            maximum_images
        )

        self.inference_batch_size = int(
            inference_batch_size
        )

        self.minimum_improvement = float(
            minimum_improvement
        )

        self.early_stopping_patience_evaluations = (
            None
            if early_stopping_patience_evaluations is None
            else int(early_stopping_patience_evaluations)
        )

        self.early_stopping_start_epoch = int(
            early_stopping_start_epoch
        )

        if self.every_n_epochs <= 0:
            raise ValueError(
                "every_n_epochs must be positive."
            )

        if self.inference_batch_size <= 0:
            raise ValueError(
                "inference_batch_size must be positive."
            )

        if (
            not np.isfinite(
                self.minimum_improvement
            )
            or self.minimum_improvement < 0.0
        ):
            raise ValueError(
                "minimum_improvement must be finite "
                "and non-negative."
            )

        if (
            self.early_stopping_patience_evaluations is not None
            and self.early_stopping_patience_evaluations <= 0
        ):
            raise ValueError(
                "early_stopping_patience_evaluations must be positive or None."
            )

        if self.early_stopping_start_epoch < 0:
            raise ValueError(
                "early_stopping_start_epoch must be non-negative."
            )

        self.val_arrays = val_arrays
        self.output_path = Path(
            output_path
        ).expanduser()

        self.history_path = Path(
            history_path
        ).expanduser()

        if self.output_path.suffix.lower() != ".keras":
            raise ValueError(
                "Instance checkpoint output_path must end "
                "with '.keras'."
            )

        if self.history_path.suffix.lower() != ".json":
            raise ValueError(
                "Instance checkpoint history_path must end "
                "with '.json'."
            )

        if (
            self.output_path.resolve()
            == self.history_path.resolve()
        ):
            raise ValueError(
                "output_path and history_path must differ."
            )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.indices = selected_indices(
            validation_length,
            self.maximum_images,
        )

        if len(self.indices) == 0:
            raise ValueError(
                "No validation images were selected."
            )

        self.selection_metric = str(INSTANCE_SELECTION_METRIC)
        allowed_selection_metrics = {"mask_map50_95", "mask_map50", "f1"}
        if self.selection_metric not in allowed_selection_metrics:
            raise ValueError(
                "INSTANCE_SELECTION_METRIC must be one of "
                f"{sorted(allowed_selection_metrics)}; received "
                f"{self.selection_metric!r}."
            )

        self.best_selection_score = -1.0
        self.best_epoch = 0
        self.evaluations_without_improvement = 0
        self.history: list[
            dict[str, object]
        ] = []

        if (
            resume_history
            and self.history_path.is_file()
        ):
            self._load_existing_history()

    def _load_existing_history(self) -> None:
        """Restore the previous best score after interrupted training."""
        try:
            loaded = json.loads(
                self.history_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(
                "Could not read existing instance-checkpoint "
                f"history: {self.history_path}"
            ) from error

        if not isinstance(loaded, list):
            raise ValueError(
                "Instance-checkpoint history must contain "
                "a JSON list."
            )

        validated_history: list[
            dict[str, object]
        ] = []

        for position, record in enumerate(
            loaded
        ):
            if not isinstance(record, dict):
                raise ValueError(
                    f"History record {position} is not "
                    "a dictionary."
                )

            if "epoch" not in record or "f1" not in record:
                raise ValueError(
                    f"History record {position} is missing "
                    "'epoch' or 'f1'."
                )

            epoch = int(
                record["epoch"]
            )

            f1 = float(
                record["f1"]
            )

            if epoch <= 0:
                raise ValueError(
                    f"History record {position} contains "
                    f"invalid epoch {epoch}."
                )

            if (
                not np.isfinite(f1)
                or not 0.0 <= f1 <= 1.0
            ):
                raise ValueError(
                    f"History record {position} contains "
                    f"invalid F1 {f1}."
                )

            validated_history.append(
                dict(record)
            )

        self.history = validated_history

        # Only resume the previous best when its checkpoint exists.
        if (
            self.history
            and self.output_path.is_file()
        ):
            # A best score recorded under a different selection metric is not
            # comparable, so it is discarded rather than silently used as a
            # threshold the new metric can never clear.
            comparable = [
                record
                for record in self.history
                if str(
                    record.get("selection_metric", "f1")
                )
                == self.selection_metric
            ]

            if comparable:
                best_record = max(
                    comparable,
                    key=lambda record: (
                        float(
                            record.get(
                                "selection_score",
                                record["f1"],
                            )
                        ),
                        -int(record["epoch"]),
                    ),
                )

                self.best_selection_score = float(
                    best_record.get(
                        "selection_score",
                        best_record["f1"],
                    )
                )

                self.best_epoch = int(
                    best_record["epoch"]
                )

                self.evaluations_without_improvement = sum(
                    1
                    for record in self.history
                    if int(record["epoch"]) > self.best_epoch
                    and int(record["epoch"])
                    >= self.early_stopping_start_epoch
                )
            else:
                print(
                    "Instance-checkpoint history was written under a "
                    "different selection metric; starting its best score "
                    "from scratch."
                )

    def _write_history_atomically(self) -> None:
        """Write valid JSON without risking a partially written history."""
        import os
        import tempfile

        contents = json.dumps(
            self.history,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.history_path.stem}.",
                suffix=".tmp",
                dir=self.history_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                temporary_file.write(
                    contents
                )

                temporary_file.flush()

                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                self.history_path,
            )

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def _save_model_atomically(self) -> None:
        """Save to a temporary .keras file before replacing the checkpoint."""
        import os
        import uuid

        temporary_path = (
            self.output_path.parent
            / (
                f".{self.output_path.stem}."
                f"{uuid.uuid4().hex}.tmp.keras"
            )
        )

        try:
            self.model.save(
                temporary_path,
                overwrite=True,
            )

            if not temporary_path.is_file():
                raise OSError(
                    "Keras did not create the temporary "
                    f"checkpoint: {temporary_path}"
                )

            os.replace(
                temporary_path,
                self.output_path,
            )

        finally:
            if temporary_path.is_file():
                temporary_path.unlink()

    def on_train_begin(
        self,
        logs=None,
    ) -> None:
        del logs

        print(
            "Instance-F1 checkpoint evaluation:",
            f"{len(self.indices)} validation image(s),",
            f"every {self.every_n_epochs} epoch(s),",
            f"inference batch size {self.inference_batch_size}.",
        )

        if self.early_stopping_patience_evaluations is not None:
            print(
                "Instance-F1 early stopping:",
                f"patience={self.early_stopping_patience_evaluations} "
                "evaluations,",
                f"active from epoch {self.early_stopping_start_epoch}.",
            )

        if self.best_epoch > 0:
            print(
                "Resumed instance-F1 checkpoint history:",
                f"best {self.selection_metric}="
                f"{self.best_selection_score:.4f} "
                f"at epoch {self.best_epoch}.",
            )

    #: Every key this callback contributes to `logs`. It must publish all of
    #: them on every epoch, evaluating or not. Keras's CSVLogger fixes its
    #: column set from the first epoch it sees and it re-derives that set after
    #: every resume (`on_train_begin` sets `self.keys = None`), so a resume
    #: landing on a non-evaluating epoch would otherwise drop these columns
    #: from the log file permanently, and rows written before and after would
    #: have different widths under one header. A positional reader then
    #: silently misassigns every column after the gap.
    LOG_KEYS = (
        "val_instance_precision",
        "val_instance_recall",
        "val_instance_f1",
        "val_instance_mask_map50",
        "val_instance_mask_map50_95",
        "val_instance_checkpoint_semantic_miou",
        "val_instance_selection_score",
    )

    def on_epoch_end(
        self,
        epoch: int,
        logs=None,
    ) -> None:
        epoch_number = int(epoch) + 1

        if (
            epoch_number != 1
            and epoch_number
            % self.every_n_epochs
            != 0
        ):
            # Placeholders, not stale values: this epoch was not evaluated and
            # must not look as though it was. Readers drop non-finite entries.
            if logs is not None:
                for key in self.LOG_KEYS:
                    logs.setdefault(key, float("nan"))
            return

        if self.model is None:
            raise RuntimeError(
                "InstanceF1Checkpoint has no attached model."
            )

        logs = (
            logs
            if logs is not None
            else {}
        )

        (
            semantic_report,
            instance_report,
        ) = evaluate_model_arrays(
            model=self.model,
            arrays=self.val_arrays,
            indices=self.indices,
            show_progress=False,
            inference_batch_size=(
                self.inference_batch_size
            ),
            iou_threshold=(
                INSTANCE_EVALUATION_IOU
            ),
        )

        overall = instance_report.get(
            "overall"
        )

        if not isinstance(overall, dict):
            raise ValueError(
                "Instance evaluation report has no valid "
                "'overall' section."
            )

        required_metrics = {
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        }

        missing_metrics = sorted(
            required_metrics - set(overall)
        )

        if missing_metrics:
            raise KeyError(
                "Overall instance report is missing: "
                f"{missing_metrics}."
            )

        tp = int(overall["tp"])
        fp = int(overall["fp"])
        fn = int(overall["fn"])

        calculated_metrics = metric_from_counts(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        precision = float(
            calculated_metrics["precision"]
        )

        recall = float(
            calculated_metrics["recall"]
        )

        f1 = float(
            calculated_metrics["f1"]
        )

        reported_f1 = float(
            overall["f1"]
        )

        if not np.isclose(
            f1,
            reported_f1,
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError(
                "Reported instance F1 disagrees with "
                f"TP/FP/FN: reported={reported_f1}, "
                f"calculated={f1}."
            )

        semantic_miou = float(
            semantic_report[
                "foreground_mean_iou"
            ]
        )

        mask_map50 = float(
            instance_report["mask_map50"]
        )
        mask_map50_95 = float(
            instance_report["mask_map50_95"]
        )

        for name, value in {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "semantic_miou": semantic_miou,
            "mask_map50": mask_map50,
            "mask_map50_95": mask_map50_95,
        }.items():
            if (
                not np.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"Invalid validation {name}: {value}."
                )

        logs[
            "val_instance_precision"
        ] = precision

        logs[
            "val_instance_recall"
        ] = recall

        logs[
            "val_instance_f1"
        ] = f1

        logs[
            "val_instance_mask_map50"
        ] = mask_map50

        logs[
            "val_instance_mask_map50_95"
        ] = mask_map50_95

        logs[
            "val_instance_checkpoint_semantic_miou"
        ] = semantic_miou

        selection_score = float(
            {
                "mask_map50_95": mask_map50_95,
                "mask_map50": mask_map50,
                "f1": f1,
            }[self.selection_metric]
        )

        logs["val_instance_selection_score"] = selection_score

        improved = (
            selection_score
            > self.best_selection_score
            + self.minimum_improvement
        )

        previous_best_selection_score = self.best_selection_score
        previous_best_epoch = self.best_epoch
        previous_evaluations_without_improvement = (
            self.evaluations_without_improvement
        )

        if improved:
            self._save_model_atomically()

            self.best_selection_score = selection_score
            self.best_epoch = epoch_number

        if improved:
            self.evaluations_without_improvement = 0
        elif (
            self.early_stopping_patience_evaluations is not None
            and epoch_number >= self.early_stopping_start_epoch
        ):
            self.evaluations_without_improvement += 1

        stop_requested = bool(
            self.early_stopping_patience_evaluations is not None
            and epoch_number >= self.early_stopping_start_epoch
            and self.evaluations_without_improvement
            >= self.early_stopping_patience_evaluations
        )

        record: dict[str, object] = {
            "epoch": epoch_number,
            "images_evaluated": int(
                len(self.indices)
            ),
            "iou_threshold": float(
                INSTANCE_EVALUATION_IOU
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mask_map50": mask_map50,
            "mask_map50_95": mask_map50_95,
            "semantic_foreground_miou": (
                semantic_miou
            ),
            "checkpoint_saved": bool(
                improved
            ),
            "selection_metric": self.selection_metric,
            "selection_score": selection_score,
            "per_class": {
                name: {
                    key: metrics[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                    if key in metrics
                }
                for name, metrics in instance_report[
                    "per_class"
                ].items()
            },
            "per_class_ap": {
                name: {
                    "ap50": metrics.get("ap50"),
                    "ap50_95": metrics.get("ap50_95"),
                }
                for name, metrics in instance_report[
                    "mask_average_precision"
                ]["per_class"].items()
            },
            "best_selection_score_after_epoch": float(
                self.best_selection_score
            ),
            "best_epoch_after_epoch": int(
                self.best_epoch
            ),
            "evaluations_without_improvement": int(
                self.evaluations_without_improvement
            ),
            "early_stopping_requested": stop_requested,
        }

        if "elapsed_seconds" in instance_report:
            record["elapsed_seconds"] = float(
                instance_report[
                    "elapsed_seconds"
                ]
            )

        self.history.append(
            record
        )

        try:
            self._write_history_atomically()

        except Exception:
            # Restore in-memory best state if history persistence fails.
            if improved:
                self.best_selection_score = (
                    previous_best_selection_score
                )

                self.best_epoch = (
                    previous_best_epoch
                )

            self.evaluations_without_improvement = (
                previous_evaluations_without_improvement
            )

            raise

        print(
            f"Validation instance metrics at epoch "
            f"{epoch_number}: "
            f"TP={tp}, FP={fp}, FN={fn}, "
            f"P={precision:.4f}, "
            f"R={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"mask mAP50={mask_map50:.4f}, "
            f"mask mAP50-95={mask_map50_95:.4f}, "
            f"semantic mIoU={semantic_miou:.4f}"
        )

        if improved:
            print(
                "Saved new best validation-instance checkpoint: "
                f"{self.output_path} (epoch {epoch_number}, "
                f"{self.selection_metric}={selection_score:.4f})"
            )

        else:
            print(
                "Instance checkpoint not replaced. "
                f"Current best: epoch {self.best_epoch}, "
                f"{self.selection_metric}="
                f"{self.best_selection_score:.4f}"
            )

        if stop_requested:
            self.model.stop_training = True
            print(
                f"Stopping: validation {self.selection_metric} did not "
                f"improve for {self.evaluations_without_improvement} "
                "scheduled evaluations."
            )

# =============================================================================
# 12. VISUAL DIAGNOSTICS AND PREDICTION OUTPUT
# =============================================================================
def _as_rgb_uint8(
    image_rgb: np.ndarray,
    *,
    name: str = "image_rgb",
    float_range: str = "auto",
) -> np.ndarray:
    """
    Convert a validated image to contiguous RGB uint8.

    Accepted shapes:
        [H, W]
        [H, W, 1]
        [H, W, 3]

    float_range:
        "auto"  - infer [0,1] or [0,255]
        "unit"  - require floating values in [0,1]
        "byte"  - require floating values in [0,255]
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "name must be a non-empty string."
        )

    if float_range not in {
        "auto",
        "unit",
        "byte",
    }:
        raise ValueError(
            "float_range must be 'auto', 'unit', or 'byte'."
        )

    if np.ma.isMaskedArray(image_rgb):
        mask = np.ma.getmaskarray(image_rgb)

        if np.any(mask):
            raise ValueError(
                f"{name} contains masked pixels."
            )

        image = np.asarray(
            image_rgb.data
        )
    else:
        image = np.asarray(
            image_rgb
        )

    if image.ndim == 2:
        image = np.repeat(
            image[..., None],
            3,
            axis=-1,
        )

    elif (
        image.ndim == 3
        and image.shape[-1] == 1
    ):
        image = np.repeat(
            image,
            3,
            axis=-1,
        )

    if (
        image.ndim != 3
        or image.shape[-1] != 3
    ):
        raise ValueError(
            f"{name} must have shape [H,W], [H,W,1], "
            f"or [H,W,3]; received {image.shape}."
        )

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            f"{name} must not be empty; "
            f"received shape {image.shape}."
        )

    # Boolean masks should be displayed as 0 and 255,
    # rather than nearly black values 0 and 1.
    if image.dtype == np.bool_:
        return np.ascontiguousarray(
            image.astype(np.uint8) * 255
        )

    if np.issubdtype(
        image.dtype,
        np.floating,
    ):
        image = image.astype(
            np.float32,
            copy=False,
        )

        if not np.all(
            np.isfinite(image)
        ):
            invalid_count = int(
                np.count_nonzero(
                    ~np.isfinite(image)
                )
            )

            raise ValueError(
                f"{name} contains {invalid_count:,} "
                "NaN or infinity value(s)."
            )

        minimum = float(
            image.min()
        )

        maximum = float(
            image.max()
        )

        unit_tolerance = 1e-4
        byte_tolerance = 1e-3

        if float_range == "unit":
            if (
                minimum < -unit_tolerance
                or maximum > 1.0 + unit_tolerance
            ):
                raise ValueError(
                    f"{name} must contain floating values "
                    f"in [0,1]; observed "
                    f"[{minimum:.6g}, {maximum:.6g}]."
                )

            image = (
                np.clip(
                    image,
                    0.0,
                    1.0,
                )
                * 255.0
            )

        elif float_range == "byte":
            if (
                minimum < -byte_tolerance
                or maximum > 255.0 + byte_tolerance
            ):
                raise ValueError(
                    f"{name} must contain floating values "
                    f"in [0,255]; observed "
                    f"[{minimum:.6g}, {maximum:.6g}]."
                )

            image = np.clip(
                image,
                0.0,
                255.0,
            )

        else:
            appears_normalized = (
                minimum >= -unit_tolerance
                and maximum
                <= 1.0 + unit_tolerance
            )

            appears_byte_scaled = (
                minimum >= -byte_tolerance
                and maximum
                <= 255.0 + byte_tolerance
            )

            if appears_normalized:
                image = (
                    np.clip(
                        image,
                        0.0,
                        1.0,
                    )
                    * 255.0
                )

            elif appears_byte_scaled:
                image = np.clip(
                    image,
                    0.0,
                    255.0,
                )

            else:
                raise ValueError(
                    f"{name} floating values must be in "
                    f"[0,1] or [0,255]; observed "
                    f"[{minimum:.6g}, {maximum:.6g}]."
                )

    elif np.issubdtype(
        image.dtype,
        np.integer,
    ):
        minimum = int(
            image.min()
        )

        maximum = int(
            image.max()
        )

        if minimum < 0 or maximum > 255:
            raise ValueError(
                f"{name} integer values must be in "
                f"[0,255]; observed "
                f"[{minimum}, {maximum}]."
            )

        image = image.astype(
            np.float32,
            copy=False,
        )

    else:
        raise TypeError(
            f"{name} has unsupported dtype "
            f"{image.dtype}."
        )

    # Round to the nearest integer and avoid silent
    # uint8 wrapping.
    image_uint8 = np.floor(
        np.clip(
            image,
            0.0,
            255.0,
        )
        + 0.5
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        image_uint8
    )
def _as_label_map(
    labels_map: np.ndarray,
    *,
    name: str = "labels_map",
    integer_tolerance: float = 1e-6,
) -> np.ndarray:
    """Return a validated contiguous int32 semantic label map."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "name must be a non-empty string."
        )

    if (
        not np.isfinite(integer_tolerance)
        or integer_tolerance < 0.0
    ):
        raise ValueError(
            "integer_tolerance must be finite and non-negative."
        )

    if (
        isinstance(NUM_CLASSES, bool)
        or not isinstance(
            NUM_CLASSES,
            (int, np.integer),
        )
        or int(NUM_CLASSES) <= 1
    ):
        raise ValueError(
            f"NUM_CLASSES must be an integer greater than one; "
            f"received {NUM_CLASSES!r}."
        )

    number_of_classes = int(
        NUM_CLASSES
    )

    if np.ma.isMaskedArray(labels_map):
        mask = np.ma.getmaskarray(
            labels_map
        )

        if np.any(mask):
            raise ValueError(
                f"{name} contains masked pixels."
            )

        labels = np.asarray(
            labels_map.data
        )
    else:
        labels = np.asarray(
            labels_map
        )

    if (
        labels.ndim == 3
        and labels.shape[-1] == 1
    ):
        labels = labels[..., 0]

    if labels.ndim != 2:
        raise ValueError(
            f"{name} must have shape [H,W] or [H,W,1]; "
            f"received {labels.shape}."
        )

    if labels.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if np.issubdtype(
        labels.dtype,
        np.floating,
    ):
        floating_labels = labels.astype(
            np.float64,
            copy=False,
        )

        finite = np.isfinite(
            floating_labels
        )

        if not np.all(finite):
            invalid_count = int(
                np.count_nonzero(~finite)
            )

            raise ValueError(
                f"{name} contains {invalid_count:,} "
                "NaN or infinity value(s)."
            )

        rounded = np.rint(
            floating_labels
        )

        differences = np.abs(
            floating_labels - rounded
        )

        maximum_difference = float(
            differences.max()
        )

        if maximum_difference > integer_tolerance:
            invalid_positions = np.argwhere(
                differences > integer_tolerance
            )

            first_y, first_x = (
                invalid_positions[0].tolist()
            )

            raise ValueError(
                f"{name} contains non-integer class values. "
                f"Maximum rounding difference="
                f"{maximum_difference:.6g}; first invalid value "
                f"at ({first_y}, {first_x}) is "
                f"{floating_labels[first_y, first_x]:.8g}."
            )

        minimum = float(
            rounded.min()
        )

        maximum = float(
            rounded.max()
        )

        if (
            minimum < 0.0
            or maximum
            >= number_of_classes
        ):
            invalid = np.unique(
                rounded[
                    (rounded < 0.0)
                    | (
                        rounded
                        >= number_of_classes
                    )
                ]
            )

            raise ValueError(
                f"{name} semantic IDs must be in "
                f"0..{number_of_classes - 1}; "
                f"found {invalid[:16].tolist()}."
            )

        validated = rounded.astype(
            np.int32
        )

    elif (
        np.issubdtype(
            labels.dtype,
            np.integer,
        )
        or labels.dtype == np.bool_
    ):
        # Check before int32 conversion to prevent large unsigned
        # integers from silently wrapping.
        minimum = int(
            labels.min()
        )

        maximum = int(
            labels.max()
        )

        if (
            minimum < 0
            or maximum
            >= number_of_classes
        ):
            invalid = np.unique(
                labels[
                    (labels < 0)
                    | (
                        labels
                        >= number_of_classes
                    )
                ]
            )

            raise ValueError(
                f"{name} semantic IDs must be in "
                f"0..{number_of_classes - 1}; "
                f"found {invalid[:16].tolist()}."
            )

        validated = labels.astype(
            np.int32,
            copy=False,
        )

    else:
        raise TypeError(
            f"{name} must contain integer class IDs; "
            f"received dtype {labels.dtype}."
        )

    return np.ascontiguousarray(
        validated
    )


def _as_probability_map(
    probability: np.ndarray,
    *,
    name: str,
    boundary_tolerance: float = 1e-4,
) -> np.ndarray:
    """Return a validated contiguous float32 probability map."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "name must be a non-empty string."
        )

    if (
        not np.isfinite(boundary_tolerance)
        or boundary_tolerance < 0.0
    ):
        raise ValueError(
            "boundary_tolerance must be finite and non-negative."
        )

    if np.ma.isMaskedArray(probability):
        mask = np.ma.getmaskarray(
            probability
        )

        if np.any(mask):
            raise ValueError(
                f"{name} contains masked pixels."
            )

        values = np.asarray(
            probability.data
        )
    else:
        values = np.asarray(
            probability
        )

    if (
        values.ndim == 3
        and values.shape[-1] == 1
    ):
        values = values[..., 0]

    if values.ndim != 2:
        raise ValueError(
            f"{name} must have shape [H,W] or [H,W,1]; "
            f"received {values.shape}."
        )

    if values.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not (
        np.issubdtype(
            values.dtype,
            np.number,
        )
        or values.dtype == np.bool_
    ):
        raise TypeError(
            f"{name} must contain numeric probabilities; "
            f"received dtype {values.dtype}."
        )

    if np.issubdtype(
        values.dtype,
        np.complexfloating,
    ):
        raise TypeError(
            f"{name} cannot contain complex values."
        )

    values_float64 = values.astype(
        np.float64,
        copy=False,
    )

    finite = np.isfinite(
        values_float64
    )

    if not np.all(finite):
        invalid_count = int(
            np.count_nonzero(~finite)
        )

        raise ValueError(
            f"{name} contains {invalid_count:,} "
            "NaN or infinity value(s)."
        )

    minimum = float(
        values_float64.min()
    )

    maximum = float(
        values_float64.max()
    )

    if (
        minimum < -boundary_tolerance
        or maximum
        > 1.0 + boundary_tolerance
    ):
        invalid = values_float64[
            (values_float64 < -boundary_tolerance)
            | (
                values_float64
                > 1.0 + boundary_tolerance
            )
        ]

        raise ValueError(
            f"{name} must contain probabilities in [0,1]; "
            f"observed range [{minimum:.8g}, {maximum:.8g}] "
            f"with {invalid.size:,} out-of-range value(s)."
        )

    validated = np.clip(
        values_float64,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    return np.ascontiguousarray(
        validated
    )

def colourize_semantic(
    labels_map: np.ndarray,
) -> np.ndarray:
    """Map validated semantic IDs to validated RGB class colours."""
    labels = _as_label_map(
        labels_map,
        name="semantic labels",
    )

    if np.ma.isMaskedArray(
        CLASS_COLOURS_RGB
    ):
        if np.any(
            np.ma.getmaskarray(
                CLASS_COLOURS_RGB
            )
        ):
            raise ValueError(
                "CLASS_COLOURS_RGB contains masked values."
            )

        palette = np.asarray(
            CLASS_COLOURS_RGB.data
        )
    else:
        palette = np.asarray(
            CLASS_COLOURS_RGB
        )

    expected_shape = (
        NUM_CLASSES,
        3,
    )

    if palette.shape != expected_shape:
        raise ValueError(
            "CLASS_COLOURS_RGB must have shape "
            f"{expected_shape}; received {palette.shape}."
        )

    if not (
        np.issubdtype(
            palette.dtype,
            np.number,
        )
        or palette.dtype == np.bool_
    ):
        raise TypeError(
            "CLASS_COLOURS_RGB must contain numeric values; "
            f"received dtype {palette.dtype}."
        )

    if np.issubdtype(
        palette.dtype,
        np.complexfloating,
    ):
        raise TypeError(
            "CLASS_COLOURS_RGB cannot contain complex values."
        )

    palette_float = palette.astype(
        np.float64,
        copy=False,
    )

    if not np.all(
        np.isfinite(palette_float)
    ):
        raise ValueError(
            "CLASS_COLOURS_RGB contains NaN or infinity."
        )

    rounded_palette = np.rint(
        palette_float
    )

    if not np.allclose(
        palette_float,
        rounded_palette,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "CLASS_COLOURS_RGB must contain integer colour values."
        )

    minimum = float(
        rounded_palette.min()
    )

    maximum = float(
        rounded_palette.max()
    )

    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            "CLASS_COLOURS_RGB values must be in [0,255]; "
            f"observed [{minimum:.0f}, {maximum:.0f}]."
        )

    palette_uint8 = rounded_palette.astype(
        np.uint8
    )

    return np.ascontiguousarray(
        palette_uint8[labels]
    )


def _fit_panel_title(
    title: str,
    maximum_width: int,
    font_scale: float,
    thickness: int,
) -> tuple[str, float]:
    """Fit a single-line OpenCV title without exceeding the panel width."""
    if not isinstance(title, str):
        raise TypeError(
            "title must be a string."
        )

    if (
        isinstance(maximum_width, bool)
        or not isinstance(
            maximum_width,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "maximum_width must be an integer."
        )

    if (
        isinstance(thickness, bool)
        or not isinstance(
            thickness,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "thickness must be an integer."
        )

    maximum_width = int(
        maximum_width
    )

    thickness = int(
        thickness
    )

    if maximum_width <= 0:
        raise ValueError(
            "maximum_width must be positive."
        )

    if thickness <= 0:
        raise ValueError(
            "thickness must be positive."
        )

    if (
        isinstance(font_scale, bool)
        or not isinstance(
            font_scale,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(font_scale)
        or float(font_scale) <= 0.0
    ):
        raise ValueError(
            "font_scale must be finite and positive."
        )

    font_scale = float(
        font_scale
    )

    normalized_title = " ".join(
        title.split()
    )

    if not normalized_title:
        raise ValueError(
            "title must contain visible text."
        )

    # OpenCV Hershey fonts support ASCII only.
    normalized_title = (
        normalized_title
        .encode(
            "ascii",
            errors="replace",
        )
        .decode("ascii")
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    def text_width(
        text: str,
        scale: float,
    ) -> int:
        return int(
            cv2.getTextSize(
                text,
                font,
                scale,
                thickness,
            )[0][0]
        )

    original_width = text_width(
        normalized_title,
        font_scale,
    )

    if original_width <= maximum_width:
        return (
            normalized_title,
            font_scale,
        )

    minimum_scale = min(
        font_scale,
        0.30,
    )

    fitted_scale = max(
        minimum_scale,
        font_scale
        * maximum_width
        / max(original_width, 1),
    )

    if (
        text_width(
            normalized_title,
            fitted_scale,
        )
        <= maximum_width
    ):
        return (
            normalized_title,
            fitted_scale,
        )

    suffix = "..."

    if (
        text_width(
            suffix,
            fitted_scale,
        )
        > maximum_width
    ):
        for candidate in (
            "..",
            ".",
            "",
        ):
            if (
                text_width(
                    candidate,
                    fitted_scale,
                )
                <= maximum_width
            ):
                return (
                    candidate,
                    fitted_scale,
                )

    low = 0
    high = len(
        normalized_title
    )

    while low < high:
        middle = (
            low + high + 1
        ) // 2

        candidate = (
            normalized_title[:middle]
            .rstrip()
            + suffix
        )

        if (
            text_width(
                candidate,
                fitted_scale,
            )
            <= maximum_width
        ):
            low = middle
        else:
            high = middle - 1

    fitted_title = (
        normalized_title[:low]
        .rstrip()
        + suffix
    )

    return (
        fitted_title,
        fitted_scale,
    )


def add_panel_title(
    image_rgb: np.ndarray,
    title: str,
) -> np.ndarray:
    """Add a fitted title above the image without covering PCB pixels."""
    image = _as_rgb_uint8(
        image_rgb,
        name="panel image",
    )

    height, width = image.shape[:2]

    margin = max(
        2,
        min(
            12,
            width // 64,
        ),
    )

    available_width = max(
        1,
        width - 2 * margin,
    )

    initial_font_scale = max(
        0.40,
        min(
            0.68,
            min(width, height)
            / 820.0,
        ),
    )

    thickness = (
        1
        if width < 900
        else 2
    )

    fitted_title, fitted_scale = (
        _fit_panel_title(
            title=title,
            maximum_width=available_width,
            font_scale=initial_font_scale,
            thickness=thickness,
        )
    )

    (
        _,
        text_height,
    ), baseline = cv2.getTextSize(
        fitted_title,
        cv2.FONT_HERSHEY_SIMPLEX,
        fitted_scale,
        thickness,
    )

    # The header height must depend only on the panel geometry, never on the
    # title. Panels are concatenated side by side, and a long title that gets
    # shrunk to fit would otherwise produce a shorter panel than its
    # neighbour, so the row could not be concatenated at all. _fit_panel_title
    # only ever shrinks, so sizing from the initial scale is also an upper
    # bound on the fitted text.
    (
        _,
        reference_text_height,
    ), reference_baseline = cv2.getTextSize(
        "REFERENCE",
        cv2.FONT_HERSHEY_SIMPLEX,
        initial_font_scale,
        thickness,
    )

    header_height = max(
        30,
        reference_text_height
        + reference_baseline
        + 12,
    )

    panel = np.zeros(
        (
            height + header_height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    panel[
        header_height:
    ] = image

    # Subtle separator between title and source image.
    panel[
        header_height - 1,
        :,
    ] = (
        45,
        45,
        45,
    )

    if fitted_title:
        origin_y = (
            (
                header_height
                - text_height
                - baseline
            )
            // 2
            + text_height
        )

        cv2.putText(
            panel,
            fitted_title,
            (
                margin,
                max(
                    text_height,
                    origin_y,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            fitted_scale,
            (
                255,
                255,
                255,
            ),
            thickness,
            cv2.LINE_AA,
        )

    return np.ascontiguousarray(
        panel
    )


def probability_heatmap(
    probability: np.ndarray,
    gamma: float = 1.0,
    zero_is_black: bool = True,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Render a validated probability map as a deterministic RGB heatmap."""
    if (
        isinstance(gamma, bool)
        or not isinstance(
            gamma,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(gamma)
        or float(gamma) <= 0.0
    ):
        raise ValueError(
            f"gamma must be finite and positive; received {gamma!r}."
        )

    if not isinstance(
        zero_is_black,
        (bool, np.bool_),
    ):
        raise TypeError(
            "zero_is_black must be boolean."
        )

    if (
        isinstance(colormap, bool)
        or not isinstance(
            colormap,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "colormap must be an OpenCV integer colormap ID."
        )

    values = _as_probability_map(
        probability,
        name="probability heatmap",
    )

    display_values = np.power(
        values.astype(
            np.float64,
            copy=False,
        ),
        float(gamma),
    )

    encoded = np.floor(
        np.clip(
            display_values,
            0.0,
            1.0,
        )
        * 255.0
        + 0.5
    ).astype(
        np.uint8
    )

    encoded = np.ascontiguousarray(
        encoded
    )

    try:
        bgr = cv2.applyColorMap(
            encoded,
            int(colormap),
        )

    except cv2.error as error:
        raise ValueError(
            f"OpenCV rejected colormap ID {colormap}."
        ) from error

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    if bool(zero_is_black):
        # Only exact zero probabilities become black.
        rgb[values == 0.0] = (
            0,
            0,
            0,
        )

    return np.ascontiguousarray(
        rgb
    )


def write_rgb_image(
    path: Path,
    image_rgb: np.ndarray,
) -> None:
    """Encode and atomically save an RGB image, including Unicode paths."""
    import os
    import tempfile

    destination = Path(
        path
    ).expanduser()

    extension = (
        destination.suffix.lower()
    )

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if extension not in supported_extensions:
        raise ValueError(
            f"Unsupported or missing image extension "
            f"{extension!r}; choose one of "
            f"{sorted(supported_extensions)}."
        )

    if (
        destination.exists()
        and destination.is_dir()
    ):
        raise IsADirectoryError(
            f"Image destination is a directory: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = _as_rgb_uint8(
        image_rgb,
        name="image to save",
    )

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    encoding_parameters: list[int] = []

    if extension == ".png":
        encoding_parameters = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    elif extension in {
        ".jpg",
        ".jpeg",
    }:
        encoding_parameters = [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ]

    elif extension == ".webp":
        encoding_parameters = [
            cv2.IMWRITE_WEBP_QUALITY,
            95,
        ]

    try:
        success, encoded = cv2.imencode(
            extension,
            bgr,
            encoding_parameters,
        )

    except cv2.error as error:
        raise OSError(
            f"OpenCV could not encode image as "
            f"{extension}: {destination}"
        ) from error

    if (
        not success
        or encoded is None
        or encoded.size == 0
    ):
        raise OSError(
            f"OpenCV produced no encoded image data: "
            f"{destination}"
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}.",
            suffix=extension,
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(
                encoded.tobytes()
            )

            temporary_file.flush()

            os.fsync(
                temporary_file.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.is_file()
        ):
            temporary_path.unlink()


def _validated_instance_inputs(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
) -> tuple[
    np.ndarray,
    dict[int, int],
    np.ndarray,
]:
    """Validate an instance map and its instance-to-class mapping."""
    if (
        isinstance(MAX_INSTANCE_ID, bool)
        or not isinstance(
            MAX_INSTANCE_ID,
            (int, np.integer),
        )
        or int(MAX_INSTANCE_ID) <= 0
    ):
        raise ValueError(
            "MAX_INSTANCE_ID must be a positive integer."
        )

    maximum_allowed_id = int(
        MAX_INSTANCE_ID
    )

    if (
        isinstance(NUM_CLASSES, bool)
        or not isinstance(
            NUM_CLASSES,
            (int, np.integer),
        )
        or int(NUM_CLASSES) <= 1
    ):
        raise ValueError(
            "NUM_CLASSES must be greater than one."
        )

    number_of_classes = int(
        NUM_CLASSES
    )

    if not isinstance(
        class_by_instance,
        dict,
    ):
        raise TypeError(
            "class_by_instance must be a dictionary."
        )

    if np.ma.isMaskedArray(
        instance_map
    ):
        mask = np.ma.getmaskarray(
            instance_map
        )

        if np.any(mask):
            raise ValueError(
                "instance_map contains masked pixels."
            )

        instances = np.asarray(
            instance_map.data
        )
    else:
        instances = np.asarray(
            instance_map
        )

    if (
        instances.ndim == 3
        and instances.shape[-1] == 1
    ):
        instances = instances[..., 0]

    if instances.ndim != 2:
        raise ValueError(
            "instance_map must have shape [H,W] or "
            f"[H,W,1]; received {instances.shape}."
        )

    if instances.size == 0:
        raise ValueError(
            "instance_map must not be empty."
        )

    if not np.issubdtype(
        instances.dtype,
        np.integer,
    ):
        raise TypeError(
            "instance_map must use an integer dtype; "
            f"received {instances.dtype}."
        )

    minimum_id = int(
        instances.min()
    )

    maximum_id = int(
        instances.max()
    )

    if minimum_id < 0:
        raise ValueError(
            f"instance_map contains negative ID {minimum_id}."
        )

    if maximum_id > maximum_allowed_id:
        raise OverflowError(
            f"instance_map contains ID {maximum_id:,}, "
            f"which exceeds MAX_INSTANCE_ID="
            f"{maximum_allowed_id:,}."
        )

    instances = np.ascontiguousarray(
        instances.astype(
            np.int64,
            copy=False,
        )
    )

    mapping: dict[int, int] = {}

    for (
        raw_instance_id,
        raw_class_id,
    ) in class_by_instance.items():
        if (
            isinstance(raw_instance_id, bool)
            or not isinstance(
                raw_instance_id,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "class_by_instance keys must be integer "
                f"instance IDs; received "
                f"{raw_instance_id!r}."
            )

        if (
            isinstance(raw_class_id, bool)
            or not isinstance(
                raw_class_id,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "class_by_instance values must be integer "
                f"class IDs; instance {raw_instance_id!r} "
                f"has value {raw_class_id!r}."
            )

        instance_id = int(
            raw_instance_id
        )

        class_id = int(
            raw_class_id
        )

        if (
            instance_id <= 0
            or instance_id
            > maximum_allowed_id
        ):
            raise ValueError(
                f"Invalid mapped instance ID "
                f"{instance_id}; expected "
                f"1..{maximum_allowed_id}."
            )

        if not 1 <= class_id < number_of_classes:
            raise ValueError(
                f"Instance {instance_id} has invalid "
                f"foreground class {class_id}; expected "
                f"1..{number_of_classes - 1}."
            )

        if class_id not in CLASS_NAMES:
            raise KeyError(
                f"CLASS_NAMES has no entry for "
                f"class {class_id}."
            )

        class_name = CLASS_NAMES[
            class_id
        ]

        if (
            not isinstance(class_name, str)
            or not class_name.strip()
        ):
            raise ValueError(
                f"CLASS_NAMES[{class_id}] must be a "
                "non-empty string."
            )

        if instance_id in mapping:
            raise ValueError(
                f"Duplicate normalized instance ID "
                f"{instance_id} in class_by_instance."
            )

        mapping[
            instance_id
        ] = class_id

    present_ids = np.unique(
        instances
    )

    present_ids = present_ids[
        present_ids > 0
    ].astype(
        np.int64,
        copy=False,
    )

    present_set = {
        int(value)
        for value in present_ids
    }

    mapped_set = set(
        mapping
    )

    missing_mapping = sorted(
        present_set - mapped_set
    )

    stale_mapping = sorted(
        mapped_set - present_set
    )

    if missing_mapping or stale_mapping:
        raise ValueError(
            "instance_map and class_by_instance disagree: "
            f"unmapped IDs={missing_mapping[:16]}, "
            f"absent mapped IDs={stale_mapping[:16]}."
        )

    return (
        instances,
        mapping,
        np.ascontiguousarray(
            present_ids
        ),
    )

def _instance_pixel_scores(
    semantic_scores: np.ndarray,
    flat_instance_ids: np.ndarray,
    foreground_positions: np.ndarray,
    class_lookup: np.ndarray,
    image_shape: tuple[int, int],
    *,
    probability_tolerance: float = 1e-4,
    softmax_sum_tolerance: float = 2e-3,
) -> np.ndarray:
    """Extract validated class-specific semantic confidence for each instance pixel."""
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
    ):
        raise TypeError(
            "image_shape must be a (height, width) tuple."
        )

    height, width = image_shape

    if (
        isinstance(height, bool)
        or not isinstance(height, (int, np.integer))
        or isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
    ):
        raise TypeError(
            "image_shape height and width must be integers."
        )

    height = int(height)
    width = int(width)

    if height <= 0 or width <= 0:
        raise ValueError(
            f"image_shape must be positive; received {image_shape}."
        )

    if (
        not np.isfinite(probability_tolerance)
        or probability_tolerance < 0.0
    ):
        raise ValueError(
            "probability_tolerance must be finite and non-negative."
        )

    if (
        not np.isfinite(softmax_sum_tolerance)
        or softmax_sum_tolerance < 0.0
    ):
        raise ValueError(
            "softmax_sum_tolerance must be finite and non-negative."
        )

    expected_pixel_count = height * width

    instance_ids = np.asarray(
        flat_instance_ids
    )

    if (
        instance_ids.ndim != 1
        or len(instance_ids) != expected_pixel_count
    ):
        raise ValueError(
            "flat_instance_ids must have shape "
            f"[{expected_pixel_count}]; received {instance_ids.shape}."
        )

    if not np.issubdtype(
        instance_ids.dtype,
        np.integer,
    ):
        raise TypeError(
            "flat_instance_ids must use an integer dtype."
        )

    if instance_ids.size and int(instance_ids.min()) < 0:
        raise ValueError(
            "flat_instance_ids contains a negative instance ID."
        )

    instance_ids = instance_ids.astype(
        np.int64,
        copy=False,
    )

    positions = np.asarray(
        foreground_positions
    )

    if positions.ndim != 1:
        raise ValueError(
            "foreground_positions must be one-dimensional."
        )

    if not np.issubdtype(
        positions.dtype,
        np.integer,
    ):
        raise TypeError(
            "foreground_positions must use an integer dtype."
        )

    positions = positions.astype(
        np.int64,
        copy=False,
    )

    if positions.size:
        if (
            int(positions[0]) < 0
            or int(positions[-1])
            >= expected_pixel_count
        ):
            raise IndexError(
                "foreground_positions contains an index outside "
                f"0..{expected_pixel_count - 1}."
            )

        # The caller uses np.flatnonzero(), so positions should be
        # strictly increasing and unique.
        if (
            positions.size > 1
            and np.any(
                positions[1:]
                <= positions[:-1]
            )
        ):
            raise ValueError(
                "foreground_positions must be strictly increasing "
                "and contain no duplicates."
            )

    lookup = np.asarray(
        class_lookup
    )

    if lookup.ndim != 1:
        raise ValueError(
            "class_lookup must be one-dimensional."
        )

    if not np.issubdtype(
        lookup.dtype,
        np.integer,
    ):
        raise TypeError(
            "class_lookup must use an integer dtype."
        )

    lookup = lookup.astype(
        np.int32,
        copy=False,
    )

    foreground_ids = instance_ids[
        positions
    ]

    if foreground_ids.size:
        if np.any(foreground_ids <= 0):
            raise ValueError(
                "foreground_positions references background "
                "instance ID zero."
            )

        maximum_foreground_id = int(
            foreground_ids.max()
        )

        if maximum_foreground_id >= len(lookup):
            raise IndexError(
                f"class_lookup has length {len(lookup)}, but "
                f"instance ID {maximum_foreground_id} is required."
            )

        foreground_classes = lookup[
            foreground_ids
        ]

        invalid_classes = (
            (foreground_classes <= 0)
            | (
                foreground_classes
                >= NUM_CLASSES
            )
        )

        if np.any(invalid_classes):
            invalid_ids = np.unique(
                foreground_ids[
                    invalid_classes
                ]
            )

            raise ValueError(
                "class_lookup contains invalid foreground classes "
                f"for instance IDs {invalid_ids[:16].tolist()}."
            )

    else:
        foreground_classes = np.empty(
            0,
            dtype=np.int32,
        )

    if np.ma.isMaskedArray(
        semantic_scores
    ):
        mask = np.ma.getmaskarray(
            semantic_scores
        )

        if np.any(mask):
            raise ValueError(
                "semantic_scores contains masked values."
            )

        scores_raw = np.asarray(
            semantic_scores.data
        )
    else:
        scores_raw = np.asarray(
            semantic_scores
        )

    if not (
        np.issubdtype(
            scores_raw.dtype,
            np.number,
        )
        or scores_raw.dtype == np.bool_
    ):
        raise TypeError(
            "semantic_scores must contain numeric probabilities."
        )

    if np.issubdtype(
        scores_raw.dtype,
        np.complexfloating,
    ):
        raise TypeError(
            "semantic_scores cannot contain complex values."
        )

    scores = scores_raw.astype(
        np.float32,
        copy=False,
    )

    valid_shapes = {
        (height, width),
        (
            height,
            width,
            NUM_CLASSES,
        ),
    }

    if scores.shape not in valid_shapes:
        raise ValueError(
            "semantic_scores must have shape [H,W] or "
            f"[H,W,{NUM_CLASSES}]; received {scores.shape}."
        )

    finite = np.isfinite(
        scores
    )

    if not np.all(finite):
        invalid_count = int(
            np.count_nonzero(~finite)
        )

        raise ValueError(
            f"semantic_scores contains {invalid_count:,} "
            "NaN or infinity value(s)."
        )

    minimum = float(
        scores.min()
    )

    maximum = float(
        scores.max()
    )

    if (
        minimum < -probability_tolerance
        or maximum
        > 1.0 + probability_tolerance
    ):
        raise ValueError(
            "semantic_scores must contain probabilities in [0,1]; "
            f"observed [{minimum:.8g}, {maximum:.8g}]."
        )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    if scores.ndim == 2:
        foreground_scores = (
            scores.reshape(-1)[
                positions
            ]
        )

    else:
        probability_sums = np.sum(
            scores,
            axis=-1,
            dtype=np.float64,
        )

        maximum_sum_error = float(
            np.max(
                np.abs(
                    probability_sums - 1.0
                )
            )
        )

        if (
            maximum_sum_error
            > softmax_sum_tolerance
        ):
            worst_position = np.unravel_index(
                int(
                    np.argmax(
                        np.abs(
                            probability_sums - 1.0
                        )
                    )
                ),
                probability_sums.shape,
            )

            raise ValueError(
                "Semantic class probabilities must sum to one. "
                f"Maximum error={maximum_sum_error:.6g} "
                f"at pixel {worst_position}."
            )

        flattened_scores = scores.reshape(
            -1,
            NUM_CLASSES,
        )

        foreground_scores = (
            flattened_scores[
                positions,
                foreground_classes,
            ]
        )

    return np.ascontiguousarray(
        np.clip(
            foreground_scores,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )
    )


def summarise_instances(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
    semantic_confidence: np.ndarray,
    instance_center_scores: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    """Compute class-specific confidence, geometry and bounds in one image scan.

    When ``instance_center_scores`` is supplied, the reported ``confidence``
    becomes ``sqrt(centre score) * mean semantic probability`` instead of the
    mean semantic probability alone. The semantic term saturates near 1.0 for
    every decoded instance, so on its own it cannot rank detections; the
    centre score is the part that varies. The raw semantic statistics stay
    available under the ``semantic_confidence_*`` keys.
    """
    (
        instances,
        mapping,
        present_ids,
    ) = _validated_instance_inputs(
        instance_map,
        class_by_instance,
    )

    detection_scores: dict[int, float] | None = None
    if instance_center_scores is not None:
        if not isinstance(instance_center_scores, dict):
            raise TypeError(
                "instance_center_scores must be a dictionary."
            )
        detection_scores = {}
        for instance_id in present_ids.tolist():
            instance_id = int(instance_id)
            if instance_id not in instance_center_scores:
                raise ValueError(
                    f"instance_center_scores has no score for present "
                    f"instance ID {instance_id}."
                )
            score = float(instance_center_scores[instance_id])
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Instance {instance_id} has invalid centre score "
                    f"{score}; expected a finite value in [0,1]."
                )
            detection_scores[instance_id] = score

    height, width = instances.shape

    maximum_id = (
        int(present_ids.max())
        if present_ids.size
        else 0
    )

    vector_length = (
        maximum_id + 1
    )

    class_lookup = np.zeros(
        vector_length,
        dtype=np.int32,
    )

    for (
        instance_id,
        class_id,
    ) in mapping.items():
        class_lookup[
            instance_id
        ] = class_id

    flat_ids = instances.reshape(
        -1
    )

    foreground_positions = np.flatnonzero(
        flat_ids
    ).astype(
        np.int64,
        copy=False,
    )

    # This also validates semantic_confidence when no instances exist.
    pixel_scores = _instance_pixel_scores(
        semantic_scores=semantic_confidence,
        flat_instance_ids=flat_ids,
        foreground_positions=foreground_positions,
        class_lookup=class_lookup,
        image_shape=(
            height,
            width,
        ),
    )

    if present_ids.size == 0:
        return []

    foreground_ids = flat_ids[
        foreground_positions
    ]

    ys, xs = np.divmod(
        foreground_positions,
        width,
    )

    area = np.bincount(
        foreground_ids,
        minlength=vector_length,
    ).astype(
        np.int64
    )

    sum_x = np.bincount(
        foreground_ids,
        weights=xs.astype(
            np.float64,
            copy=False,
        ),
        minlength=vector_length,
    )

    sum_y = np.bincount(
        foreground_ids,
        weights=ys.astype(
            np.float64,
            copy=False,
        ),
        minlength=vector_length,
    )

    score_sum = np.bincount(
        foreground_ids,
        weights=pixel_scores.astype(
            np.float64,
            copy=False,
        ),
        minlength=vector_length,
    )

    score_squared_sum = np.bincount(
        foreground_ids,
        weights=np.square(
            pixel_scores.astype(
                np.float64,
                copy=False,
            )
        ),
        minlength=vector_length,
    )

    min_x = np.full(
        vector_length,
        width,
        dtype=np.int64,
    )

    min_y = np.full(
        vector_length,
        height,
        dtype=np.int64,
    )

    max_x = np.full(
        vector_length,
        -1,
        dtype=np.int64,
    )

    max_y = np.full(
        vector_length,
        -1,
        dtype=np.int64,
    )

    score_min = np.full(
        vector_length,
        np.inf,
        dtype=np.float64,
    )

    score_max = np.full(
        vector_length,
        -np.inf,
        dtype=np.float64,
    )

    np.minimum.at(
        min_x,
        foreground_ids,
        xs,
    )

    np.minimum.at(
        min_y,
        foreground_ids,
        ys,
    )

    np.maximum.at(
        max_x,
        foreground_ids,
        xs,
    )

    np.maximum.at(
        max_y,
        foreground_ids,
        ys,
    )

    np.minimum.at(
        score_min,
        foreground_ids,
        pixel_scores,
    )

    np.maximum.at(
        score_max,
        foreground_ids,
        pixel_scores,
    )

    if int(area.sum()) != len(
        foreground_positions
    ):
        raise RuntimeError(
            "Instance-area accumulation does not match "
            "the number of foreground pixels."
        )

    results: list[
        dict[str, object]
    ] = []

    for raw_instance_id in present_ids:
        instance_id = int(
            raw_instance_id
        )

        count = int(
            area[instance_id]
        )

        if count <= 0:
            raise RuntimeError(
                f"Instance {instance_id} unexpectedly "
                "has zero area."
            )

        x0 = int(
            min_x[instance_id]
        )

        y0 = int(
            min_y[instance_id]
        )

        x1 = int(
            max_x[instance_id]
        )

        y1 = int(
            max_y[instance_id]
        )

        if not (
            0 <= x0 <= x1 < width
            and 0 <= y0 <= y1 < height
        ):
            raise RuntimeError(
                f"Instance {instance_id} produced invalid "
                f"bounds {[x0, y0, x1, y1]}."
            )

        box_width = (
            x1 - x0 + 1
        )

        box_height = (
            y1 - y0 + 1
        )

        box_area = (
            box_width * box_height
        )

        class_id = int(
            mapping[instance_id]
        )

        class_name = str(
            CLASS_NAMES[class_id]
        )

        mean_confidence = float(
            score_sum[instance_id]
            / count
        )

        center_score = (
            None
            if detection_scores is None
            else detection_scores[instance_id]
        )

        # The square root keeps a mid-strength peak from erasing a confident
        # semantic response while still ordering detections by centre
        # evidence. A fallback instance carries FALLBACK_INSTANCE_SCORE, so it
        # always ranks below every centre-backed detection.
        detection_confidence = (
            mean_confidence
            if center_score is None
            else float(
                min(
                    1.0,
                    math.sqrt(center_score) * mean_confidence,
                )
            )
        )

        confidence_variance = max(
            0.0,
            float(
                score_squared_sum[
                    instance_id
                ]
                / count
                - mean_confidence**2
            ),
        )

        confidence_standard_deviation = float(
            np.sqrt(
                confidence_variance
            )
        )

        centroid_x = float(
            sum_x[instance_id]
            / count
        )

        centroid_y = float(
            sum_y[instance_id]
            / count
        )

        touches_image_border = bool(
            x0 == 0
            or y0 == 0
            or x1 == width - 1
            or y1 == height - 1
        )

        equivalent_diameter = float(
            np.sqrt(
                4.0
                * count
                / np.pi
            )
        )

        results.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": class_name,

                # Ranking confidence. Equals the mean semantic probability
                # when no centre scores were supplied.
                "confidence": detection_confidence,
                "detection_score": detection_confidence,
                "center_score": center_score,

                "semantic_confidence_mean": (
                    mean_confidence
                ),
                "semantic_confidence_std": (
                    confidence_standard_deviation
                ),
                "semantic_confidence_min": float(
                    score_min[instance_id]
                ),
                "semantic_confidence_max": float(
                    score_max[instance_id]
                ),

                # Backwards-compatible field names.
                "confidence_mean": mean_confidence,
                "confidence_min": float(
                    score_min[instance_id]
                ),
                "confidence_max": float(
                    score_max[instance_id]
                ),

                "area_pixels": count,

                # xyxy maximum coordinates are inclusive.
                "bbox_xyxy": [
                    x0,
                    y0,
                    x1,
                    y1,
                ],

                "bbox_xywh": [
                    x0,
                    y0,
                    box_width,
                    box_height,
                ],

                "bbox_area_pixels": int(
                    box_area
                ),

                "bbox_fill_ratio": float(
                    count / box_area
                ),

                "bbox_aspect_ratio_width_over_height": float(
                    box_width
                    / box_height
                ),

                "centroid_xy": [
                    centroid_x,
                    centroid_y,
                ],

                "equivalent_diameter_pixels": (
                    equivalent_diameter
                ),

                "touches_image_border": (
                    touches_image_border
                ),
            }
        )

    return results


def filter_instances_by_confidence(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
    semantic_confidence: np.ndarray,
    instance_center_scores: dict[int, float] | None,
    minimum_confidence: float,
) -> tuple[np.ndarray, dict[int, int], dict[int, float] | None, int]:
    """Drop instances scoring below ``minimum_confidence`` from a decoded image.

    Mask AP is rank-aware, so a weak instance costs a labelled benchmark almost
    nothing -- it sorts to the bottom and barely moves the curve. Unlabelled
    deployment has no ranking to hide behind: every retained instance is drawn,
    counted and acted on. Measured on 16 real boards, 54% of all detections
    landed in the 0.2-0.4 band and were duplicates of objects already found.

    Filtering happens on the *reported* confidence (sqrt(centre score) x mean
    semantic probability), not the raw centre score, so what is removed is
    exactly what a reader of the overlay would judge weak. The instance map is
    rewritten as well as the dictionaries, so overlays, JSON and counts cannot
    disagree about what survived.
    """
    minimum_confidence = float(minimum_confidence)
    if not np.isfinite(minimum_confidence):
        raise ValueError("minimum_confidence must be finite.")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError(
            f"minimum_confidence must lie in [0,1]; received {minimum_confidence}."
        )
    if minimum_confidence <= 0.0:
        return instance_map, class_by_instance, instance_center_scores, 0

    summaries = summarise_instances(
        instance_map,
        class_by_instance,
        semantic_confidence,
        instance_center_scores,
    )
    removed_ids = {
        int(summary["instance_id"])
        for summary in summaries
        if float(summary["confidence"]) < minimum_confidence
    }
    if not removed_ids:
        return instance_map, class_by_instance, instance_center_scores, 0

    filtered_map = instance_map.copy()
    filtered_map[np.isin(filtered_map, sorted(removed_ids))] = 0
    filtered_classes = {
        instance_id: class_id
        for instance_id, class_id in class_by_instance.items()
        if instance_id not in removed_ids
    }
    filtered_scores = (
        None
        if instance_center_scores is None
        else {
            instance_id: score
            for instance_id, score in instance_center_scores.items()
            if instance_id not in removed_ids
        }
    )
    return filtered_map, filtered_classes, filtered_scores, len(removed_ids)


def _instance_outline_colour(
    instance_id: int,
    class_id: int,
) -> np.ndarray:
    """Return a bright, deterministic and well-separated RGB outline colour."""
    if (
        isinstance(instance_id, bool)
        or not isinstance(
            instance_id,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "instance_id must be an integer."
        )

    if (
        isinstance(class_id, bool)
        or not isinstance(
            class_id,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "class_id must be an integer."
        )

    instance_id = int(
        instance_id
    )

    class_id = int(
        class_id
    )

    if not 1 <= instance_id <= MAX_INSTANCE_ID:
        raise ValueError(
            f"instance_id must be in 1..{MAX_INSTANCE_ID}; "
            f"received {instance_id}."
        )

    if not 1 <= class_id < NUM_CLASSES:
        raise ValueError(
            f"class_id must be in 1..{NUM_CLASSES - 1}; "
            f"received {class_id}."
        )

    # Golden-angle separation prevents neighbouring IDs from
    # receiving visually similar outline colours.
    golden_angle_degrees = 137.50776405003785

    class_hue_offset = (
        360.0
        * (class_id - 1)
        / max(NUM_CLASSES - 1, 1)
    )

    hue_degrees = (
        class_hue_offset
        + instance_id
        * golden_angle_degrees
    ) % 360.0

    # OpenCV uint8 HSV hue range is 0..179.
    hue_opencv = int(
        np.floor(
            hue_degrees / 2.0
            + 0.5
        )
    ) % 180

    hsv = np.asarray(
        [
            [
                [
                    hue_opencv,
                    235,
                    255,
                ]
            ]
        ],
        dtype=np.uint8,
    )

    rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )[0, 0]

    return np.ascontiguousarray(
        rgb.astype(
            np.uint8,
            copy=False,
        )
    )


def make_instance_overlay(
    image_rgb: np.ndarray,
    instance_map: np.ndarray,
    instances: list[dict[str, object]],
    *,
    mask_alpha: float = 0.48,
    contour_thickness: int = 2,
    draw_bounding_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw class masks, unique outlines, bounding boxes and readable labels."""
    if (
        isinstance(mask_alpha, bool)
        or not isinstance(
            mask_alpha,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(mask_alpha)
        or not 0.0 <= float(mask_alpha) <= 1.0
    ):
        raise ValueError(
            "mask_alpha must be finite and in [0,1]."
        )

    if (
        isinstance(contour_thickness, bool)
        or not isinstance(
            contour_thickness,
            (int, np.integer),
        )
        or int(contour_thickness) <= 0
    ):
        raise ValueError(
            "contour_thickness must be a positive integer."
        )

    if not isinstance(
        draw_bounding_boxes,
        (bool, np.bool_),
    ):
        raise TypeError(
            "draw_bounding_boxes must be boolean."
        )

    if not isinstance(
        draw_labels,
        (bool, np.bool_),
    ):
        raise TypeError(
            "draw_labels must be boolean."
        )

    if not isinstance(
        instances,
        (list, tuple),
    ):
        raise TypeError(
            "instances must be a list or tuple of dictionaries."
        )

    mask_alpha = float(
        mask_alpha
    )

    contour_thickness = int(
        contour_thickness
    )

    overlay = _as_rgb_uint8(
        image_rgb,
        name="instance-overlay image",
    ).copy()

    raw_instance_map = np.asarray(
        instance_map
    )

    if (
        raw_instance_map.ndim == 3
        and raw_instance_map.shape[-1] == 1
    ):
        raw_instance_map = (
            raw_instance_map[..., 0]
        )

    if (
        raw_instance_map.ndim != 2
        or raw_instance_map.shape
        != overlay.shape[:2]
    ):
        raise ValueError(
            "instance_map and image spatial shapes differ: "
            f"{raw_instance_map.shape} versus "
            f"{overlay.shape[:2]}."
        )

    # Validate the palette through the authoritative semantic
    # colour function instead of silently casting bad values.
    palette = colourize_semantic(
        np.arange(
            NUM_CLASSES,
            dtype=np.int32,
        )[None, :]
    )[0]

    result_by_id: dict[
        int,
        dict[str, object]
    ] = {}

    class_mapping: dict[
        int,
        int
    ] = {}

    required_result_keys = {
        "instance_id",
        "class_id",
        "confidence",
        "area_pixels",
        "bbox_xyxy",
    }

    for position, raw_result in enumerate(
        instances
    ):
        if not isinstance(
            raw_result,
            dict,
        ):
            raise TypeError(
                f"instances[{position}] must be a dictionary."
            )

        missing = sorted(
            required_result_keys
            - set(raw_result)
        )

        if missing:
            raise KeyError(
                f"instances[{position}] is missing keys: "
                f"{missing}."
            )

        raw_instance_id = raw_result[
            "instance_id"
        ]

        raw_class_id = raw_result[
            "class_id"
        ]

        if (
            isinstance(raw_instance_id, bool)
            or not isinstance(
                raw_instance_id,
                (int, np.integer),
            )
        ):
            raise TypeError(
                f"instances[{position}]['instance_id'] "
                "must be an integer."
            )

        if (
            isinstance(raw_class_id, bool)
            or not isinstance(
                raw_class_id,
                (int, np.integer),
            )
        ):
            raise TypeError(
                f"instances[{position}]['class_id'] "
                "must be an integer."
            )

        instance_id = int(
            raw_instance_id
        )

        class_id = int(
            raw_class_id
        )

        if instance_id in result_by_id:
            raise ValueError(
                f"Duplicate instance result ID {instance_id}."
            )

        if not 1 <= instance_id <= MAX_INSTANCE_ID:
            raise ValueError(
                f"Invalid instance ID {instance_id}."
            )

        if not 1 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"Invalid class ID {class_id} for "
                f"instance {instance_id}."
            )

        expected_class_name = str(
            CLASS_NAMES[class_id]
        )

        if (
            "class_name" in raw_result
            and str(raw_result["class_name"])
            != expected_class_name
        ):
            raise ValueError(
                f"Instance {instance_id} class name "
                f"{raw_result['class_name']!r} disagrees with "
                f"CLASS_NAMES[{class_id}]="
                f"{expected_class_name!r}."
            )

        confidence = float(
            raw_result["confidence"]
        )

        if (
            not np.isfinite(confidence)
            or confidence < -1e-4
            or confidence > 1.0 + 1e-4
        ):
            raise ValueError(
                f"Instance {instance_id} has invalid "
                f"confidence {confidence}."
            )

        area_pixels = raw_result[
            "area_pixels"
        ]

        if (
            isinstance(area_pixels, bool)
            or not isinstance(
                area_pixels,
                (int, np.integer),
            )
            or int(area_pixels) <= 0
        ):
            raise ValueError(
                f"Instance {instance_id} has invalid "
                f"area_pixels={area_pixels!r}."
            )

        bbox = raw_result[
            "bbox_xyxy"
        ]

        if (
            not isinstance(
                bbox,
                (list, tuple, np.ndarray),
            )
            or len(bbox) != 4
        ):
            raise ValueError(
                f"Instance {instance_id} bbox_xyxy must "
                "contain four integer coordinates."
            )

        validated_bbox: list[int] = []

        for coordinate in bbox:
            if (
                isinstance(coordinate, bool)
                or not isinstance(
                    coordinate,
                    (int, np.integer),
                )
            ):
                raise TypeError(
                    f"Instance {instance_id} bbox coordinates "
                    "must be integers."
                )

            validated_bbox.append(
                int(coordinate)
            )

        normalized_result = dict(
            raw_result
        )

        normalized_result.update(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": expected_class_name,
                "confidence": float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                ),
                "area_pixels": int(
                    area_pixels
                ),
                "bbox_xyxy": validated_bbox,
            }
        )

        result_by_id[
            instance_id
        ] = normalized_result

        class_mapping[
            instance_id
        ] = class_id

    (
        instance_ids,
        validated_mapping,
        present_ids,
    ) = _validated_instance_inputs(
        raw_instance_map,
        class_mapping,
    )

    if set(result_by_id) != {
        int(value)
        for value in present_ids
    }:
        raise ValueError(
            "Instance summaries do not exactly match "
            "the IDs in instance_map."
        )

    image_height, image_width = (
        instance_ids.shape
    )

    prepared_instances: list[
        dict[str, object]
    ] = []

    # First pass: validate boxes and blend all masks.
    for raw_instance_id in present_ids:
        instance_id = int(
            raw_instance_id
        )

        result = result_by_id[
            instance_id
        ]

        class_id = validated_mapping[
            instance_id
        ]

        x0, y0, x1, y1 = [
            int(value)
            for value
            in result["bbox_xyxy"]
        ]

        if not (
            0 <= x0 <= x1 < image_width
            and 0 <= y0 <= y1 < image_height
        ):
            raise ValueError(
                f"Instance {instance_id} has out-of-range "
                f"bbox {[x0, y0, x1, y1]}."
            )

        local_ids = instance_ids[
            y0:
            y1 + 1,
            x0:
            x1 + 1,
        ]

        local_mask = (
            local_ids == instance_id
        )

        actual_area = int(
            np.count_nonzero(
                local_mask
            )
        )

        reported_area = int(
            result["area_pixels"]
        )

        if actual_area != reported_area:
            raise ValueError(
                f"Instance {instance_id} bbox contains "
                f"{actual_area} instance pixels, but "
                f"area_pixels reports {reported_area}. "
                "The bbox may not contain the complete instance."
            )

        # The stored bbox should be tight on all four sides.
        if not (
            np.any(local_mask[0, :])
            and np.any(local_mask[-1, :])
            and np.any(local_mask[:, 0])
            and np.any(local_mask[:, -1])
        ):
            raise ValueError(
                f"Instance {instance_id} bbox "
                f"{[x0, y0, x1, y1]} is not tight."
            )

        region = overlay[
            y0:
            y1 + 1,
            x0:
            x1 + 1,
        ]

        fill_colour = palette[
            class_id
        ].astype(
            np.float32
        )

        source_pixels = region[
            local_mask
        ].astype(
            np.float32
        )

        blended_pixels = (
            (
                1.0 - mask_alpha
            )
            * source_pixels
            + mask_alpha
            * fill_colour
        )

        region[
            local_mask
        ] = np.floor(
            np.clip(
                blended_pixels,
                0.0,
                255.0,
            )
            + 0.5
        ).astype(
            np.uint8
        )

        contour_result = cv2.findContours(
            local_mask.astype(
                np.uint8
            ),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = contour_result[
            -2
        ]

        shifted_contours: list[
            np.ndarray
        ] = []

        for contour in contours:
            shifted = contour.astype(
                np.int32,
                copy=True,
            )

            shifted[..., 0] += x0
            shifted[..., 1] += y0

            shifted_contours.append(
                shifted
            )

        prepared_instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": result[
                    "class_name"
                ],
                "confidence": float(
                    result["confidence"]
                ),
                "bbox_xyxy": [
                    x0,
                    y0,
                    x1,
                    y1,
                ],
                "contours": shifted_contours,
                "outline": (
                    _instance_outline_colour(
                        instance_id,
                        class_id,
                    )
                ),
            }
        )

    # Second pass: draw every contour after all mask blending.
    for prepared in prepared_instances:
        outline = np.asarray(
            prepared["outline"],
            dtype=np.uint8,
        )

        outline_tuple = tuple(
            int(value)
            for value in outline
        )

        contours = prepared[
            "contours"
        ]

        if contours:
            cv2.drawContours(
                overlay,
                contours,
                -1,
                outline_tuple,
                contour_thickness,
                lineType=cv2.LINE_AA,
            )

        if bool(
            draw_bounding_boxes
        ):
            x0, y0, x1, y1 = [
                int(value)
                for value
                in prepared["bbox_xyxy"]
            ]

            cv2.rectangle(
                overlay,
                (
                    x0,
                    y0,
                ),
                (
                    x1,
                    y1,
                ),
                outline_tuple,
                1,
                lineType=cv2.LINE_AA,
            )

    if not bool(draw_labels):
        return np.ascontiguousarray(
            overlay
        )

    occupied_label_boxes: list[
        tuple[int, int, int, int]
    ] = []

    def overlap_area(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        left = max(
            first[0],
            second[0],
        )

        top = max(
            first[1],
            second[1],
        )

        right = min(
            first[2],
            second[2],
        )

        bottom = min(
            first[3],
            second[3],
        )

        if right < left or bottom < top:
            return 0

        return (
            right - left + 1
        ) * (
            bottom - top + 1
        )

    # Third pass: labels are drawn last so masks cannot overwrite them.
    for prepared in prepared_instances:
        instance_id = int(
            prepared["instance_id"]
        )

        class_name = str(
            prepared["class_name"]
        )

        confidence = float(
            prepared["confidence"]
        )

        x0, y0, x1, y1 = [
            int(value)
            for value
            in prepared["bbox_xyxy"]
        ]

        outline = np.asarray(
            prepared["outline"],
            dtype=np.uint8,
        )

        outline_tuple = tuple(
            int(value)
            for value in outline
        )

        raw_label = (
            f"#{instance_id} "
            f"{class_name} "
            f"{confidence:.2f}"
        )

        initial_font_scale = max(
            0.36,
            min(
                0.52,
                image_width / 1050.0,
            ),
        )

        text_thickness = 1
        horizontal_padding = 3
        vertical_padding = 2

        maximum_text_width = max(
            1,
            image_width
            - 2 * horizontal_padding
            - 2,
        )

        label, font_scale = (
            _fit_panel_title(
                title=raw_label,
                maximum_width=(
                    maximum_text_width
                ),
                font_scale=(
                    initial_font_scale
                ),
                thickness=(
                    text_thickness
                ),
            )
        )

        if not label:
            continue

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        label_width = min(
            image_width,
            text_width
            + 2 * horizontal_padding,
        )

        label_height = (
            text_height
            + baseline
            + 2 * vertical_padding
        )

        if label_height > image_height:
            continue

        label_x = min(
            max(
                0,
                x0,
            ),
            max(
                0,
                image_width
                - label_width,
            ),
        )

        candidate_tops = [
            y0 - label_height - 2,
            y1 + 2,
            y0 + 2,
            y1 - label_height - 1,
        ]

        candidate_boxes: list[
            tuple[int, int, int, int]
        ] = []

        for candidate_top in candidate_tops:
            top = min(
                max(
                    0,
                    candidate_top,
                ),
                image_height
                - label_height,
            )

            candidate = (
                label_x,
                top,
                label_x
                + label_width
                - 1,
                top
                + label_height
                - 1,
            )

            if candidate not in candidate_boxes:
                candidate_boxes.append(
                    candidate
                )

        non_overlapping = [
            candidate
            for candidate in candidate_boxes
            if all(
                overlap_area(
                    candidate,
                    occupied,
                )
                == 0
                for occupied
                in occupied_label_boxes
            )
        ]

        if non_overlapping:
            selected_box = (
                non_overlapping[0]
            )

        else:
            selected_box = min(
                candidate_boxes,
                key=lambda candidate: sum(
                    overlap_area(
                        candidate,
                        occupied,
                    )
                    for occupied
                    in occupied_label_boxes
                ),
            )

        (
            label_left,
            label_top,
            label_right,
            label_bottom,
        ) = selected_box

        cv2.rectangle(
            overlay,
            (
                label_left,
                label_top,
            ),
            (
                label_right,
                label_bottom,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.rectangle(
            overlay,
            (
                label_left,
                label_top,
            ),
            (
                label_right,
                label_bottom,
            ),
            outline_tuple,
            1,
            lineType=cv2.LINE_AA,
        )

        text_origin = (
            label_left
            + horizontal_padding,
            label_top
            + vertical_padding
            + text_height,
        )

        cv2.putText(
            overlay,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (
                255,
                255,
                255,
            ),
            text_thickness,
            cv2.LINE_AA,
        )

        occupied_label_boxes.append(
            selected_box
        )

    return np.ascontiguousarray(
        overlay
    )
def foreground_preview_indices(
    arrays: dict[str, np.ndarray],
    count: int,
    maximum_candidates: int = 512,
) -> np.ndarray:
    """Select deterministic previews with rare-class and foreground diversity."""
    if not isinstance(arrays, dict):
        raise TypeError(
            "arrays must be a dictionary."
        )

    required = {
        "images",
        "semantic",
    }

    missing = sorted(
        required - set(arrays)
    )

    if missing:
        raise KeyError(
            f"arrays is missing: {missing}."
        )

    total = len(
        arrays["images"]
    )

    if len(arrays["semantic"]) != total:
        raise ValueError(
            "images and semantic arrays have different lengths."
        )

    if (
        "instance" in arrays
        and len(arrays["instance"]) != total
    ):
        raise ValueError(
            "images and instance arrays have different lengths."
        )

    if (
        isinstance(count, bool)
        or not isinstance(
            count,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "count must be an integer."
        )

    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(
            maximum_candidates,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "maximum_candidates must be an integer."
        )

    count = int(
        count
    )

    maximum_candidates = int(
        maximum_candidates
    )

    if count < 0:
        raise ValueError(
            "count must be non-negative."
        )

    if maximum_candidates <= 0:
        raise ValueError(
            "maximum_candidates must be positive."
        )

    if count == 0 or total == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    count = min(
        count,
        total,
    )

    candidate_count = min(
        total,
        max(
            count,
            maximum_candidates,
        ),
    )

    if candidate_count == total:
        candidate_indices = np.arange(
            total,
            dtype=np.int64,
        )

    else:
        stratified_count = (
            candidate_count + 1
        ) // 2

        # Midpoints of equal dataset intervals avoid bias towards
        # the first and last files.
        stratified_indices = np.floor(
            (
                np.arange(
                    stratified_count,
                    dtype=np.float64,
                )
                + 0.5
            )
            * total
            / stratified_count
        ).astype(
            np.int64
        )

        stratified_indices = np.clip(
            stratified_indices,
            0,
            total - 1,
        )

        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    SEED,
                    2027,
                    total,
                    candidate_count,
                ]
            )
        )

        random_order = rng.permutation(
            total
        )

        candidate_list = [
            int(index)
            for index
            in stratified_indices
        ]

        candidate_set = set(
            candidate_list
        )

        for raw_index in random_order:
            if (
                len(candidate_list)
                >= candidate_count
            ):
                break

            index = int(
                raw_index
            )

            if index not in candidate_set:
                candidate_list.append(
                    index
                )

                candidate_set.add(
                    index
                )

        if len(candidate_list) != candidate_count:
            raise RuntimeError(
                "Could not create the requested candidate pool."
            )

        candidate_indices = np.asarray(
            candidate_list,
            dtype=np.int64,
        )

    profiles: dict[
        int,
        dict[str, object],
    ] = {}

    class_image_frequency = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    has_instance_maps = (
        "instance" in arrays
    )

    for raw_index in candidate_indices:
        index = int(
            raw_index
        )

        labels = _as_label_map(
            arrays["semantic"][index],
            name=f"semantic preview candidate {index}",
        )

        class_pixel_counts = np.bincount(
            labels.reshape(-1),
            minlength=NUM_CLASSES,
        ).astype(
            np.int64
        )

        if len(class_pixel_counts) != NUM_CLASSES:
            raise ValueError(
                f"Preview candidate {index} contains an "
                "invalid semantic class."
            )

        present_classes = frozenset(
            int(class_id)
            for class_id
            in np.flatnonzero(
                class_pixel_counts[1:]
            )
            + 1
        )

        foreground_pixels = int(
            class_pixel_counts[1:].sum()
        )

        instance_count = 0

        if has_instance_maps:
            instance_map = np.asarray(
                arrays["instance"][index]
            )

            if (
                instance_map.ndim == 3
                and instance_map.shape[-1] == 1
            ):
                instance_map = instance_map[..., 0]

            if instance_map.shape != labels.shape:
                raise ValueError(
                    f"Candidate {index} semantic/instance "
                    f"shapes differ: {labels.shape} versus "
                    f"{instance_map.shape}."
                )

            if not np.issubdtype(
                instance_map.dtype,
                np.integer,
            ):
                raise TypeError(
                    f"Candidate {index} instance map must "
                    "use an integer dtype."
                )

            if (
                instance_map.size
                and int(instance_map.min()) < 0
            ):
                raise ValueError(
                    f"Candidate {index} contains a negative "
                    "instance ID."
                )

            unique_instances = np.unique(
                instance_map
            )

            instance_count = int(
                np.count_nonzero(
                    unique_instances > 0
                )
            )

        profiles[index] = {
            "present_classes": (
                present_classes
            ),
            "class_pixel_counts": (
                class_pixel_counts
            ),
            "foreground_pixels": (
                foreground_pixels
            ),
            "instance_count": (
                instance_count
            ),
        }

        for class_id in present_classes:
            class_image_frequency[
                class_id
            ] += 1

    remaining = {
        int(index)
        for index
        in candidate_indices
    }

    selected: list[int] = []
    covered_classes: set[int] = set()

    while (
        remaining
        and len(selected) < count
    ):
        def candidate_score(
            index: int,
        ) -> tuple[
            int,
            float,
            int,
            float,
            float,
            int,
            int,
            int,
            int,
        ]:
            profile = profiles[index]

            present_classes = profile[
                "present_classes"
            ]

            class_pixel_counts = profile[
                "class_pixel_counts"
            ]

            new_classes = (
                present_classes
                - covered_classes
            )

            new_class_rarity = sum(
                1.0
                / max(
                    1,
                    int(
                        class_image_frequency[
                            class_id
                        ]
                    ),
                )
                for class_id
                in new_classes
            )

            new_class_visibility = sum(
                (
                    1.0
                    / max(
                        1,
                        int(
                            class_image_frequency[
                                class_id
                            ]
                        ),
                    )
                )
                * float(
                    np.log1p(
                        class_pixel_counts[
                            class_id
                        ]
                    )
                )
                for class_id
                in new_classes
            )

            all_class_rarity = sum(
                1.0
                / max(
                    1,
                    int(
                        class_image_frequency[
                            class_id
                        ]
                    ),
                )
                for class_id
                in present_classes
            )

            return (
                int(bool(new_classes)),
                new_class_rarity,
                len(new_classes),
                new_class_visibility,
                all_class_rarity,
                len(present_classes),
                int(
                    profile[
                        "instance_count"
                    ]
                ),
                int(
                    profile[
                        "foreground_pixels"
                    ]
                ),
                -index,
            )

        best_index = max(
            remaining,
            key=candidate_score,
        )

        selected.append(
            best_index
        )

        covered_classes.update(
            profiles[
                best_index
            ]["present_classes"]
        )

        remaining.remove(
            best_index
        )

    if len(selected) != count:
        raise RuntimeError(
            f"Selected {len(selected)} preview images; "
            f"expected {count}."
        )

    return np.ascontiguousarray(
        np.asarray(
            selected,
            dtype=np.int64,
        )
    )


def _offset_target_rgb(
    offset_target: np.ndarray,
    *,
    valid_threshold: float = 0.5,
    minimum_valid_brightness: int = 48,
) -> np.ndarray:
    """Visualize offset direction, magnitude and valid-pixel support."""
    if (
        not np.isfinite(valid_threshold)
        or not 0.0 < valid_threshold < 1.0
    ):
        raise ValueError(
            "valid_threshold must be in (0,1)."
        )

    if (
        isinstance(minimum_valid_brightness, bool)
        or not isinstance(
            minimum_valid_brightness,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "minimum_valid_brightness must be an integer."
        )

    minimum_valid_brightness = int(
        minimum_valid_brightness
    )

    if not 0 <= minimum_valid_brightness <= 255:
        raise ValueError(
            "minimum_valid_brightness must be in [0,255]."
        )

    if np.ma.isMaskedArray(
        offset_target
    ):
        mask = np.ma.getmaskarray(
            offset_target
        )

        if np.any(mask):
            raise ValueError(
                "offset_target contains masked values."
            )

        offset_raw = np.asarray(
            offset_target.data
        )
    else:
        offset_raw = np.asarray(
            offset_target
        )

    if (
        offset_raw.ndim != 3
        or offset_raw.shape[-1] != 3
    ):
        raise ValueError(
            "offset_target must have shape [H,W,3] "
            f"(dx, dy, valid); received {offset_raw.shape}."
        )

    if not np.issubdtype(
        offset_raw.dtype,
        np.number,
    ):
        raise TypeError(
            "offset_target must contain numeric values."
        )

    if np.issubdtype(
        offset_raw.dtype,
        np.complexfloating,
    ):
        raise TypeError(
            "offset_target cannot contain complex values."
        )

    offset = offset_raw.astype(
        np.float32,
        copy=False,
    )

    if not np.all(
        np.isfinite(offset)
    ):
        raise ValueError(
            "offset_target contains NaN or infinity."
        )

    height, width = offset.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "offset_target must not be empty."
        )

    dx_normalized = offset[..., 0]
    dy_normalized = offset[..., 1]
    valid_values = offset[..., 2]

    probability_tolerance = 1e-4
    offset_tolerance = 1e-4

    if (
        float(valid_values.min())
        < -probability_tolerance
        or float(valid_values.max())
        > 1.0 + probability_tolerance
    ):
        raise ValueError(
            "Offset valid-mask values must be in [0,1]."
        )

    distance_to_binary = np.minimum(
        np.abs(valid_values),
        np.abs(valid_values - 1.0),
    )

    if (
        float(distance_to_binary.max())
        > probability_tolerance
    ):
        raise ValueError(
            "Offset valid-mask values must be binary zero/one."
        )

    valid = (
        valid_values
        >= valid_threshold
    )

    if np.any(valid):
        valid_dx = dx_normalized[
            valid
        ]

        valid_dy = dy_normalized[
            valid
        ]

        if (
            float(valid_dx.min())
            < -1.0 - offset_tolerance
            or float(valid_dx.max())
            > 1.0 + offset_tolerance
            or float(valid_dy.min())
            < -1.0 - offset_tolerance
            or float(valid_dy.max())
            > 1.0 + offset_tolerance
        ):
            raise ValueError(
                "Normalized valid offsets must be in [-1,1]."
            )

    invalid = ~valid

    if np.any(invalid):
        maximum_invalid_offset = float(
            max(
                np.max(
                    np.abs(
                        dx_normalized[
                            invalid
                        ]
                    )
                ),
                np.max(
                    np.abs(
                        dy_normalized[
                            invalid
                        ]
                    )
                ),
            )
        )

        if maximum_invalid_offset > offset_tolerance:
            raise ValueError(
                "Invalid offset pixels must contain zero dx/dy; "
                f"maximum observed magnitude component is "
                f"{maximum_invalid_offset:.6g}."
            )

    dx_pixels = (
        np.clip(
            dx_normalized,
            -1.0,
            1.0,
        )
        * width
    )

    dy_pixels = (
        np.clip(
            dy_normalized,
            -1.0,
            1.0,
        )
        * height
    )

    angle = np.mod(
        np.arctan2(
            dy_pixels,
            dx_pixels,
        ),
        2.0 * np.pi,
    )

    magnitude = np.hypot(
        dx_pixels,
        dy_pixels,
    )

    maximum_possible_magnitude = max(
        float(
            np.hypot(
                width,
                height,
            )
        ),
        1.0,
    )

    normalized_magnitude = np.clip(
        magnitude
        / maximum_possible_magnitude,
        0.0,
        1.0,
    )

    hue = (
        np.floor(
            angle
            * (
                180.0
                / (
                    2.0
                    * np.pi
                )
            )
            + 0.5
        )
        .astype(np.uint16)
        % 180
    ).astype(
        np.uint8
    )

    magnitude_visible = (
        magnitude > 1e-8
    )

    brightness = (
        minimum_valid_brightness
        + (
            255
            - minimum_valid_brightness
        )
        * np.sqrt(
            normalized_magnitude
        )
    )

    hsv = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    hsv[..., 0] = np.where(
        valid,
        hue,
        0,
    ).astype(
        np.uint8
    )

    hsv[..., 1] = np.where(
        valid
        & magnitude_visible,
        255,
        0,
    ).astype(
        np.uint8
    )

    hsv[..., 2] = np.where(
        valid,
        np.floor(
            np.clip(
                brightness,
                0.0,
                255.0,
            )
            + 0.5
        ),
        0,
    ).astype(
        np.uint8
    )

    rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return np.ascontiguousarray(
        rgb
    )


def save_dataset_preview(
    output_dir: Path = MODEL_OUTPUT_DIR,
    array_dir: Path = ARRAY_DIR,
) -> None:
    """Verify preprocessing and representative modes from every V5 category."""
    validate_augmentation_configuration()

    arrays = load_split_arrays(
        "train",
        array_dir=array_dir,
    )

    required_arrays = {
        "images",
        "semantic",
        "instance",
    }

    missing = sorted(
        required_arrays - set(arrays)
    )

    if missing:
        raise KeyError(
            f"Training arrays are missing: {missing}."
        )

    indices = foreground_preview_indices(
        arrays,
        count=1,
    )

    if len(indices) == 0:
        raise ValueError(
            "The training split is empty; "
            "no preview can be created."
        )

    index = int(
        indices[0]
    )

    original_image = np.asarray(
        normalize_image(
            arrays["images"][index]
        ),
        dtype=np.float32,
    )

    original_semantic = _as_label_map(
        arrays["semantic"][index],
        name=f"training semantic map {index}",
    ).copy()

    original_instance = np.asarray(
        arrays["instance"][index]
    )

    if (
        original_instance.ndim == 3
        and original_instance.shape[-1] == 1
    ):
        original_instance = (
            original_instance[..., 0]
        )

    if (
        original_instance.shape
        != original_semantic.shape
    ):
        raise ValueError(
            "Original semantic and instance shapes differ: "
            f"{original_semantic.shape} versus "
            f"{original_instance.shape}."
        )

    if not np.issubdtype(
        original_instance.dtype,
        np.integer,
    ):
        raise TypeError(
            "Original instance map must use an integer dtype."
        )

    if (
        original_instance.size
        and int(original_instance.min()) < 0
    ):
        raise ValueError(
            "Original instance map contains a negative ID."
        )

    original_instance = (
        original_instance.astype(
            np.int32,
            copy=True,
        )
    )

    expected_image_shape = (
        original_semantic.shape[0],
        original_semantic.shape[1],
        3,
    )

    if (
        original_image.shape
        != expected_image_shape
    ):
        raise ValueError(
            f"Normalized preview image has shape "
            f"{original_image.shape}; expected "
            f"{expected_image_shape}."
        )

    if not np.all(
        np.isfinite(original_image)
    ):
        raise ValueError(
            "Normalized preview image contains NaN or infinity."
        )

    image_minimum = float(
        original_image.min()
    )

    image_maximum = float(
        original_image.max()
    )

    if (
        image_minimum < -1e-4
        or image_maximum > 1.0 + 1e-4
    ):
        raise ValueError(
            "Normalized preview image must be in [0,1]; "
            f"observed [{image_minimum:.6g}, "
            f"{image_maximum:.6g}]."
        )

    # One representative from every V5 augmentation family:
    # original, exact rotation, brightness/contrast, gamma/exposure,
    # zoom/translation, blur/sharpen, noise, colour and combined.
    representative_modes = tuple(
        dict.fromkeys(
            (
                0,
                (
                    EXACT_ROTATION_MODE_COUNT
                    + 1
                )
                // 2,
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 16
                ),
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 24
                ),
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 30
                ),
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 36
                ),
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 41
                ),
                (
                    REALISTIC_VARIANT_MODE_OFFSET
                    + 45
                ),
                (
                    AUGMENTATION_CYCLE_LENGTH
                    - 1
                ),
            )
        )
    )

    invalid_modes = [
        mode
        for mode in representative_modes
        if (
            mode < 0
            or mode
            >= AUGMENTATION_CYCLE_LENGTH
        )
    ]

    if invalid_modes:
        raise ValueError(
            f"Invalid representative preview modes: "
            f"{invalid_modes}."
        )

    rows: list[
        np.ndarray
    ] = []

    for mode in representative_modes:
        if mode == 0:
            sample_image = (
                original_image.copy()
            )

            sample_semantic = (
                original_semantic.copy()
            )

            sample_instance = (
                original_instance.copy()
            )

        else:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        SEED,
                        999,
                        index,
                        mode,
                    ]
                )
            )

            (
                sample_image,
                sample_semantic,
                sample_instance,
            ) = augment_training_sample(
                image=original_image.copy(),
                semantic=(
                    original_semantic.copy()
                ),
                instance=(
                    original_instance.copy()
                ),
                mode=mode,
                rng=rng,
            )

        (
            sample_semantic,
            distance_target,
            boundary_target,
        ) = build_all_targets(
            sample_semantic,
            sample_instance,
        )
        # The preview panels below were written for the centre and
        # offset heads. Kept alive by deriving stand-ins from the one
        # field that now exists, so the preview still renders rather
        # than crashing a run at its first epoch.
        center_target = np.repeat(
            distance_target[..., 0:1], NUM_CLASSES - 1, axis=-1
        )
        offset_target = np.concatenate(
            [
                distance_target[..., 0:1],
                distance_target[..., 0:1],
                distance_target[..., 1:2],
            ],
            axis=-1,
        )

        sample_semantic = _as_label_map(
            sample_semantic,
            name=(
                f"preview semantic target "
                f"for mode {mode}"
            ),
        )

        sample_image_uint8 = (
            _as_rgb_uint8(
                sample_image,
                name=(
                    f"preview image for mode "
                    f"{mode}"
                ),
                float_range="unit",
            )
        )

        full_height, full_width = (
            sample_semantic.shape
        )

        if (
            sample_image_uint8.shape[:2]
            != (
                full_height,
                full_width,
            )
        ):
            raise ValueError(
                f"Mode {mode} image/semantic shapes differ."
            )

        center_target = np.asarray(
            center_target,
            dtype=np.float32,
        )

        expected_center_shape = (
            INSTANCE_HEAD_SIZE,
            INSTANCE_HEAD_SIZE,
            NUM_CLASSES - 1,
        )

        if (
            center_target.shape
            != expected_center_shape
        ):
            raise ValueError(
                f"Mode {mode} centre target has shape "
                f"{center_target.shape}; expected "
                f"{expected_center_shape}."
            )

        if not np.all(
            np.isfinite(center_target)
        ):
            raise ValueError(
                f"Mode {mode} centre target contains "
                "NaN or infinity."
            )

        if (
            float(center_target.min())
            < -1e-4
            or float(center_target.max())
            > 1.0 + 1e-4
        ):
            raise ValueError(
                f"Mode {mode} centre target must be in [0,1]."
            )

        expected_offset_shape = (
            INSTANCE_HEAD_SIZE,
            INSTANCE_HEAD_SIZE,
            3,
        )

        if (
            np.asarray(
                offset_target
            ).shape
            != expected_offset_shape
        ):
            raise ValueError(
                f"Mode {mode} offset target has shape "
                f"{np.asarray(offset_target).shape}; expected "
                f"{expected_offset_shape}."
            )

        boundary_map = _as_probability_map(
            boundary_target,
            name=(
                f"boundary target for mode "
                f"{mode}"
            ),
        )

        if (
            boundary_map.shape
            != (
                full_height,
                full_width,
            )
        ):
            raise ValueError(
                f"Mode {mode} boundary target has shape "
                f"{boundary_map.shape}; expected "
                f"{(full_height, full_width)}."
            )

        center_full_channels = (
            resize_channels(
                center_target,
                full_width,
                full_height,
            )
        )

        center_full = np.max(
            center_full_channels,
            axis=-1,
        )

        offset_full = cv2.resize(
            _offset_target_rgb(
                offset_target
            ),
            (
                full_width,
                full_height,
            ),
            interpolation=(
                cv2.INTER_NEAREST
            ),
        )

        mode_title = (
            f"TRAIN {index} | MODE {mode}: "
            f"{augmentation_mode_name(mode)}"
        )

        panels = [
            add_panel_title(
                sample_image_uint8,
                mode_title,
            ),
            add_panel_title(
                colourize_semantic(
                    sample_semantic
                ),
                "SEMANTIC TARGET",
            ),
            add_panel_title(
                probability_heatmap(
                    center_full
                ),
                "CENTRE TARGET (MAX CLASS)",
            ),
            add_panel_title(
                offset_full,
                (
                    "OFFSET TARGET "
                    "(HUE=DIRECTION, VALUE=MAGNITUDE)"
                ),
            ),
            add_panel_title(
                probability_heatmap(
                    boundary_map
                ),
                "INSTANCE BOUNDARY TARGET",
            ),
        ]

        panel_shapes = {
            panel.shape
            for panel in panels
        }

        if len(panel_shapes) != 1:
            raise RuntimeError(
                f"Mode {mode} preview panel shapes "
                f"disagree: {panel_shapes}."
            )

        rows.append(
            np.concatenate(
                panels,
                axis=1,
            )
        )

    if not rows:
        raise RuntimeError(
            "No dataset-preview rows were created."
        )

    preview = np.concatenate(
        rows,
        axis=0,
    )

    output_path = (
        Path(output_dir)
        / (
            "dataset_and_augmentation_"
            "preview.png"
        )
    )

    write_rgb_image(
        output_path,
        preview,
    )

    selected_classes = sorted(
        int(value)
        for value
        in np.unique(
            original_semantic
        )
        if int(value) > 0
    )

    print(
        "Preview saved:",
        output_path,
        f"(training index {index}, "
        f"foreground classes={selected_classes}, "
        f"modes={list(representative_modes)})",
    )
class _NonFatalPreviewCallback(tf.keras.callbacks.Callback):
    """Run ValidationPreviewCallback without letting it kill training.

    Previews are diagnostics. On a run measured in days, losing the whole run
    because a cosmetic panel could not be assembled is a far worse outcome
    than losing one preview image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._inner = ValidationPreviewCallback(*args, **kwargs)

    def set_model(self, model):
        super().set_model(model)
        self._inner.set_model(model)

    def set_params(self, params):
        super().set_params(params)
        self._inner.set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        try:
            self._inner.on_epoch_end(epoch, logs)
        except Exception as error:
            print(f"WARNING: validation preview skipped this epoch: {error}")


class ValidationPreviewCallback(tf.keras.callbacks.Callback):
    """Save validated, memory-safe validation diagnostics in a 2x4 grid."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        output_dir: Path,
        preview_count: int = PREVIEW_IMAGE_COUNT,
        every_n_epochs: int = PREVIEW_EVERY_N_EPOCHS,
        fail_on_error: bool = True,
    ):
        super().__init__()

        required = {"images", "semantic"}
        missing = required - set(arrays)
        if missing:
            raise KeyError(
                f"Validation arrays are missing required entries: {sorted(missing)}."
            )

        total = len(arrays["images"])
        if len(arrays["semantic"]) != total:
            raise ValueError(
                "Validation image and semantic array lengths differ: "
                f"{total} versus {len(arrays['semantic'])}."
            )
        if total == 0:
            raise ValueError("The validation split is empty.")

        self.arrays = arrays
        self.output_dir = Path(output_dir)
        self.preview_count = int(preview_count)
        self.every_n_epochs = int(every_n_epochs)
        self.fail_on_error = bool(fail_on_error)

        if self.preview_count <= 0:
            raise ValueError("preview_count must be positive.")
        if self.every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive.")

        self.indices = np.asarray(
            foreground_preview_indices(
                arrays,
                min(self.preview_count, total),
            ),
            dtype=np.int64,
        )
        if self.indices.size == 0:
            raise ValueError("No validation preview images were selected.")
        if self.indices.ndim != 1:
            raise ValueError(
                f"Preview indices must be one-dimensional; got {self.indices.shape}."
            )
        if np.any(self.indices < 0) or np.any(self.indices >= total):
            raise IndexError("Preview selection returned an out-of-range index.")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("Preview selection returned duplicate indices.")
        self.indices.setflags(write=False)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def semantic_error_rgb(
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        """
        Black: correct background
        Green: correct foreground class
        Red: missed foreground
        Yellow: false foreground
        Magenta: wrong foreground class
        """
        target = _as_label_map(ground_truth)
        predicted = _as_label_map(prediction)

        if target.shape != predicted.shape:
            raise ValueError(
                "Ground-truth and predicted semantic shapes differ: "
                f"{target.shape} versus {predicted.shape}."
            )

        target_foreground = target > 0
        predicted_foreground = predicted > 0
        visual = np.zeros((*target.shape, 3), dtype=np.uint8)

        visual[target_foreground & (predicted == target)] = (60, 190, 90)
        visual[target_foreground & (~predicted_foreground)] = (235, 55, 55)
        visual[(~target_foreground) & predicted_foreground] = (255, 205, 35)
        visual[
            target_foreground
            & predicted_foreground
            & (predicted != target)
        ] = (220, 65, 220)

        return np.ascontiguousarray(visual)

    @staticmethod
    def fused_center_preview(
        semantic_probabilities: np.ndarray,
        center_probabilities_small: np.ndarray,
    ) -> np.ndarray:
        """Create the same semantic-gated centre evidence used by decoding."""
        semantic = np.asarray(semantic_probabilities, dtype=np.float32)
        center_small = np.asarray(center_probabilities_small, dtype=np.float32)

        if semantic.ndim != 3 or semantic.shape[-1] != NUM_CLASSES:
            raise ValueError(
                "Semantic probabilities must have shape "
                f"[H,W,{NUM_CLASSES}]; received {semantic.shape}."
            )
        if (
            center_small.ndim != 3
            or center_small.shape[-1] != NUM_CLASSES - 1
        ):
            raise ValueError(
                "Centre probabilities must have shape "
                f"[h,w,{NUM_CLASSES - 1}]; received {center_small.shape}."
            )
        if not np.all(np.isfinite(semantic)):
            raise ValueError("Semantic probabilities contain NaN or infinity.")
        if not np.all(np.isfinite(center_small)):
            raise ValueError("Centre probabilities contain NaN or infinity.")

        semantic_min = float(semantic.min())
        semantic_max = float(semantic.max())
        center_min = float(center_small.min())
        center_max = float(center_small.max())
        if semantic_min < -1e-4 or semantic_max > 1.0 + 1e-4:
            raise ValueError(
                "Semantic probabilities must be in [0,1]; "
                f"observed [{semantic_min:.6g}, {semantic_max:.6g}]."
            )
        if center_min < -1e-4 or center_max > 1.0 + 1e-4:
            raise ValueError(
                "Centre probabilities must be in [0,1]; "
                f"observed [{center_min:.6g}, {center_max:.6g}]."
            )
        if not np.allclose(
            semantic.sum(axis=-1),
            1.0,
            atol=2e-3,
            rtol=2e-3,
        ):
            raise ValueError(
                "Semantic class probabilities do not sum approximately to one."
            )

        height, width = semantic.shape[:2]
        center_full = np.clip(
            resize_channels(center_small, width, height),
            0.0,
            1.0,
        )
        fused = center_full * np.sqrt(
            np.clip(semantic[..., 1:], 0.0, 1.0)
        )
        return np.ascontiguousarray(
            np.max(fused, axis=-1).astype(np.float32)
        )

    @staticmethod
    def offset_preview_rgb(
        offset_vectors_small: np.ndarray,
        foreground_mask: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Show offset direction as hue and displacement as brightness."""
        offsets = np.asarray(offset_vectors_small, dtype=np.float32)
        foreground = np.asarray(foreground_mask, dtype=bool)

        if offsets.ndim != 3 or offsets.shape[-1] != 2:
            raise ValueError(
                "Offset prediction must have shape [h,w,2]; "
                f"received {offsets.shape}."
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("Offset prediction contains NaN or infinity.")

        minimum = float(offsets.min())
        maximum = float(offsets.max())
        if minimum < -1.001 or maximum > 1.001:
            raise ValueError(
                "Offset prediction must be in approximately [-1,1]; "
                f"observed [{minimum:.6g}, {maximum:.6g}]."
            )

        height, width = (int(output_shape[0]), int(output_shape[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid output shape: {output_shape}.")
        if foreground.shape != (height, width):
            raise ValueError(
                "Foreground mask shape does not match output shape: "
                f"{foreground.shape} versus {(height, width)}."
            )

        offsets_full = resize_channels(offsets, width, height)
        dx = offsets_full[..., 0] * width
        dy = offsets_full[..., 1] * height
        angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        magnitude = np.hypot(dx, dy)
        normalized_magnitude = np.clip(
            magnitude / max(float(np.hypot(width, height)), 1.0),
            0.0,
            1.0,
        )

        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 0] = np.rint(
            angle * (179.0 / (2.0 * np.pi))
        ).astype(np.uint8)
        hsv[..., 1] = np.where(foreground, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(
            foreground,
            np.rint(55.0 + 200.0 * np.sqrt(normalized_magnitude)),
            0.0,
        ).astype(np.uint8)
        return np.ascontiguousarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    @staticmethod
    def _finite_log_value(
        logs: dict[str, object],
        key: str,
    ) -> float | None:
        value = logs.get(key)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        epoch_number = int(epoch) + 1
        if (
            epoch_number != 1
            and epoch_number % self.every_n_epochs != 0
        ):
            return
        if self.model is None:
            raise RuntimeError(
                "ValidationPreviewCallback has no attached model."
            )

        logs = {} if logs is None else logs
        title_parts = [f"VALIDATION EPOCH {epoch_number}"]
        log_metrics = (
            ("val_semantic_foreground_miou", "FG mIoU"),
            ("val_boundary_boundary_f1", "Boundary F1"),
            ("val_loss", "Loss"),
        )
        for key, label in log_metrics:
            value = self._finite_log_value(logs, key)
            if value is not None:
                title_parts.append(f"{label} {value:.3f}")
        epoch_title = " | ".join(title_parts)

        sample_grids: list[np.ndarray] = []
        for raw_index in self.indices:
            array_index = int(raw_index)
            try:
                normalized_image = np.asarray(
                    normalize_image(self.arrays["images"][array_index]),
                    dtype=np.float32,
                )
                expected_shape = (IMG_SIZE, IMG_SIZE, 3)
                if normalized_image.shape != expected_shape:
                    raise ValueError(
                        f"Validation image {array_index} has shape "
                        f"{normalized_image.shape}; expected {expected_shape}."
                    )
                if not np.all(np.isfinite(normalized_image)):
                    raise ValueError(
                        f"Validation image {array_index} contains NaN or infinity."
                    )

                model_outputs = unpack_model_outputs(
                    self.model(
                        tf.convert_to_tensor(
                            normalized_image[None, ...],
                            dtype=tf.float32,
                        ),
                        training=False,
                    )
                )
                (
                    semantic_probabilities,
                    inner_distance,
                    boundary_probability,
                ) = output_probabilities(model_outputs, 0)

                target_semantic = _as_label_map(
                    self.arrays["semantic"][array_index]
                )
                predicted_semantic = np.argmax(
                    semantic_probabilities,
                    axis=-1,
                ).astype(np.int32)
                if target_semantic.shape != predicted_semantic.shape:
                    raise ValueError(
                        f"Target and prediction shapes differ for validation "
                        f"image {array_index}: {target_semantic.shape} versus "
                        f"{predicted_semantic.shape}."
                    )

                (
                    predicted_instance_map,
                    predicted_instance_classes,
                    _,
                    predicted_center_scores,
                ) = decode_instances(
                    semantic_probabilities,
                    inner_distance,
                    boundary_probability,
                )
                instance_summaries = summarise_instances(
                    predicted_instance_map,
                    predicted_instance_classes,
                    semantic_probabilities,
                    predicted_center_scores,
                )

                image_uint8 = _as_rgb_uint8(
                    self.arrays["images"][array_index],
                    name=f"validation image {array_index}",
                )
                predicted_distance = resize_channels(
                    np.asarray(inner_distance, dtype=np.float32),
                    IMG_SIZE,
                    IMG_SIZE,
                )[..., 0]
                target_instance_map = np.asarray(
                    self.arrays["instance"][array_index], dtype=np.int32
                )
                target_distance = resize_channels(
                    build_inner_distance_target(
                        target_semantic, target_instance_map
                    )[..., 0:1],
                    IMG_SIZE,
                    IMG_SIZE,
                )[..., 0]
                semantic_error = self.semantic_error_rgb(
                    target_semantic,
                    predicted_semantic,
                )
                instance_overlay = make_instance_overlay(
                    image_uint8,
                    predicted_instance_map,
                    instance_summaries,
                )

                panels = [
                    add_panel_title(image_uint8, "VALIDATION IMAGE"),
                    add_panel_title(
                        colourize_semantic(target_semantic),
                        "GROUND-TRUTH SEMANTIC",
                    ),
                    add_panel_title(
                        colourize_semantic(predicted_semantic),
                        "PREDICTED SEMANTIC",
                    ),
                    add_panel_title(
                        semantic_error,
                        "ERROR: GREEN OK | RED MISS | YELLOW FP | MAGENTA CLASS",
                    ),
                    add_panel_title(
                        probability_heatmap(target_distance),
                        "TARGET INNER DISTANCE",
                    ),
                    add_panel_title(
                        probability_heatmap(predicted_distance),
                        "PREDICTED INNER DISTANCE",
                    ),
                    add_panel_title(
                        probability_heatmap(boundary_probability),
                        "PREDICTED INSTANCE BOUNDARY",
                    ),
                    add_panel_title(
                        instance_overlay,
                        f"DECODED INSTANCES: {len(instance_summaries)}",
                    ),
                ]

                panel_shapes = {panel.shape for panel in panels}
                if len(panel_shapes) != 1:
                    raise RuntimeError(
                        f"Preview panel shapes disagree: {panel_shapes}."
                    )
                top_row = np.concatenate(panels[:4], axis=1)
                bottom_row = np.concatenate(panels[4:], axis=1)
                sample_grid = np.concatenate([top_row, bottom_row], axis=0)
                titled = add_panel_title(
                    sample_grid,
                    f"{epoch_title} | VALIDATION INDEX {array_index}",
                )
                sample_grids.append(titled)
                write_rgb_image(
                    self.output_dir
                    / f"epoch_{epoch_number:04d}_val_{array_index:05d}.png",
                    titled,
                )

            except Exception as error:
                message = (
                    f"Could not create validation preview for index "
                    f"{array_index} at epoch {epoch_number}: {error}"
                )
                if self.fail_on_error:
                    raise RuntimeError(message) from error
                print("WARNING:", message)

        if not sample_grids:
            print(
                f"WARNING: No validation preview was produced "
                f"for epoch {epoch_number}."
            )
            return

        grid_widths = {grid.shape[1] for grid in sample_grids}
        if len(grid_widths) != 1:
            raise RuntimeError(
                f"Validation sample-grid widths disagree: {grid_widths}."
            )

        # A combined sheet as well, but only while it stays viewable.
        if len(sample_grids) <= PREVIEW_CONTACT_SHEET_MAX_IMAGES:
            write_rgb_image(
                self.output_dir / f"epoch_{epoch_number:04d}.png",
                np.concatenate(sample_grids, axis=0),
            )
        print(
            "Validation previews:",
            self.output_dir,
            f"({len(sample_grids)} image(s), one file each, "
            "inference batch size 1)",
        )

# =============================================================================
# 13. RUNTIME, OPTIMIZER, AND TRAINING
# =============================================================================


def _validated_training_class_weights(class_weights: np.ndarray) -> np.ndarray:
    """Return finite, positive class weights with the exact semantic shape."""
    weights = np.asarray(class_weights, dtype=np.float32)
    if weights.shape != (NUM_CLASSES,):
        raise ValueError(
            f"class_weights must have shape ({NUM_CLASSES},); "
            f"received {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("class_weights contain NaN or infinity.")
    if np.any(weights <= 0.0):
        raise ValueError(
            f"Every class weight must be positive; got {weights.tolist()}."
        )
    return np.ascontiguousarray(weights)


def _atomic_replace_text(path: Path, text: str) -> None:
    """Write UTF-8 text completely, then atomically replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(str(text))
            if text and not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _json_compatible(value):
    """Convert common NumPy, TensorFlow, and Path values to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if tf.is_tensor(value):
        return _json_compatible(value.numpy())
    return str(value)


def _model_output_shapes(model: Model) -> dict[str, list[int | None]]:
    """Return serializable output shapes keyed by their training names."""
    if isinstance(model.output, dict):
        named_outputs = model.output.items()
    else:
        named_outputs = zip(model.output_names, model.outputs)
    return {
        str(name): [
            None if dimension is None else int(dimension)
            for dimension in tensor.shape
        ]
        for name, tensor in named_outputs
    }


def configure_runtime() -> None:
    """Configure reproducible TensorFlow execution before model construction."""
    tf.keras.utils.set_random_seed(int(SEED))

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    print(
        "Keras version:",
        getattr(tf.keras, "__version__", "bundled with TensorFlow"),
    )
    print(
        "Mixed-precision policy:",
        tf.keras.mixed_precision.global_policy().name,
    )
    print("Physical GPUs found:", gpus)

    if REQUIRE_GPU and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU was found. Run inside the WSL TensorFlow "
            "environment where the NVIDIA GPU is visible."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            # This normally means TensorFlow initialized the GPU before this
            # function was called. Do not hide that important configuration fact.
            print(
                f"WARNING: could not enable memory growth for {gpu.name}: "
                f"{error}"
            )

    logical_gpus = tf.config.list_logical_devices("GPU") if gpus else []
    print("Logical GPUs available:", logical_gpus)

    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        device_name = details.get("device_name", "unknown GPU")
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
        except (RuntimeError, ValueError):
            memory_growth = "unavailable"
        print(
            f"GPU {gpu.name}: {device_name}; "
            f"memory_growth={memory_growth}"
        )


def cosine_learning_rate(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    peak_lr: float,
    minimum_lr: float,
) -> float:
    """Linear warm-up followed by one continuous cosine decay."""
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    warmup_epochs = int(warmup_epochs)
    peak_lr = float(peak_lr)
    minimum_lr = float(minimum_lr)

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative; received {epoch}.")
    if total_epochs <= 0:
        raise ValueError(f"EPOCHS must be positive; received {total_epochs}.")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError(
            "WARMUP_EPOCHS must satisfy 0 <= WARMUP_EPOCHS < EPOCHS; "
            f"received {warmup_epochs} and {total_epochs}."
        )
    if not np.isfinite(peak_lr) or peak_lr <= 0.0:
        raise ValueError(f"LEARNING_RATE must be positive; got {peak_lr}.")
    if not np.isfinite(minimum_lr) or minimum_lr < 0.0:
        raise ValueError(
            f"MIN_LEARNING_RATE must be non-negative; got {minimum_lr}."
        )
    if minimum_lr > peak_lr:
        raise ValueError(
            "MIN_LEARNING_RATE cannot exceed LEARNING_RATE: "
            f"{minimum_lr} > {peak_lr}."
        )

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(peak_lr * (epoch + 1) / warmup_epochs)

    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 1:
        return peak_lr

    decay_position = np.clip(
        epoch - warmup_epochs,
        0,
        decay_epochs - 1,
    )
    progress = float(decay_position / (decay_epochs - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_lr + (peak_lr - minimum_lr) * cosine)


def scratch_learning_rate(epoch: int, current_lr: float) -> float:
    """Learning-rate schedule for a new random initialization."""
    del current_lr
    return cosine_learning_rate(
        epoch,
        total_epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        peak_lr=LEARNING_RATE,
        minimum_lr=MIN_LEARNING_RATE,
    )


def fine_tune_learning_rate(epoch: int, current_lr: float) -> float:
    """Low-learning-rate schedule for an explicitly loaded checkpoint."""
    del current_lr
    return cosine_learning_rate(
        epoch,
        total_epochs=FINE_TUNE_EPOCHS,
        warmup_epochs=FINE_TUNE_WARMUP_EPOCHS,
        peak_lr=FINE_TUNE_LEARNING_RATE,
        minimum_lr=FINE_TUNE_MIN_LEARNING_RATE,
    )


def validate_transfer_learning_configuration() -> None:
    """Validate the staged V5 -> V7 optimization schedule."""
    total_epochs = int(TRANSFER_EPOCHS)
    boundaries = {
        "head_warmup": int(TRANSFER_HEAD_WARMUP_EPOCHS),
        "deep_start": int(TRANSFER_DEEP_UNFREEZE_EPOCH),
        "p5_start": int(TRANSFER_P5_UNFREEZE_EPOCH),
        "p4_start": int(TRANSFER_P4_UNFREEZE_EPOCH),
        "full_fine_tune": int(TRANSFER_FULL_FINE_TUNE_EPOCH),
    }
    if total_epochs <= 0:
        raise ValueError("TRANSFER_EPOCHS must be positive.")
    if boundaries["head_warmup"] != boundaries["deep_start"]:
        raise ValueError(
            "TRANSFER_HEAD_WARMUP_EPOCHS must equal "
            "TRANSFER_DEEP_UNFREEZE_EPOCH."
        )
    ordered = [
        boundaries["deep_start"],
        boundaries["p5_start"],
        boundaries["p4_start"],
        boundaries["full_fine_tune"],
        total_epochs,
    ]
    if ordered[0] <= 0 or any(
        later <= earlier for earlier, later in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            "Transfer stages must satisfy 0 < deep < p5 < p4 < full "
            f"< epochs; received {ordered}."
        )
    if not 0 <= int(TRANSFER_LR_WARMUP_EPOCHS) <= ordered[0]:
        raise ValueError(
            "TRANSFER_LR_WARMUP_EPOCHS must fall inside the new-layer "
            "warm-up stage."
        )

    learning_rates = np.asarray(
        [
            TRANSFER_NEW_LAYER_LEARNING_RATE,
            TRANSFER_ADAPTATION_LEARNING_RATE,
            TRANSFER_P4_UNFREEZE_LEARNING_RATE,
            TRANSFER_FINE_TUNE_LEARNING_RATE,
            TRANSFER_MIN_LEARNING_RATE,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(learning_rates)) or np.any(learning_rates <= 0):
        raise ValueError(
            "All transfer-learning rates must be finite and positive."
        )
    if not (
        learning_rates[0]
        >= learning_rates[1]
        >= learning_rates[2]
        >= learning_rates[3]
        >= learning_rates[4]
    ):
        raise ValueError(
            "Transfer learning rates must descend from new-layer warm-up to "
            f"minimum; received {learning_rates.tolist()}."
        )


def transfer_learning_rate(epoch: int, current_lr: float) -> float:
    """Piecewise schedule for new layers during progressive unfreezing."""
    del current_lr
    validate_transfer_learning_configuration()
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative; received {epoch}.")

    head_end = int(TRANSFER_HEAD_WARMUP_EPOCHS)
    p4_start = int(TRANSFER_P4_UNFREEZE_EPOCH)
    full_start = int(TRANSFER_FULL_FINE_TUNE_EPOCH)
    warmup_epochs = int(TRANSFER_LR_WARMUP_EPOCHS)

    if epoch < head_end:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(
                TRANSFER_NEW_LAYER_LEARNING_RATE
                * (epoch + 1)
                / warmup_epochs
            )
        return float(TRANSFER_NEW_LAYER_LEARNING_RATE)

    if epoch < p4_start:
        progress = (epoch - head_end) / max(p4_start - head_end - 1, 1)
        return float(
            TRANSFER_ADAPTATION_LEARNING_RATE
            + (
                TRANSFER_P4_UNFREEZE_LEARNING_RATE
                - TRANSFER_ADAPTATION_LEARNING_RATE
            )
            * np.clip(progress, 0.0, 1.0)
        )

    if epoch < full_start:
        progress = (epoch - p4_start) / max(full_start - p4_start - 1, 1)
        return float(
            TRANSFER_P4_UNFREEZE_LEARNING_RATE
            + (
                TRANSFER_FINE_TUNE_LEARNING_RATE
                - TRANSFER_P4_UNFREEZE_LEARNING_RATE
            )
            * np.clip(progress, 0.0, 1.0)
        )

    return cosine_learning_rate(
        epoch - full_start,
        total_epochs=int(TRANSFER_EPOCHS) - full_start,
        warmup_epochs=0,
        peak_lr=float(TRANSFER_FINE_TUNE_LEARNING_RATE),
        minimum_lr=float(TRANSFER_MIN_LEARNING_RATE),
    )


def _linear_unfreeze_multiplier(epoch: int, start: int, end: int) -> float:
    """Ramp one transferred group from 0 to 1, beginning gently at 0.1."""
    epoch = int(epoch)
    start = int(start)
    end = int(end)
    if end <= start:
        raise ValueError(f"Invalid unfreeze interval: start={start}, end={end}.")
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    progress = (epoch - start) / max(end - start - 1, 1)
    return float(0.1 + 0.9 * np.clip(progress, 0.0, 1.0))


def transfer_group_multipliers(epoch: int) -> dict[str, float]:
    """Unfreeze deepest compatible V5 features first, then p5, then p4."""
    validate_transfer_learning_configuration()
    return {
        "deep": _linear_unfreeze_multiplier(
            epoch,
            TRANSFER_DEEP_UNFREEZE_EPOCH,
            TRANSFER_P5_UNFREEZE_EPOCH,
        ),
        "p5": _linear_unfreeze_multiplier(
            epoch,
            TRANSFER_P5_UNFREEZE_EPOCH,
            TRANSFER_P4_UNFREEZE_EPOCH,
        ),
        "p4": _linear_unfreeze_multiplier(
            epoch,
            TRANSFER_P4_UNFREEZE_EPOCH,
            TRANSFER_FULL_FINE_TUNE_EPOCH,
        ),
    }


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v6")
class ProgressiveTransferAdamW(tf.keras.optimizers.AdamW):
    """AdamW with checkpointable gradient multipliers for transferred groups."""

    def __init__(
        self,
        transfer_variable_groups: dict[str, list[str] | tuple[str, ...]],
        initial_transfer_multipliers: dict[str, float] | None = None,
        **kwargs,
    ):
        normalized_groups: dict[str, tuple[str, ...]] = {}
        path_owner: dict[str, str] = {}
        for raw_group_name, raw_paths in dict(transfer_variable_groups).items():
            group_name = str(raw_group_name)
            paths = tuple(sorted({str(path) for path in raw_paths}))
            if not group_name or not paths:
                raise ValueError(
                    "Every transfer optimizer group needs a name and variables."
                )
            for path in paths:
                if not path:
                    raise ValueError("Transfer variable paths cannot be empty.")
                previous = path_owner.get(path)
                if previous is not None:
                    raise ValueError(
                        f"Variable {path!r} belongs to both {previous!r} and "
                        f"{group_name!r}."
                    )
                path_owner[path] = group_name
            normalized_groups[group_name] = paths

        if not normalized_groups:
            raise ValueError("ProgressiveTransferAdamW needs transfer groups.")
        configured_initial = {
            str(name): float(value)
            for name, value in dict(initial_transfer_multipliers or {}).items()
        }
        unknown_initial = sorted(set(configured_initial) - set(normalized_groups))
        if unknown_initial:
            raise ValueError(
                "Initial multipliers contain unknown groups: "
                f"{unknown_initial}."
            )

        resolved_initial = {
            name: float(configured_initial.get(name, 0.0))
            for name in normalized_groups
        }
        for name, value in resolved_initial.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Initial transfer multiplier {name!r} must be in [0,1]."
                )

        # Keras 3 BaseOptimizer.__setattr__ raises if any attribute is set
        # before super().__init__(). Every check above therefore runs on local
        # names only, and nothing is assigned to self until after this call.
        super().__init__(**kwargs)

        self.transfer_variable_groups = normalized_groups
        self.initial_transfer_multipliers = resolved_initial
        self._transfer_path_owner = path_owner
        self.transfer_gradient_multipliers = {
            name: self.add_variable(
                shape=(),
                initializer=tf.keras.initializers.Constant(value),
                dtype="float32",
                name=f"transfer_{name}_gradient_multiplier",
            )
            for name, value in self.initial_transfer_multipliers.items()
        }

    def set_transfer_multipliers(self, values: dict[str, float]) -> None:
        supplied = {str(name): float(value) for name, value in values.items()}
        if set(supplied) != set(self.transfer_gradient_multipliers):
            raise ValueError(
                "Transfer multiplier groups do not match optimizer groups: "
                f"received={sorted(supplied)}, "
                f"expected={sorted(self.transfer_gradient_multipliers)}."
            )
        for name, value in supplied.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Transfer multiplier {name!r} must be in [0,1]; got {value}."
                )
            self.transfer_gradient_multipliers[name].assign(value)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        scaled_pairs = []
        for gradient, variable in list(grads_and_vars):
            if gradient is not None:
                path = _transfer_variable_path(variable)
                group_name = self._transfer_path_owner.get(path)
                if group_name is not None:
                    multiplier = tf.cast(
                        self.transfer_gradient_multipliers[group_name],
                        gradient.dtype,
                    )
                    if isinstance(gradient, tf.IndexedSlices):
                        gradient = tf.IndexedSlices(
                            gradient.values * multiplier,
                            gradient.indices,
                            gradient.dense_shape,
                        )
                    else:
                        gradient = gradient * multiplier
            scaled_pairs.append((gradient, variable))
        return super().apply_gradients(scaled_pairs, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transfer_variable_groups": {
                    name: list(paths)
                    for name, paths in self.transfer_variable_groups.items()
                },
                "initial_transfer_multipliers": dict(
                    self.initial_transfer_multipliers
                ),
            }
        )
        return config


def _find_progressive_transfer_optimizer(optimizer):
    """Unwrap mixed-precision wrappers and return the transfer optimizer."""
    current = optimizer
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ProgressiveTransferAdamW):
            return current
        next_optimizer = None
        for attribute in ("inner_optimizer", "optimizer", "_optimizer"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_optimizer = candidate
                break
        current = next_optimizer
    return None


class ProgressiveTransferScheduleCallback(tf.keras.callbacks.Callback):
    """Apply the deterministic top-down unfreezing schedule each epoch."""

    def on_train_begin(self, logs=None) -> None:
        del logs
        optimizer = _find_progressive_transfer_optimizer(self.model.optimizer)
        if optimizer is None:
            raise RuntimeError(
                "Transfer run is not using ProgressiveTransferAdamW."
            )

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        del logs
        optimizer = _find_progressive_transfer_optimizer(self.model.optimizer)
        if optimizer is None:
            raise RuntimeError(
                "Could not find ProgressiveTransferAdamW at epoch start."
            )
        multipliers = transfer_group_multipliers(epoch)
        optimizer.set_transfer_multipliers(multipliers)
        stage = (
            "new V7 layers only"
            if epoch < TRANSFER_DEEP_UNFREEZE_EPOCH
            else (
                "unfreezing SPPF/attention"
                if epoch < TRANSFER_P5_UNFREEZE_EPOCH
                else (
                    "unfreezing p5"
                    if epoch < TRANSFER_P4_UNFREEZE_EPOCH
                    else (
                        "unfreezing p4"
                        if epoch < TRANSFER_FULL_FINE_TUNE_EPOCH
                        else "full-model fine-tuning"
                    )
                )
            )
        )
        formatted = ", ".join(
            f"{name}={value:.3f}" for name, value in multipliers.items()
        )
        print(
            f"Progressive transfer stage at epoch {int(epoch) + 1}: "
            f"{stage}; gradient multipliers: {formatted}"
        )


def build_optimizer(
    *,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    transfer_variable_groups: dict[str, tuple[str, ...]] | None = None,
    transfer_variables: tuple[object, ...] = (),
):
    """Build AdamW and fail loudly if requested optimizer features are absent."""
    peak_lr = float(learning_rate)
    weight_decay = float(weight_decay)
    accumulation_steps = int(GRADIENT_ACCUMULATION_STEPS)
    ema_momentum = float(EMA_MOMENTUM)
    global_clipnorm = 5.0

    if not np.isfinite(peak_lr) or peak_lr <= 0.0:
        raise ValueError(f"LEARNING_RATE must be positive; got {peak_lr}.")
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(
            f"WEIGHT_DECAY must be finite and non-negative; got {weight_decay}."
        )
    if accumulation_steps < 1:
        raise ValueError(
            "GRADIENT_ACCUMULATION_STEPS must be at least one; "
            f"received {accumulation_steps}."
        )
    if USE_EMA and not 0.0 <= ema_momentum < 1.0:
        raise ValueError(
            f"EMA_MOMENTUM must be in [0,1); received {ema_momentum}."
        )

    try:
        signature = inspect.signature(tf.keras.optimizers.AdamW)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        parameters = {}
        accepts_kwargs = False

    def supports(name: str) -> bool:
        return name in parameters or accepts_kwargs

    if USE_EMA and not supports("use_ema"):
        raise RuntimeError(
            "USE_EMA=True, but this TensorFlow/Keras AdamW implementation "
            "does not expose optimizer EMA. Upgrade TensorFlow or disable EMA "
            "explicitly instead of silently training a different setup."
        )
    if accumulation_steps > 1 and not supports(
        "gradient_accumulation_steps"
    ):
        raise RuntimeError(
            "Gradient accumulation was requested, but this TensorFlow/Keras "
            "AdamW implementation does not support it."
        )

    kwargs: dict[str, object] = {
        "learning_rate": peak_lr,
        "weight_decay": weight_decay,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "global_clipnorm": global_clipnorm,
    }
    if USE_EMA:
        kwargs.update(
            {
                "use_ema": True,
                "ema_momentum": ema_momentum,
            }
        )
    if accumulation_steps > 1:
        kwargs["gradient_accumulation_steps"] = accumulation_steps

    normalized_transfer_groups = {
        str(name): tuple(str(path) for path in paths)
        for name, paths in dict(transfer_variable_groups or {}).items()
    }
    transfer_variables = tuple(transfer_variables)
    if bool(normalized_transfer_groups) != bool(transfer_variables):
        raise ValueError(
            "Transfer optimizer groups and transfer variable objects must be "
            "provided together."
        )
    if normalized_transfer_groups:
        optimizer = ProgressiveTransferAdamW(
            transfer_variable_groups=normalized_transfer_groups,
            initial_transfer_multipliers={
                name: 0.0 for name in normalized_transfer_groups
            },
            **kwargs,
        )
    else:
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
    if hasattr(optimizer, "exclude_from_weight_decay"):
        exclusion_kwargs: dict[str, object] = {
            "var_names": ["bias", "beta", "gamma"]
        }
        if transfer_variables:
            exclusion_parameters = inspect.signature(
                optimizer.exclude_from_weight_decay
            ).parameters
            if "var_list" not in exclusion_parameters:
                raise RuntimeError(
                    "This Keras optimizer cannot exclude exact transferred "
                    "variables from weight decay. Upgrade TensorFlow/Keras "
                    "instead of silently changing the freeze contract."
                )
            # Exact exclusion prevents decoupled AdamW decay from moving a
            # transferred tensor while its gradient multiplier is zero.
            exclusion_kwargs["var_list"] = list(transfer_variables)
        optimizer.exclude_from_weight_decay(**exclusion_kwargs)
    elif transfer_variables:
        raise RuntimeError(
            "Transfer training requires optimizer.exclude_from_weight_decay "
            "to keep gradient-frozen V5 tensors exactly unchanged."
        )

    print("AdamW configuration:")
    displayed_kwargs = dict(kwargs)
    if normalized_transfer_groups:
        displayed_kwargs["progressive_transfer_variable_counts"] = {
            name: len(paths)
            for name, paths in normalized_transfer_groups.items()
        }
    print(json.dumps(_json_compatible(displayed_kwargs), indent=2))
    return optimizer


def compile_model(
    model: Model,
    class_weights: np.ndarray,
    *,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    transfer_variable_groups: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Validate and compile the complete four-head instance model."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a Keras Model; got {type(model)}.")

    weights = _validated_training_class_weights(class_weights)
    validate_model_output_shapes(model)

    if isinstance(model.output, dict):
        observed_outputs = set(model.output)
    else:
        observed_outputs = set(model.output_names)
    expected_outputs = {"semantic", "inner_distance", "boundary"}
    if observed_outputs != expected_outputs:
        raise ValueError(
            "Model outputs do not match the training contract: "
            f"observed={sorted(observed_outputs)}, "
            f"expected={sorted(expected_outputs)}."
        )

    configured_loss_weights = {
        "semantic": float(SEMANTIC_LOSS_WEIGHT),
        "inner_distance": float(INNER_DISTANCE_LOSS_WEIGHT),
        "boundary": float(BOUNDARY_LOSS_WEIGHT),
    }
    loss_weight_values = np.asarray(
        list(configured_loss_weights.values()),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(loss_weight_values)):
        raise ValueError("Loss weights contain NaN or infinity.")
    if np.any(loss_weight_values < 0.0) or not np.any(
        loss_weight_values > 0.0
    ):
        raise ValueError(
            f"Loss weights must be non-negative and not all zero: "
            f"{configured_loss_weights}."
        )

    semantic_metrics: list[tf.keras.metrics.Metric] = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"),
        SparseMeanIoUFromLogits(name="mean_iou"),
        ForegroundMeanIoUFromLogits(name="foreground_miou"),
    ]
    # Every IoU metric argmaxes the full 512x512x5 logit map and scatters a
    # float64 confusion matrix over 524,288 elements, and float64 runs at 1/64
    # rate on this GPU. The four per-class metrics are display-only: nothing
    # selects, stops, or reports from them, and per-class IoU is already
    # produced by semantic_report_from_confusion at evaluation time and by the
    # instance evaluation every INSTANCE_CHECKPOINT_EVERY_N_EPOCHS epochs.
    if PER_CLASS_IOU_METRICS_DURING_TRAINING:
        for class_id in range(1, NUM_CLASSES):
            try:
                class_name = str(CLASS_NAMES[class_id])
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(
                    f"CLASS_NAMES has no entry for class {class_id}."
                ) from error
            metric_suffix = "".join(
                character if character.isalnum() else "_"
                for character in class_name.lower()
            ).strip("_")
            semantic_metrics.append(
                ClassIoUFromLogits(
                    class_id,
                    name=f"iou_{metric_suffix or f'class_{class_id}'}",
                )
            )

    normalized_transfer_groups = {
        str(name): tuple(str(path) for path in paths)
        for name, paths in dict(transfer_variable_groups or {}).items()
    }
    transfer_paths = {
        path for paths in normalized_transfer_groups.values() for path in paths
    }
    model_variables_by_path: dict[str, object] = {}
    for variable in model.trainable_variables:
        path = _transfer_variable_path(variable)
        if path in model_variables_by_path:
            raise ValueError(f"Duplicate trainable variable path: {path!r}.")
        model_variables_by_path[path] = variable
    missing_transfer_paths = sorted(
        transfer_paths - set(model_variables_by_path)
    )
    if missing_transfer_paths:
        raise ValueError(
            "Transfer optimizer paths are absent from the target model: "
            f"{missing_transfer_paths}."
        )
    transfer_variables = tuple(
        model_variables_by_path[path] for path in sorted(transfer_paths)
    )

    compile_kwargs: dict[str, object] = {
        "optimizer": build_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            transfer_variable_groups=normalized_transfer_groups,
            transfer_variables=transfer_variables,
        ),
        "loss": {
            "semantic": SemanticFocalDiceLovaszFromLogits(
                weights,
                focal_weight=0.45,
                dice_weight=0.30,
                lovasz_weight=0.25,
            ),
            "inner_distance": MaskedInnerDistanceLoss(),
            "boundary": BoundaryBCEDiceFromLogits(),
        },
        "loss_weights": configured_loss_weights,
        "metrics": {
            "semantic": semantic_metrics,
            "boundary": [BoundaryF1FromLogits()],
        },
        # This model is 464 mostly-small layers, which is the shape of graph
        # XLA fusion helps most. Whether the custom Lovasz sort and the float64
        # confusion-matrix metrics compile is a property of the installed
        # TensorFlow, so it is measured (benchmark mode) rather than assumed,
        # and left off unless that measurement says otherwise.
        "jit_compile": bool(USE_XLA_JIT),
        "run_eagerly": False,
    }
    try:
        compile_parameters = inspect.signature(model.compile).parameters
    except (TypeError, ValueError):
        compile_parameters = {}
    if "auto_scale_loss" in compile_parameters:
        compile_kwargs["auto_scale_loss"] = True
    if "steps_per_execution" in compile_parameters and STEPS_PER_EXECUTION > 1:
        compile_kwargs["steps_per_execution"] = int(STEPS_PER_EXECUTION)

    model.compile(**compile_kwargs)


def save_model_summary(model: Model, path: Path) -> None:
    """Atomically save an expanded summary with essential runtime metadata."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a Keras Model; got {type(model)}.")

    lines: list[str] = [
        f"Model name: {model.name}",
        f"TensorFlow version: {tf.__version__}",
        (
            "Mixed-precision policy: "
            f"{tf.keras.mixed_precision.global_policy().name}"
        ),
        f"Total parameters: {int(model.count_params()):,}",
        f"Input shape: {model.input_shape}",
        f"Output shapes: {_model_output_shapes(model)}",
        "",
    ]

    def capture(line="", *args, **kwargs) -> None:
        del args, kwargs
        lines.append(str(line))

    summary_kwargs: dict[str, object] = {"print_fn": capture}
    try:
        summary_parameters = inspect.signature(model.summary).parameters
    except (TypeError, ValueError):
        summary_parameters = {}
    if "expand_nested" in summary_parameters:
        summary_kwargs["expand_nested"] = True
    if "show_trainable" in summary_parameters:
        summary_kwargs["show_trainable"] = True
    model.summary(**summary_kwargs)
    _atomic_replace_text(Path(path), "\n".join(lines))


def save_training_configuration(
    class_weights: np.ndarray,
    model: Model,
    *,
    output_dir: Path = MODEL_OUTPUT_DIR,
    array_dir: Path = ARRAY_DIR,
    training_mode: str = "from_scratch",
    source_model_path: Path | None = None,
    epochs: int = EPOCHS,
    peak_learning_rate: float = LEARNING_RATE,
    minimum_learning_rate: float = MIN_LEARNING_RATE,
    warmup_epochs: int = WARMUP_EPOCHS,
    weight_decay: float = WEIGHT_DECAY,
    instance_checkpoint_every_n_epochs: int = (
        INSTANCE_CHECKPOINT_EVERY_N_EPOCHS
    ),
    instance_checkpoint_max_images: int = INSTANCE_CHECKPOINT_MAX_IMAGES,
    instance_early_stopping_patience: int | None = None,
    transfer_manifest: dict[str, object] | None = None,
) -> None:
    """Save an auditable description of the exact experiment configuration."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a Keras Model; got {type(model)}.")
    weights = _validated_training_class_weights(class_weights)
    output_dir = Path(output_dir)
    array_dir = Path(array_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_model_path = (
        None if source_model_path is None else Path(source_model_path)
    )
    transfer_manifest = (
        None if transfer_manifest is None else dict(transfer_manifest)
    )
    is_v5_transfer = str(training_mode) == "transfer_from_v5"
    if is_v5_transfer and transfer_manifest is None:
        raise ValueError(
            "transfer_from_v5 training requires an audited transfer manifest."
        )
    if not is_v5_transfer and transfer_manifest is not None:
        raise ValueError(
            "A transfer manifest was supplied for a non-transfer run."
        )

    classes: list[dict[str, object]] = []
    for class_id in range(NUM_CLASSES):
        try:
            class_name = str(CLASS_NAMES[class_id])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"CLASS_NAMES has no entry for class {class_id}."
            ) from error
        classes.append({"class_id": class_id, "class_name": class_name})

    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    non_trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.non_trainable_weights)
    )

    gpu_descriptions: list[dict[str, str]] = []
    for gpu in tf.config.list_physical_devices("GPU"):
        details = tf.config.experimental.get_device_details(gpu)
        gpu_descriptions.append(
            {
                "physical_name": str(gpu.name),
                "device_name": str(details.get("device_name", "unknown GPU")),
            }
        )

    optimizer = getattr(model, "optimizer", None)
    optimizer_record = None
    if optimizer is not None:
        optimizer_record = {
            "class_name": optimizer.__class__.__name__,
            "config": _json_compatible(optimizer.get_config()),
        }

    configuration = {
        "version": str(model.name),
        "training": str(training_mode),
        "pretrained_encoder": bool(is_v5_transfer),
        "pretrained_weights_loaded": bool(is_v5_transfer),
        "trained_scratch_checkpoint_loaded": False,
        "complete_v6_checkpoint_loaded": bool(
            source_model_path is not None and not is_v5_transfer
        ),
        "source_model": (
            str(source_model_path) if source_model_path is not None else None
        ),
        "paths": {
            "dataset_root": str(DATASET_ROOT),
            "array_dir": str(array_dir),
            "model_output_dir": str(output_dir),
        },
        "runtime": {
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(
                tf.keras,
                "__version__",
                "bundled with TensorFlow",
            ),
            "mixed_precision_policy": (
                tf.keras.mixed_precision.global_policy().name
            ),
            "physical_gpus": gpu_descriptions,
            "seed": int(SEED),
        },
        "model": {
            "name": str(model.name),
            "input_shape": _json_compatible(model.input_shape),
            "output_shapes": _model_output_shapes(model),
            "total_parameters": int(model.count_params()),
            "trainable_parameters": trainable_parameters,
            "non_trainable_parameters": non_trainable_parameters,
            "classes": classes,
        },
        "spatial_contract": {
            "image_size": int(IMG_SIZE),
            "full_resolution": [int(IMG_SIZE), int(IMG_SIZE)],
            "instance_head_size": int(INSTANCE_HEAD_SIZE),
            "instance_head_resolution": [
                int(INSTANCE_HEAD_SIZE),
                int(INSTANCE_HEAD_SIZE),
            ],
            "offset_normalization": (
                "dx / instance_head_width, dy / instance_head_height"
            ),
            "instance_id_dtype": str(np.dtype(INSTANCE_ID_DTYPE)),
            "maximum_instance_id": int(MAX_INSTANCE_ID),
        },
        "preprocessing": {
            "method": "shared letterbox for prepare and predict",
            "letterbox_fill_value": int(LETTERBOX_FILL_VALUE),
        },
        "optimization": {
            "optimizer": optimizer_record,
            "batch_size": int(BATCH_SIZE),
            "gradient_accumulation_steps": int(
                GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_batch_size": int(
                BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "epochs": int(epochs),
            "peak_learning_rate": float(peak_learning_rate),
            "minimum_learning_rate": float(minimum_learning_rate),
            "warmup_epochs": int(warmup_epochs),
            "schedule": (
                "new-layer warm-up, discriminative progressive unfreezing, "
                "then full-model cosine fine-tuning"
                if is_v5_transfer
                else "linear warm-up then cosine decay"
            ),
            "weight_decay": float(weight_decay),
            "use_ema": bool(USE_EMA),
            "ema_momentum": float(EMA_MOMENTUM) if USE_EMA else None,
            "global_clipnorm": 5.0,
        },
        "transfer_learning": (
            {
                "source_architecture": "Model_v5_PCB_Instance",
                "target_architecture": str(model.name),
                "copy_policy": str(transfer_manifest["method"]),
                "copied_layer_count": int(
                    transfer_manifest["copied_layer_count"]
                ),
                "copied_parameter_count": int(
                    transfer_manifest["copied_parameter_count"]
                ),
                "target_parameter_coverage": float(
                    transfer_manifest["target_parameter_coverage"]
                ),
                "manifest_file": "transfer_manifest.json",
                "new_layers_base_learning_rate": float(
                    TRANSFER_NEW_LAYER_LEARNING_RATE
                ),
                "adaptation_learning_rate": float(
                    TRANSFER_ADAPTATION_LEARNING_RATE
                ),
                "p4_unfreeze_learning_rate": float(
                    TRANSFER_P4_UNFREEZE_LEARNING_RATE
                ),
                "full_fine_tune_learning_rate": float(
                    TRANSFER_FINE_TUNE_LEARNING_RATE
                ),
                "minimum_learning_rate": float(
                    TRANSFER_MIN_LEARNING_RATE
                ),
                "unfreeze_epochs_zero_based": {
                    "deep_sppf_attention": int(
                        TRANSFER_DEEP_UNFREEZE_EPOCH
                    ),
                    "p5": int(TRANSFER_P5_UNFREEZE_EPOCH),
                    "p4": int(TRANSFER_P4_UNFREEZE_EPOCH),
                    "all_transferred_at_full_rate": int(
                        TRANSFER_FULL_FINE_TUNE_EPOCH
                    ),
                },
                "fresh_optimizer": True,
                "source_optimizer_state_loaded": False,
                "transferred_variables_excluded_from_weight_decay": True,
                "source_checkpoint_modified": False,
            }
            if is_v5_transfer
            else None
        ),
        "losses": {
            "semantic": {
                "name": "SemanticFocalDiceLovaszFromLogits",
                "component_weights": {
                    "focal": 0.45,
                    "dice": 0.30,
                    "lovasz": 0.25,
                },
                "class_weights": weights.tolist(),
            },
            "inner_distance": {
                "name": "MaskedInnerDistanceLoss",
                "delta": 0.10,
                "rim_weight": 2.0,
            },
            "boundary": {"name": "BoundaryBCEDiceFromLogits"},
            # These must match compile_model's loss_weights exactly. They did
            # not: the record still named `center` and `offset`, the two heads
            # V7 removed, and omitted `inner_distance`, the one it trains. A run
            # is reconstructed from this file, and on this project a config that
            # misdescribes a run has already cost one.
            "head_weights": {
                "semantic": float(SEMANTIC_LOSS_WEIGHT),
                "inner_distance": float(INNER_DISTANCE_LOSS_WEIGHT),
                "boundary": float(BOUNDARY_LOSS_WEIGHT),
            },
        },
        "augmentation": {
            "source": (
                "Model_v5 augmentation implementation plus a dihedral "
                "stream and a zoom-out branch"
            ),
            "cycle_length_epochs": int(AUGMENTATION_CYCLE_LENGTH),
            "dihedral_transforms": (
                8 if USE_DIHEDRAL_AUGMENTATION else 0
            ),
            "zoom_out_range": _json_compatible(ZOOM_OUT_RANGE),
            "zoom_out_steps": _json_compatible(ZOOM_OUT_STEPS),
            "zoom_out_fill_value": int(ZOOM_OUT_FILL_VALUE),
            "zoom_out_probability": float(ZOOM_OUT_PROBABILITY),
            "mirror_side_probability": (
                float(MIRROR_SIDE_PROBABILITY)
                if USE_MIRROR_SIDE_AUGMENTATION
                else 0.0
            ),
            "mirror_sides_by_epoch": list(MIRROR_SIDES),
            "zoom_in_range": _json_compatible(ZOOM_IN_RANGE),
            "zoom_in_steps": _json_compatible(ZOOM_IN_STEPS),
            "zoom_in_probability": float(ZOOM_IN_PROBABILITY),
            "original_modes": 1,
            "exact_rotation_modes": int(EXACT_ROTATION_MODE_COUNT),
            "realistic_variant_modes": int(REALISTIC_VARIANT_MODE_COUNT),
            "exact_rotation_angles": [
                float(value) for value in EXACT_ROTATION_ANGLES
            ],
            "rotate_realistic_variants_both_directions": bool(
                ROTATE_BOTH_DIRECTIONS
            ),
            "zoom_to_fill": True,
            "zoom_safety_factor": float(ZOOM_SAFETY_FACTOR),
            "extra_zoom_range": _json_compatible(EXTRA_ZOOM_RANGE),
            "maximum_translation_fraction": float(
                MAX_TRANSLATION_FRACTION
            ),
            "brightness_limit": float(BRIGHTNESS_LIMIT),
            "contrast_limit": float(CONTRAST_LIMIT),
            "gamma_range": _json_compatible(GAMMA_RANGE),
            "exposure_gain_range": _json_compatible(EXPOSURE_GAIN_RANGE),
            "noise_std_range": _json_compatible(NOISE_STD_RANGE),
            "hue_shift_degrees": float(HUE_SHIFT_DEGREES),
            "saturation_gain_range": _json_compatible(
                SATURATION_GAIN_RANGE
            ),
            "validation_augmented": False,
            "test_augmented": False,
        },
        "checkpoint_selection": {
            "semantic_monitor": "val_semantic_foreground_miou",
            "instance_monitor": (
                f"validation instance F1 at IoU "
                f"{float(INSTANCE_EVALUATION_IOU):.4g}"
            ),
            "instance_interval_epochs": int(
                instance_checkpoint_every_n_epochs
            ),
            "instance_subset_images": int(
                instance_checkpoint_max_images
            ),
            "instance_early_stopping_patience_evaluations": (
                None
                if instance_early_stopping_patience is None
                else int(instance_early_stopping_patience)
            ),
            "final_candidate_comparison": "complete validation split",
            "primary_final_metric": "instance F1",
            "tie_breaker": (
                "mask mAP50-95, then semantic foreground mIoU, "
                "instance recall and instance precision"
            ),
        },
        "mask_average_precision": {
            "headline_metrics": ["mask mAP50", "mask mAP50-95"],
            "iou_thresholds": [
                float(value)
                for value in MASK_AP_IOU_THRESHOLDS
            ],
            "recall_interpolation_points": int(
                MASK_AP_RECALL_POINTS
            ),
            "max_detections_per_image": int(
                MASK_AP_MAX_DETECTIONS_PER_IMAGE
            ),
            "confidence_source": (
                "mean class-specific semantic probability over each "
                "decoded instance mask"
            ),
        },
    }

    destination = output_dir / "training_config.json"
    serialized = json.dumps(
        _json_compatible(configuration),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_replace_text(destination, serialized)
    print("Training configuration:", destination)


def compare_candidate_models(
    candidate_paths: list[Path],
    val_arrays: dict[str, np.ndarray],
) -> tuple[Path, list[dict[str, object]]]:
    """Select the best candidate using the complete validation instance F1."""
    if "images" not in val_arrays:
        raise KeyError("val_arrays must contain 'images'.")
    validation_size = len(val_arrays["images"])
    if validation_size == 0:
        raise ValueError("The validation split is empty.")
    if not candidate_paths:
        raise ValueError("candidate_paths must contain at least one path.")

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        path = Path(candidate).expanduser().resolve(strict=False)
        identity = os.path.normcase(str(path))
        if identity not in seen:
            seen.add(identity)
            unique_paths.append(path)

    indices = np.arange(validation_size, dtype=np.int64)
    successful_reports: list[dict[str, object]] = []
    failures: list[str] = []

    def validated_metric(value, name: str, path: Path) -> float:
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(
                f"{path} produced invalid {name}={number}; expected [0,1]."
            )
        return number

    for path in unique_paths:
        if not path.is_file():
            message = f"Candidate checkpoint does not exist: {path}"
            failures.append(message)
            print("WARNING:", message)
            continue
        if path.stat().st_size <= 0:
            message = f"Candidate checkpoint is empty: {path}"
            failures.append(message)
            print("WARNING:", message)
            continue

        candidate_model = None
        try:
            print(
                "\nComparing candidate on complete validation split:",
                path,
            )
            candidate_model = tf.keras.models.load_model(path, compile=False)
            validate_model_output_shapes(candidate_model)
            semantic_report, instance_report = evaluate_model_arrays(
                candidate_model,
                val_arrays,
                indices,
                show_progress=True,
            )

            overall = instance_report["overall"]
            instance_f1 = validated_metric(overall["f1"], "instance F1", path)
            instance_precision = validated_metric(
                overall["precision"],
                "instance precision",
                path,
            )
            instance_recall = validated_metric(
                overall["recall"],
                "instance recall",
                path,
            )
            mask_map50 = validated_metric(
                instance_report["mask_map50"],
                "mask mAP50",
                path,
            )
            mask_map50_95 = validated_metric(
                instance_report["mask_map50_95"],
                "mask mAP50-95",
                path,
            )
            foreground_miou = validated_metric(
                semantic_report["foreground_mean_iou"],
                "semantic foreground mIoU",
                path,
            )

            successful_reports.append(
                {
                    "path": str(path),
                    "status": "evaluated",
                    "images_evaluated": validation_size,
                    "instance_iou_threshold": float(
                        INSTANCE_EVALUATION_IOU
                    ),
                    "instance_f1": instance_f1,
                    "instance_precision": instance_precision,
                    "instance_recall": instance_recall,
                    "mask_map50": mask_map50,
                    "mask_map50_95": mask_map50_95,
                    "semantic_foreground_miou": foreground_miou,
                    "parameters": int(candidate_model.count_params()),
                    "file_size_bytes": int(path.stat().st_size),
                }
            )
        except Exception as error:
            message = f"Candidate evaluation failed for {path}: {error}"
            failures.append(message)
            print("WARNING:", message)
        finally:
            if candidate_model is not None:
                del candidate_model
            try:
                tf.keras.backend.clear_session(free_memory=True)
            except TypeError:
                tf.keras.backend.clear_session()
            gc.collect()

    if not successful_reports:
        detail = "\n".join(f"- {message}" for message in failures)
        raise RuntimeError(
            "No candidate checkpoint completed final validation evaluation."
            + (f"\n{detail}" if detail else "")
        )

    # Rank on the metric that is reported. Instance F1 at IoU 0.50 cannot
    # distinguish a mask that just clears 0.50 from one that is nearly exact.
    successful_reports.sort(
        key=lambda report: (
            float(report["mask_map50_95"]),
            float(report["instance_f1"]),
            float(report["semantic_foreground_miou"]),
            float(report["instance_recall"]),
            float(report["instance_precision"]),
        ),
        reverse=True,
    )
    for rank, report in enumerate(successful_reports, start=1):
        report["rank"] = rank

    selected = Path(str(successful_reports[0]["path"]))
    print(
        "\nSelected final checkpoint:",
        selected,
        f"(instance F1={successful_reports[0]['instance_f1']:.4f}, "
        f"mask mAP50-95="
        f"{successful_reports[0]['mask_map50_95']:.4f}, "
        f"foreground mIoU="
        f"{successful_reports[0]['semantic_foreground_miou']:.4f})",
    )
    if failures:
        print(
            f"WARNING: {len(failures)} candidate(s) were unavailable or failed; "
            "see the messages above."
        )
    return selected, successful_reports


def train(*, fine_tune: bool = False, transfer: bool = False) -> None:
    """Train from scratch, transfer from V5, or fine-tune complete V7."""
    import csv
    import shutil

    # Runtime policy must be selected before the model or optimizer is created.
    validate_augmentation_configuration()
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if USE_MIXED_PRECISION else "float32"
    )
    configure_runtime()  # Also establishes the reproducible global seed.

    fine_tune = bool(fine_tune)
    transfer = bool(transfer)
    if fine_tune and transfer:
        raise ValueError("fine_tune and transfer modes are mutually exclusive.")
    if fine_tune:
        run_output_dir = Path(FINE_TUNE_OUTPUT_DIR)
        run_array_dir = Path(FINE_TUNE_ARRAY_DIR)
        source_model_path: Path | None = Path(FINE_TUNE_SOURCE_MODEL)
        total_epochs = int(FINE_TUNE_EPOCHS)
        peak_learning_rate = float(FINE_TUNE_LEARNING_RATE)
        minimum_learning_rate = float(FINE_TUNE_MIN_LEARNING_RATE)
        warmup_epochs = int(FINE_TUNE_WARMUP_EPOCHS)
        weight_decay = float(FINE_TUNE_WEIGHT_DECAY)
        patience = int(EARLY_STOPPING_PATIENCE)
        configured_early_stopping_start = int(
            FINE_TUNE_INSTANCE_EARLY_STOPPING_START_EPOCH
        )
        instance_checkpoint_every_n_epochs = int(
            FINE_TUNE_INSTANCE_CHECKPOINT_EVERY_N_EPOCHS
        )
        instance_checkpoint_max_images = int(
            FINE_TUNE_INSTANCE_CHECKPOINT_MAX_IMAGES
        )
        instance_early_stopping_patience = int(
            FINE_TUNE_INSTANCE_EARLY_STOPPING_PATIENCE_EVALUATIONS
        )
        learning_rate_schedule = fine_tune_learning_rate
        run_label = "fine_tune"
    elif transfer:
        validate_transfer_learning_configuration()
        run_output_dir = Path(TRANSFER_OUTPUT_DIR)
        run_array_dir = Path(TRANSFER_ARRAY_DIR)
        source_model_path = Path(TRANSFER_SOURCE_MODEL)
        total_epochs = int(TRANSFER_EPOCHS)
        peak_learning_rate = float(TRANSFER_NEW_LAYER_LEARNING_RATE)
        minimum_learning_rate = float(TRANSFER_MIN_LEARNING_RATE)
        warmup_epochs = int(TRANSFER_LR_WARMUP_EPOCHS)
        weight_decay = float(TRANSFER_WEIGHT_DECAY)
        patience = int(EARLY_STOPPING_PATIENCE)
        configured_early_stopping_start = int(
            TRANSFER_INSTANCE_EARLY_STOPPING_START_EPOCH
        )
        instance_checkpoint_every_n_epochs = int(
            TRANSFER_INSTANCE_CHECKPOINT_EVERY_N_EPOCHS
        )
        instance_checkpoint_max_images = int(
            TRANSFER_INSTANCE_CHECKPOINT_MAX_IMAGES
        )
        instance_early_stopping_patience = int(
            TRANSFER_INSTANCE_EARLY_STOPPING_PATIENCE_EVALUATIONS
        )
        learning_rate_schedule = transfer_learning_rate
        run_label = "transfer_from_v5"
    else:
        run_output_dir = Path(MODEL_OUTPUT_DIR)
        run_array_dir = Path(ARRAY_DIR)
        source_model_path = None
        total_epochs = int(EPOCHS)
        peak_learning_rate = float(LEARNING_RATE)
        minimum_learning_rate = float(MIN_LEARNING_RATE)
        warmup_epochs = int(WARMUP_EPOCHS)
        weight_decay = float(WEIGHT_DECAY)
        patience = int(EARLY_STOPPING_PATIENCE)
        configured_early_stopping_start = int(EARLY_STOPPING_START_EPOCH)
        instance_checkpoint_every_n_epochs = int(
            INSTANCE_CHECKPOINT_EVERY_N_EPOCHS
        )
        instance_checkpoint_max_images = int(
            INSTANCE_CHECKPOINT_MAX_IMAGES
        )
        instance_early_stopping_patience = None
        learning_rate_schedule = scratch_learning_rate
        run_label = "from_scratch"

    augmentation_cycle = int(AUGMENTATION_CYCLE_LENGTH)
    # The cycle guard protects a run whose weights have not yet seen every V5
    # mode. A fine-tune starts from a model that completed the full cycle, so
    # its configured start is honoured; otherwise a 100-epoch fine-tune could
    # only ever stop in its last five epochs.
    effective_early_stopping_start = (
        configured_early_stopping_start
        if fine_tune
        else max(configured_early_stopping_start, augmentation_cycle)
    )
    if total_epochs <= augmentation_cycle:
        raise ValueError(
            "The selected epoch count must exceed AUGMENTATION_CYCLE_LENGTH "
            "so every "
            f"V5 mode is completed before stopping; received "
            f"epochs={total_epochs}, cycle={augmentation_cycle}."
        )
    if total_epochs <= effective_early_stopping_start:
        raise ValueError(
            "EPOCHS must exceed the effective early-stopping start; "
            f"received {total_epochs} and {effective_early_stopping_start}."
        )
    if patience <= 0:
        raise ValueError(
            f"EARLY_STOPPING_PATIENCE must be positive; received {patience}."
        )
    if instance_checkpoint_every_n_epochs <= 0:
        raise ValueError(
            "The selected instance-checkpoint interval must be positive."
        )
    if instance_checkpoint_max_images < 0:
        raise ValueError(
            "The selected instance-checkpoint maximum-images value cannot be "
            "negative."
        )
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError(
            "Warm-up epochs must satisfy 0 <= warmup < total epochs; "
            f"received warmup={warmup_epochs}, total={total_epochs}."
        )
    if fine_tune or transfer:
        assert source_model_path is not None
        if not source_model_path.is_file():
            raise FileNotFoundError(
                "Selected source model was not found: "
                f"{source_model_path}"
            )
        if source_model_path.resolve() == run_output_dir.resolve(strict=False):
            raise ValueError(
                "The source model must be a model file, not the output "
                "directory."
            )
        if source_model_path.parent.resolve() == run_output_dir.resolve(
            strict=False
        ):
            raise ValueError(
                "Keep the source checkpoint outside the selected output "
                "directory so it can never be overwritten."
            )

    print(
        "Mixed-precision policy:",
        tf.keras.mixed_precision.global_policy().name,
    )
    print(
        "V5 augmentation cycle: 1 original + 45 exact rotations + "
        "49 realistic variants = 95 modes."
    )
    print(
        "Training mode:",
        (
            "fine-tuning all V7 layers"
            if fine_tune
            else (
                "audited V5 -> V7 progressive transfer"
                if transfer
                else "random initialization"
            )
        ),
    )
    print(
        "Learning-rate range:",
        f"{peak_learning_rate:.3g} -> {minimum_learning_rate:.3g}",
    )
    if fine_tune:
        print("Fine-tuning source model:", source_model_path)
        print("Fine-tuning prepared arrays:", run_array_dir)
    elif transfer:
        print("Read-only Model_v5 source:", source_model_path)
        print("V7 transfer prepared arrays:", run_array_dir)
        print(
            "Transfer stages (zero-based epochs): new layers only [0, "
            f"{TRANSFER_DEEP_UNFREEZE_EPOCH}), deep unfreeze at "
            f"{TRANSFER_DEEP_UNFREEZE_EPOCH}, p5 at "
            f"{TRANSFER_P5_UNFREEZE_EPOCH}, p4 at "
            f"{TRANSFER_P4_UNFREEZE_EPOCH}, full fine-tune at "
            f"{TRANSFER_FULL_FINE_TUNE_EPOCH}."
        )
    if effective_early_stopping_start != configured_early_stopping_start:
        print(
            "Early-stopping start was raised to",
            effective_early_stopping_start,
            "to protect the complete augmentation cycle.",
        )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = run_output_dir / "validation_previews"
    tensorboard_dir = run_output_dir / "tensorboard"
    backup_dir = run_output_dir / "fit_backup"
    training_log_path = run_output_dir / "training_log.csv"
    instance_history_path = (
        run_output_dir / "instance_checkpoint_history.json"
    )
    filename_prefix = (
        "fine_tuned_" if fine_tune else ("transfer_" if transfer else "")
    )
    best_semantic_path = run_output_dir / (
        f"best_{filename_prefix}semantic_model_v7.keras"
    )
    best_instance_path = run_output_dir / (
        f"best_{filename_prefix}instance_model_v7.keras"
    )
    last_path = run_output_dir / f"last_{filename_prefix}model_v7.keras"
    final_path = run_output_dir / (
        f"best_{filename_prefix}model_v7_instance.keras"
    )
    selection_path = run_output_dir / "final_model_selection.json"

    resume_available = backup_dir.is_dir() and any(
        item.is_file() for item in backup_dir.rglob("*")
    )
    existing_checkpoints = sorted(run_output_dir.glob("*.keras"))
    if not resume_available and existing_checkpoints:
        formatted = "\n".join(
            f"- {path}" for path in existing_checkpoints[:12]
        )
        raise FileExistsError(
            "The selected output directory already contains model checkpoints "
            "but no active BackupAndRestore state. Use a new output directory "
            f"so experiments cannot be mixed:\n{formatted}"
        )

    print(
        "Training state:",
        "resuming from fit_backup"
        if resume_available
        else (
            "new fine-tuning run"
            if fine_tune
            else ("new V5 transfer run" if transfer else "new scratch run")
        ),
    )

    train_arrays = load_split_arrays("train", array_dir=run_array_dir)
    val_arrays = load_split_arrays("val", array_dir=run_array_dir)
    train_count = len(train_arrays["images"])
    validation_count = len(val_arrays["images"])
    if train_count == 0 or validation_count == 0:
        raise ValueError(
            "Training and validation splits must both be non-empty; "
            f"received train={train_count}, val={validation_count}."
        )

    validation_counts = semantic_class_pixel_counts(val_arrays["semantic"])
    if validation_counts.shape != (NUM_CLASSES,):
        raise ValueError(
            "Unexpected validation class-count shape: "
            f"{validation_counts.shape}."
        )
    missing_val_classes = [
        class_id
        for class_id in range(1, NUM_CLASSES)
        if int(validation_counts[class_id]) == 0
    ]
    if missing_val_classes:
        missing_names = [
            str(CLASS_NAMES[class_id]) for class_id in missing_val_classes
        ]
        raise ValueError(
            "Validation split is missing foreground classes: "
            f"IDs={missing_val_classes}, names={missing_names}."
        )

    print(f"Split sizes: train={train_count:,}, val={validation_count:,}")
    print("Validation pixel counts:", validation_counts.tolist())
    class_weights = compute_class_weights(train_arrays["semantic"])

    train_data = InstanceArraySequence(
        train_arrays,
        BATCH_SIZE,
        training=True,
        seed_offset=100,
    )
    val_data = InstanceArraySequence(
        val_arrays,
        BATCH_SIZE,
        training=False,
        seed_offset=200,
        shuffle=False,
    )
    if len(train_data) <= 0 or len(val_data) <= 0:
        raise ValueError(
            f"Sequences must contain batches; train={len(train_data)}, "
            f"val={len(val_data)}."
        )
    print(f"Batches per epoch: train={len(train_data):,}, val={len(val_data):,}")

    if not resume_available:
        save_dataset_preview(
            run_output_dir,
            array_dir=run_array_dir,
        )

    transfer_variable_groups: dict[str, tuple[str, ...]] = {}
    transfer_manifest: dict[str, object] | None = None
    if fine_tune:
        assert source_model_path is not None
        # compile=False intentionally discards the source optimizer and its old
        # learning-rate/EMA state. Fine-tuning receives a fresh low-LR AdamW.
        model = tf.keras.models.load_model(
            source_model_path,
            compile=False,
        )
        # Keras propagates the trainable flag through nested layers/models.
        model.trainable = True
    elif transfer:
        assert source_model_path is not None
        (
            model,
            transfer_variable_groups,
            transfer_manifest,
        ) = initialize_model_v7_from_v5(source_model_path)
        manifest_path = run_output_dir / "transfer_manifest.json"
        if resume_available:
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    "Transfer BackupAndRestore state exists but its audit "
                    f"manifest is missing: {manifest_path}"
                )
            try:
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Could not validate resume manifest: {manifest_path}"
                ) from error
            if not isinstance(existing_manifest, dict):
                raise ValueError("Transfer resume manifest must be a JSON object.")
            stable_manifest_fields = (
                "source_checkpoint_sha256",
                "source_model_name",
                "target_model_name",
                "method",
                "copied_layer_count",
                "copied_parameter_count",
                "group_parameter_counts",
                "copied_target_layers",
                "gradient_groups",
            )
            mismatched_fields = [
                field
                for field in stable_manifest_fields
                if existing_manifest.get(field) != transfer_manifest.get(field)
            ]
            if mismatched_fields:
                raise RuntimeError(
                    "Transfer source or mapping changed since the interrupted "
                    "run; refusing to restore incompatible optimizer/model "
                    f"state. Mismatched manifest fields: {mismatched_fields}."
                )
            transfer_manifest = existing_manifest
            print("Resume transfer manifest verified:", manifest_path)
        else:
            _atomic_replace_text(
                manifest_path,
                json.dumps(
                    _json_compatible(transfer_manifest),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
            )
            print("Audited transfer manifest:", manifest_path)
    else:
        # Direct scratch builder: no compatibility alias or pretrained encoder.
        model = build_model_v7_instance()
    validate_model_output_shapes(model)

    # Both modes train the complete architecture. Parameter-free fixed
    # operations (for example Sobel features) are allowed.
    visited_layers: set[int] = set()
    frozen_parameter_layers: list[tuple[str, int]] = []

    def inspect_layer_tree(parent) -> None:
        for layer in getattr(parent, "layers", []):
            identity = id(layer)
            if identity in visited_layers:
                continue
            visited_layers.add(identity)
            parameter_count = int(layer.count_params())
            if not layer.trainable and parameter_count > 0:
                frozen_parameter_layers.append(
                    (str(layer.name), parameter_count)
                )
            inspect_layer_tree(layer)

    inspect_layer_tree(model)
    if frozen_parameter_layers:
        raise RuntimeError(
            "Model contains unexpectedly frozen parameterized layers: "
            f"{frozen_parameter_layers}."
        )

    compile_model(
        model,
        class_weights,
        learning_rate=peak_learning_rate,
        weight_decay=weight_decay,
        transfer_variable_groups=(
            transfer_variable_groups if transfer else None
        ),
    )
    if fine_tune:
        print(
            "Fine-tuning initialization confirmed: a complete trained V7 "
            "checkpoint was loaded and all parameterized layers are trainable."
        )
        print("A fresh optimizer was created; source optimizer state was ignored.")
    elif transfer:
        assert transfer_manifest is not None
        print(
            "Selective transfer confirmed:",
            f"{int(transfer_manifest['copied_layer_count']):,} layers and",
            f"{int(transfer_manifest['copied_parameter_count']):,} parameters",
            "copied with exact post-copy verification.",
        )
        print(
            "V7 stem, p1-p3, BiFPN, contexts, decoders, and all heads "
            "retain their native initialization."
        )
        print(
            "A fresh progressive AdamW optimizer was created; no V5 "
            "optimizer or EMA state was loaded."
        )
    else:
        print("Scratch initialization confirmed: no pretrained weights were loaded.")
    print("Model parameters:", f"{model.count_params():,}")
    model.summary()
    save_model_summary(model, run_output_dir / "model_summary.txt")
    save_training_configuration(
        class_weights,
        model,
        output_dir=run_output_dir,
        array_dir=run_array_dir,
        training_mode=run_label,
        source_model_path=source_model_path,
        epochs=total_epochs,
        peak_learning_rate=peak_learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        instance_checkpoint_every_n_epochs=(
            instance_checkpoint_every_n_epochs
        ),
        instance_checkpoint_max_images=instance_checkpoint_max_images,
        instance_early_stopping_patience=(
            instance_early_stopping_patience
        ),
        transfer_manifest=transfer_manifest,
    )

    initial_weights_path = run_output_dir / (
        "initial_v5_transfer.weights.h5"
        if transfer
        else "initial_random.weights.h5"
    )
    if not resume_available and (transfer or not fine_tune):
        model.save_weights(initial_weights_path)
        print(
            "Initial transfer weights:"
            if transfer
            else "Initial random weights:",
            initial_weights_path,
        )
    elif resume_available:
        print(
            "BackupAndRestore will replace the newly constructed/loaded state "
            "when model.fit() begins."
        )

    # Run the complete validation loss/metric contract once before a long fit.
    validation_batch = val_data[0]
    if not isinstance(validation_batch, (tuple, list)):
        raise TypeError(
            "Validation Sequence must return (inputs, targets) or "
            "(inputs, targets, sample_weight)."
        )
    if len(validation_batch) == 2:
        preflight_inputs, preflight_targets = validation_batch
        preflight_sample_weight = None
    elif len(validation_batch) == 3:
        (
            preflight_inputs,
            preflight_targets,
            preflight_sample_weight,
        ) = validation_batch
    else:
        raise ValueError(
            "Validation Sequence returned an unsupported tuple length: "
            f"{len(validation_batch)}."
        )

    preflight_logs = model.test_on_batch(
        preflight_inputs,
        preflight_targets,
        sample_weight=preflight_sample_weight,
        return_dict=True,
    )
    invalid_preflight = {
        str(name): _json_compatible(value)
        for name, value in preflight_logs.items()
        if not np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
    }
    if invalid_preflight:
        raise FloatingPointError(
            "Preflight validation produced non-finite results: "
            f"{invalid_preflight}."
        )
    model.reset_metrics()
    print(
        "Preflight validation passed:",
        json.dumps(_json_compatible(preflight_logs), indent=2),
    )
    del preflight_inputs, preflight_targets, validation_batch

    previous_best_semantic: float | None = None
    if (
        resume_available
        and best_semantic_path.is_file()
        and training_log_path.is_file()
    ):
        # Width-aware, for the reason documented on _read_training_log: a
        # positional DictReader drops this column entirely on narrow rows, so
        # the recovered "best" would be the best of only the evaluating epochs
        # and the checkpoint threshold would be set too low.
        for row in _read_training_log(training_log_path):
            value = row.get("val_semantic_foreground_miou")
            if value is None or not np.isfinite(value):
                continue
            previous_best_semantic = (
                value
                if previous_best_semantic is None
                else max(previous_best_semantic, value)
            )
        if previous_best_semantic is not None:
            print(
                "Resume-safe previous best foreground mIoU:",
                f"{previous_best_semantic:.6f}",
            )

    callbacks: list[tf.keras.callbacks.Callback] = [
        AugmentationEpochSyncCallback(train_data),
    ]
    if transfer:
        callbacks.append(ProgressiveTransferScheduleCallback())
    callbacks.extend(
        [
            tf.keras.callbacks.LearningRateScheduler(
                learning_rate_schedule,
                verbose=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
    )

    if USE_EMA:
        if not hasattr(tf.keras.callbacks, "SwapEMAWeights"):
            raise RuntimeError(
                "USE_EMA=True, but SwapEMAWeights is unavailable in this "
                "TensorFlow/Keras version."
            )
        # Keras swaps to EMA weights for validation and keeps them swapped at
        # epoch end so all later preview/checkpoint callbacks use the EMA model.
        callbacks.append(
            tf.keras.callbacks.SwapEMAWeights(swap_on_epoch=True)
        )

    semantic_checkpoint_kwargs: dict[str, object] = {
        "filepath": best_semantic_path,
        "monitor": "val_semantic_foreground_miou",
        "mode": "max",
        "save_best_only": True,
        "verbose": 1,
    }
    if previous_best_semantic is not None:
        checkpoint_parameters = inspect.signature(
            tf.keras.callbacks.ModelCheckpoint
        ).parameters
        if "initial_value_threshold" in checkpoint_parameters:
            semantic_checkpoint_kwargs["initial_value_threshold"] = (
                previous_best_semantic
            )
        else:
            print(
                "WARNING: ModelCheckpoint cannot restore its previous best "
                "threshold in this Keras version."
            )

    early_stopping_kwargs: dict[str, object] | None = None
    if not fine_tune and not transfer:
        early_stopping_kwargs = {
            "monitor": "val_semantic_foreground_miou",
            "mode": "max",
            "patience": patience,
            "min_delta": 1e-4,
            "start_from_epoch": effective_early_stopping_start,
            "restore_best_weights": False,
            "verbose": 1,
        }
        if previous_best_semantic is not None:
            early_stopping_kwargs["baseline"] = previous_best_semantic

    backup_kwargs: dict[str, object] = {
        "backup_dir": backup_dir,
        "save_freq": "epoch",
        "delete_checkpoint": True,
    }
    backup_parameters = inspect.signature(
        tf.keras.callbacks.BackupAndRestore
    ).parameters
    if "double_checkpoint" in backup_parameters:
        backup_kwargs["double_checkpoint"] = True

    # When EMA is enabled these callbacks intentionally come after
    # SwapEMAWeights, as required for EMA previews and EMA checkpoints.
    selection_callbacks: list[tf.keras.callbacks.Callback] = [
        _NonFatalPreviewCallback(val_arrays, preview_dir),
        InstanceF1Checkpoint(
            val_arrays,
            best_instance_path,
            instance_history_path,
            every_n_epochs=instance_checkpoint_every_n_epochs,
            maximum_images=instance_checkpoint_max_images,
            early_stopping_patience_evaluations=(
                instance_early_stopping_patience
            ),
            early_stopping_start_epoch=(
                effective_early_stopping_start
            ),
        ),
        tf.keras.callbacks.ModelCheckpoint(
            **semantic_checkpoint_kwargs
        ),
        TrainingReportCallback(run_output_dir),
    ]
    if early_stopping_kwargs is not None:
        selection_callbacks.append(
            tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs)
        )

    callbacks.extend(
        selection_callbacks
        + [
            tf.keras.callbacks.CSVLogger(
                training_log_path,
                append=bool(resume_available and training_log_path.is_file()),
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=0,
                write_steps_per_second=True,
                profile_batch=0,
                # An epoch here is over half an hour, so the default
                # update_freq="epoch" leaves TensorBoard empty for that long
                # and a divergence is only visible after it has already
                # happened. Converted from steps to callback invocations,
                # which is the unit Keras actually compares update_freq
                # against; see TENSORBOARD_UPDATE_STEPS.
                update_freq=(
                    max(
                        1,
                        int(TENSORBOARD_UPDATE_STEPS)
                        // max(1, int(STEPS_PER_EXECUTION)),
                    )
                    if int(TENSORBOARD_UPDATE_STEPS) > 0
                    else "epoch"
                ),
            ),
            # Keep backup last: after EMA's epoch-end swap, a resumed epoch's
            # epoch-begin swap restores the raw/EMA variables to training order.
            tf.keras.callbacks.BackupAndRestore(**backup_kwargs),
        ]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(last_path)

    if not best_semantic_path.is_file():
        raise RuntimeError(
            "Semantic checkpoint was not created. History keys: "
            f"{sorted(history.history)}"
        )

    candidate_paths = [best_instance_path, best_semantic_path, last_path]
    if fine_tune and source_model_path is not None:
        # Safety invariant: fine-tuning must beat the source checkpoint on the
        # same complete validation split before it can replace that source.
        candidate_paths.append(source_model_path)
    selected_path, candidate_reports = compare_candidate_models(
        candidate_paths,
        val_arrays,
    )
    selection_report = {
        "selection_rule": (
            "highest complete-validation instance F1; semantic foreground "
            "mIoU, instance recall, and instance precision break ties"
        ),
        "validation_images": validation_count,
        "training_mode": run_label,
        "source_model": (
            str(source_model_path) if source_model_path is not None else None
        ),
        "source_model_is_candidate": bool(
            fine_tune and source_model_path is not None
        ),
        "transfer_manifest": (
            str(run_output_dir / "transfer_manifest.json")
            if transfer
            else None
        ),
        "selected_source": str(selected_path),
        "authoritative_copy": str(final_path),
        "candidates": candidate_reports,
    }
    _atomic_replace_text(
        selection_path,
        json.dumps(
            _json_compatible(selection_report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )

    # Copy the exact winning .keras archive instead of loading with
    # compile=False and re-serializing a different inference-only model.
    selected_resolved = selected_path.resolve()
    final_resolved = final_path.resolve(strict=False)
    if selected_resolved != final_resolved:
        with tempfile.TemporaryDirectory(
            prefix=".final_model_copy_",
            dir=run_output_dir,
        ) as temporary_directory:
            temporary_copy = Path(temporary_directory) / final_path.name
            shutil.copy2(selected_path, temporary_copy)
            if temporary_copy.stat().st_size != selected_path.stat().st_size:
                raise OSError("Final model copy has an unexpected file size.")
            os.replace(temporary_copy, final_path)
    if not final_path.is_file() or final_path.stat().st_size <= 0:
        raise OSError(f"Authoritative final model was not created: {final_path}")

    if RUN_FINAL_EVALUATION_AFTER_TRAINING:
        selected_model = tf.keras.models.load_model(final_path, compile=False)
        validate_model_output_shapes(selected_model)
        evaluated_splits: set[str] = set()
        try:
            for raw_split in FINAL_EVALUATION_SPLITS:
                split = str(raw_split).strip().lower()
                if not split or split in evaluated_splits:
                    continue
                evaluated_splits.add(split)

                if split == "train":
                    arrays = train_arrays
                elif split == "val":
                    arrays = val_arrays
                else:
                    try:
                        arrays = load_split_arrays(
                            split,
                            array_dir=run_array_dir,
                        )
                    except FileNotFoundError as error:
                        print(f"Skipping final {split} evaluation:", error)
                        continue

                indices = selected_indices(
                    len(arrays["images"]),
                    EVALUATION_MAX_IMAGES,
                )
                if indices.size == 0:
                    print(f"Skipping empty final {split} evaluation.")
                    continue
                semantic_report, instance_report = evaluate_model_arrays(
                    selected_model,
                    arrays,
                    indices,
                    show_progress=True,
                )
                save_evaluation_reports(
                    split,
                    final_path,
                    semantic_report,
                    instance_report,
                    (
                        run_output_dir
                        / "performance"
                        / f"{split}_final_evaluation"
                    ),
                )
        finally:
            del selected_model
            tf.keras.backend.clear_session()
            gc.collect()

    print("\nTraining complete.")
    print("Best semantic checkpoint:", best_semantic_path)
    print("Best instance checkpoint:", best_instance_path)
    print("Last-epoch checkpoint:", last_path)
    print("Authoritative selected model:", final_path)
    print("Selection report:", selection_path)
    print("Training log:", training_log_path)
    print("Validation previews:", preview_dir)
    print("TensorBoard logs:", tensorboard_dir)

# =============================================================================
# 14. PREDICTION WITH THE SAME LETTERBOX USED FOR TRAINING
# =============================================================================
def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Atomically save a non-pickled NumPy array."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def find_prediction_images(
    source: Path,
    excluded_roots: tuple[Path, ...] | list[Path] = (),
) -> list[Path]:
    """Find deterministic, unique inputs without recursively reading outputs."""
    source = Path(source).expanduser().resolve(strict=False)
    extensions = {
        str(extension).lower()
        for extension in IMAGE_EXTENSIONS
    }
    if not extensions or any(
        not extension.startswith(".") for extension in extensions
    ):
        raise ValueError(
            "IMAGE_EXTENSIONS must contain extensions such as '.png' and '.jpg'."
        )

    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported image type: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Prediction source not found: {source}")

    excluded = tuple(
        Path(root).expanduser().resolve(strict=False)
        for root in excluded_roots
    )

    def inside_excluded_root(path: Path) -> bool:
        resolved = path.resolve(strict=False)
        for root in excluded:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    images: list[Path] = []
    seen_files: set[str] = set()
    for path in source.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in extensions
            or inside_excluded_root(path)
        ):
            continue

        resolved_identity = os.path.normcase(
            str(path.resolve(strict=False))
        )
        if resolved_identity in seen_files:
            continue
        seen_files.add(resolved_identity)
        images.append(path)

    images.sort(
        key=lambda path: (
            path.relative_to(source).as_posix().casefold(),
            path.relative_to(source).as_posix(),
        )
    )
    if not images:
        raise RuntimeError(f"No prediction images found inside: {source}")
    return images


def prediction_subdirectory(
    image_path: Path,
    source: Path,
    root: Path,
) -> Path:
    """Give each source image a deterministic collision-free result folder."""
    import hashlib

    image_path = Path(image_path)
    source = Path(source)
    root = Path(root)

    if source.is_dir():
        try:
            relative = image_path.relative_to(source)
        except ValueError as error:
            raise ValueError(
                f"Image {image_path} is not inside prediction source {source}."
            ) from error
    else:
        if image_path.resolve(strict=False) != source.resolve(strict=False):
            raise ValueError(
                f"Image {image_path} does not match source file {source}."
            )
        relative = Path(image_path.name)

    extension = relative.suffix.lower().lstrip(".") or "image"
    relative_key = relative.as_posix()
    digest = hashlib.sha256(relative_key.encode("utf-8")).hexdigest()[:10]
    folder_name = f"{relative.stem}__{extension}_{digest}"
    return root / relative.parent / folder_name


def predict_one_image(
    model: Model,
    image_path: Path,
    output_dir: Path,
    model_path: Path | None = None,
) -> dict[str, object]:
    """Predict one image, restore native geometry, and save complete results."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a Keras Model; got {type(model)}.")

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    model_path = Path(
        MODEL_FOR_INFERENCE if model_path is None else model_path
    )
    original_rgb = _as_rgb_uint8(
        read_rgb_image(image_path),
        name=f"prediction image {image_path}",
    )
    original_height, original_width = original_rgb.shape[:2]

    letterboxed, metadata = letterbox_rgb(original_rgb)
    letterboxed = _as_rgb_uint8(
        letterboxed,
        name=f"letterboxed image {image_path}",
    )
    expected_shape = (IMG_SIZE, IMG_SIZE, 3)
    if letterboxed.shape != expected_shape:
        raise ValueError(
            f"letterbox_rgb returned {letterboxed.shape}; expected {expected_shape}."
        )
    if not isinstance(metadata, dict):
        raise TypeError(
            f"letterbox_rgb metadata must be a dictionary; got {type(metadata)}."
        )

    normalized = np.asarray(normalize_image(letterboxed), dtype=np.float32)
    if normalized.shape != expected_shape:
        raise ValueError(
            f"normalize_image returned {normalized.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized prediction image contains NaN or infinity.")
    batch = tf.convert_to_tensor(normalized[None, ...], dtype=tf.float32)

    inference_start = time.perf_counter()
    # unpack_model_outputs transfers every output to NumPy, which also
    # synchronizes outstanding GPU work before the timer is stopped.
    outputs = unpack_model_outputs(model(batch, training=False))
    inference_ms = 1000.0 * (time.perf_counter() - inference_start)

    decode_start = time.perf_counter()
    (
        semantic_probabilities,
        inner_distance_small,
        boundary_probability,
    ) = output_probabilities(outputs, 0)
    (
        instance_map,
        class_by_instance,
        _,
        instance_scores,
    ) = decode_instances(
        semantic_probabilities,
        inner_distance_small,
        boundary_probability,
    )
    (
        instance_map,
        class_by_instance,
        instance_scores,
        instances_below_floor,
    ) = filter_instances_by_confidence(
        instance_map,
        class_by_instance,
        semantic_probabilities,
        instance_scores,
        DEPLOYMENT_MIN_CONFIDENCE,
    )
    decoding_ms = 1000.0 * (time.perf_counter() - decode_start)

    semantic_labels = np.argmax(
        semantic_probabilities,
        axis=-1,
    ).astype(np.uint8)
    # V7's per-pixel instance evidence is the inner-distance field, not a centre
    # heatmap. This block still read `center_probabilities_small`, which no
    # longer exists in this function -- a NameError on every real prediction,
    # and the reason the public inference path could not run at all.
    inner_distance_full = resize_channels(
        inner_distance_small,
        IMG_SIZE,
        IMG_SIZE,
    )

    instance_original = restore_map_to_original(
        instance_map,
        metadata,
        cv2.INTER_NEAREST,
    ).astype(INSTANCE_ID_DTYPE, copy=False)
    semantic_original = restore_map_to_original(
        semantic_labels,
        metadata,
        cv2.INTER_NEAREST,
    ).astype(np.uint8, copy=False)
    semantic_probabilities_original = restore_map_to_original(
        semantic_probabilities,
        metadata,
        cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    boundary_original = restore_map_to_original(
        boundary_probability,
        metadata,
        cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    distance_original = restore_map_to_original(
        inner_distance_full,
        metadata,
        cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)

    expected_original_shape = (original_height, original_width)
    if instance_original.shape != expected_original_shape:
        raise ValueError(
            "Restored instance map has the wrong shape: "
            f"{instance_original.shape} versus {expected_original_shape}."
        )
    if semantic_original.shape != expected_original_shape:
        raise ValueError(
            "Restored semantic map has the wrong shape: "
            f"{semantic_original.shape} versus {expected_original_shape}."
        )
    if semantic_probabilities_original.shape != (
        original_height,
        original_width,
        NUM_CLASSES,
    ):
        raise ValueError(
            "Restored semantic probabilities have the wrong shape: "
            f"{semantic_probabilities_original.shape}."
        )
    if boundary_original.shape != expected_original_shape:
        raise ValueError(
            "Restored boundary map has the wrong shape: "
            f"{boundary_original.shape}."
        )
    # restore_map_to_original returns a 2-D map for a single-channel input, so
    # accept either form and settle on 2-D before use.
    if distance_original.ndim == 3 and distance_original.shape[-1] == 1:
        distance_original = distance_original[..., 0]
    if distance_original.shape != expected_original_shape:
        raise ValueError(
            "Restored inner-distance field has the wrong shape: "
            f"{distance_original.shape} versus {expected_original_shape}."
        )

    semantic_probabilities_original = np.clip(
        semantic_probabilities_original,
        0.0,
        1.0,
    )
    probability_sum = semantic_probabilities_original.sum(
        axis=-1,
        keepdims=True,
    )
    semantic_probabilities_original = np.divide(
        semantic_probabilities_original,
        probability_sum,
        out=np.zeros_like(semantic_probabilities_original),
        where=probability_sum > 1e-8,
    )
    empty_probability_pixels = probability_sum[..., 0] <= 1e-8
    if np.any(empty_probability_pixels):
        semantic_probabilities_original[empty_probability_pixels, 0] = 1.0

    boundary_original = np.clip(boundary_original, 0.0, 1.0)
    distance_original = np.clip(distance_original, 0.0, 1.0)

    # A tiny decoded object may disappear when a 512x512 map is restored to a
    # smaller native image. Remove its now-stale class mapping explicitly.
    present_ids = np.unique(instance_original)
    present_ids = present_ids[present_ids > 0]
    present_set = {int(instance_id) for instance_id in present_ids.tolist()}
    decoded_mapping = {
        int(instance_id): int(class_id)
        for instance_id, class_id in class_by_instance.items()
    }
    missing_mapping = sorted(present_set - set(decoded_mapping))
    if missing_mapping:
        raise ValueError(
            "Restored instance map contains IDs without classes: "
            f"{missing_mapping[:16]}."
        )
    original_class_by_instance = {
        instance_id: decoded_mapping[instance_id]
        for instance_id in sorted(present_set)
    }

    instances = summarise_instances(
        instance_original,
        original_class_by_instance,
        semantic_probabilities_original,
        instance_scores,
    )
    overlay = make_instance_overlay(
        original_rgb,
        instance_original,
        instances,
    )
    semantic_confidence_original = np.max(
        semantic_probabilities_original,
        axis=-1,
    ).astype(np.float32)
    foreground_probability = np.clip(
        1.0 - semantic_probabilities_original[..., 0], 0.0, 1.0
    )
    distance_combined_original = (
        distance_original * np.sqrt(foreground_probability)
    ).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "instance_ids": output_dir / "instance_ids.npy",
        "semantic_ids": output_dir / "semantic_ids.npy",
        "semantic_colour": output_dir / "semantic_colour.png",
        "instance_overlay": output_dir / "instances.png",
        "boundary_heatmap": output_dir / "boundary.png",
        "inner_distance": output_dir / "inner_distance.png",
        "semantic_confidence": output_dir / "semantic_confidence.png",
        "metadata": output_dir / "instances.json",
    }
    _atomic_save_npy(artifact_paths["instance_ids"], instance_original)
    _atomic_save_npy(artifact_paths["semantic_ids"], semantic_original)
    write_rgb_image(
        artifact_paths["semantic_colour"],
        colourize_semantic(semantic_original),
    )
    write_rgb_image(artifact_paths["instance_overlay"], overlay)
    write_rgb_image(
        artifact_paths["boundary_heatmap"],
        probability_heatmap(boundary_original),
    )
    write_rgb_image(
        artifact_paths["inner_distance"],
        probability_heatmap(distance_combined_original),
    )
    write_rgb_image(
        artifact_paths["semantic_confidence"],
        probability_heatmap(semantic_confidence_original),
    )

    per_class_counts = {
        str(CLASS_NAMES[class_id]): sum(
            int(instance["class_id"]) == class_id
            for instance in instances
        )
        for class_id in range(1, NUM_CLASSES)
    }
    result: dict[str, object] = {
        "source": str(image_path.resolve(strict=False)),
        "model": str(model_path.resolve(strict=False)),
        "output_directory": str(output_dir.resolve(strict=False)),
        "preprocessing": "shared aspect-ratio-preserving letterbox",
        "original_shape_hwc": [original_height, original_width, 3],
        "model_input_shape_hwc": [IMG_SIZE, IMG_SIZE, 3],
        "letterbox": _json_compatible(metadata),
        "timing_ms": {
            "model_inference_and_device_transfer": float(inference_ms),
            "probability_conversion_and_instance_decoding": float(decoding_ms),
            "combined_compute_excluding_file_io": float(
                inference_ms + decoding_ms
            ),
        },
        # Retained for compatibility with the existing console output.
        "inference_ms_excluding_file_io": float(inference_ms),
        "number_of_instances": len(instances),
        # What the deployment floor removed, so a run is auditable after the
        # fact rather than silently short of detections.
        "confidence_floor": float(DEPLOYMENT_MIN_CONFIDENCE),
        "instances_below_confidence_floor": int(instances_below_floor),
        "instances_per_class": per_class_counts,
        "artifacts": {
            name: str(path.resolve(strict=False))
            for name, path in artifact_paths.items()
            if name != "metadata"
        },
        "instances": instances,
    }
    _atomic_replace_text(
        artifact_paths["metadata"],
        json.dumps(
            _json_compatible(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return result


def predict() -> dict[str, object]:
    """Run robust folder inference and save a complete prediction manifest."""
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if USE_MIXED_PRECISION else "float32"
    )
    configure_runtime()

    model_path = Path(MODEL_FOR_INFERENCE).expanduser().resolve(strict=False)
    source = Path(PREDICT_SOURCE).expanduser().resolve(strict=False)
    output_root = Path(MODEL_OUTPUT_DIR) / "predictions"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Inference model not found: {model_path}\n"
            "Complete training first or change MODEL_FOR_INFERENCE."
        )
    if model_path.suffix.lower() not in {".keras", ".h5"}:
        raise ValueError(
            f"MODEL_FOR_INFERENCE must be a .keras or .h5 model: {model_path}"
        )

    images = find_prediction_images(
        source,
        excluded_roots=(output_root,),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Predicting {len(images)} image(s). Output: {output_root}")

    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    model = None
    run_start = time.perf_counter()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        validate_model_output_shapes(model)

        # Build/tracing and GPU initialization are intentionally excluded from
        # per-image timing, making first-image timing comparable to later images.
        warmup_batch = tf.zeros(
            (1, IMG_SIZE, IMG_SIZE, 3),
            dtype=tf.float32,
        )
        unpack_model_outputs(model(warmup_batch, training=False))
        print("Inference warm-up complete.")

        for index, image_path in enumerate(images, start=1):
            try:
                output_dir = prediction_subdirectory(
                    image_path,
                    source,
                    output_root,
                )
                result = predict_one_image(
                    model,
                    image_path,
                    output_dir,
                    model_path=model_path,
                )
                successes.append(result)
                print(
                    f"[{index}/{len(images)}] {image_path.name}: "
                    f"{result['number_of_instances']} instances, "
                    f"{result['inference_ms_excluding_file_io']:.1f} ms"
                )
            except Exception as error:
                failure = {
                    "image": str(image_path.resolve(strict=False)),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                print(
                    f"[{index}/{len(images)}] FAILED {image_path}: "
                    f"{type(error).__name__}: {error}"
                )
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()

    total_elapsed_seconds = time.perf_counter() - run_start
    aggregate_class_counts = {
        str(CLASS_NAMES[class_id]): sum(
            int(result["instances_per_class"].get(
                str(CLASS_NAMES[class_id]),
                0,
            ))
            for result in successes
        )
        for class_id in range(1, NUM_CLASSES)
    }
    inference_times = np.asarray(
        [
            float(result["inference_ms_excluding_file_io"])
            for result in successes
        ],
        dtype=np.float64,
    )
    manifest: dict[str, object] = {
        "model": str(model_path),
        "source": str(source),
        "output_root": str(output_root.resolve(strict=False)),
        "images_discovered": len(images),
        "images_succeeded": len(successes),
        "images_failed": len(failures),
        "total_instances": sum(
            int(result["number_of_instances"])
            for result in successes
        ),
        "instances_per_class": aggregate_class_counts,
        "elapsed_seconds_including_io": float(total_elapsed_seconds),
        "mean_inference_ms_excluding_io": (
            float(inference_times.mean()) if inference_times.size else None
        ),
        "median_inference_ms_excluding_io": (
            float(np.median(inference_times)) if inference_times.size else None
        ),
        "successful_results": [
            {
                "source": result["source"],
                "output_directory": result["output_directory"],
                "number_of_instances": result["number_of_instances"],
                "instances_per_class": result["instances_per_class"],
                "inference_ms_excluding_file_io": result[
                    "inference_ms_excluding_file_io"
                ],
            }
            for result in successes
        ],
        "failures": failures,
    }
    _atomic_replace_text(
        output_root / "prediction_manifest.json",
        json.dumps(
            _json_compatible(manifest),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )
    _atomic_replace_text(
        output_root / "prediction_failures.json",
        json.dumps(failures, indent=2, sort_keys=True, allow_nan=False),
    )

    print(
        f"Prediction complete: {len(successes)} succeeded, "
        f"{len(failures)} failed."
    )
    print("Prediction manifest:", output_root / "prediction_manifest.json")
    if not successes:
        raise RuntimeError(
            "Every prediction image failed. See prediction_failures.json."
        )
    return manifest

# =============================================================================
# 15. STANDALONE EVALUATION
# =============================================================================
def evaluate(
    split: str = "val",
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one labelled split and save reproducible semantic/instance reports."""
    split = str(split).strip().lower()
    if not split:
        raise ValueError("Evaluation split must not be empty.")
    if any(
        not (character.isalnum() or character in {"_", "-"})
        for character in split
    ):
        raise ValueError(
            "Evaluation split may contain only letters, digits, '_' and '-'; "
            f"received {split!r}."
        )

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if USE_MIXED_PRECISION else "float32"
    )
    configure_runtime()

    model_path = Path(MODEL_FOR_INFERENCE).expanduser().resolve(strict=False)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix.lower() not in {".keras", ".h5"}:
        raise ValueError(
            f"Evaluation model must be a .keras or .h5 file: {model_path}"
        )

    arrays = load_split_arrays(split)
    required_arrays = {"images", "semantic", "instance"}
    missing = required_arrays - set(arrays)
    if missing:
        raise KeyError(
            f"The {split!r} split is missing arrays: {sorted(missing)}."
        )

    total_images = len(arrays["images"])
    if total_images == 0:
        raise ValueError(f"The {split!r} split is empty.")
    for name in ("semantic", "instance"):
        if len(arrays[name]) != total_images:
            raise ValueError(
                f"{split!r} images and {name} arrays have different lengths: "
                f"{total_images} versus {len(arrays[name])}."
            )

    maximum_images = int(EVALUATION_MAX_IMAGES)
    indices = np.asarray(
        selected_indices(total_images, maximum_images),
        dtype=np.int64,
    )
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            f"Evaluation selected no valid {split!r} image indices."
        )
    if np.any(indices < 0) or np.any(indices >= total_images):
        raise IndexError("Evaluation selected an out-of-range image index.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Evaluation selected duplicate image indices.")

    evaluating_complete_split = indices.size == total_images
    selection_policy = (
        "complete split"
        if evaluating_complete_split
        else "deterministic equal-bin subset"
    )
    print(
        f"Evaluating {indices.size:,}/{total_images:,} {split} images "
        f"using {selection_policy}."
    )
    if not evaluating_complete_split:
        print(
            "WARNING: this is a subset evaluation. Set "
            "EVALUATION_MAX_IMAGES = 0 for authoritative full-split metrics."
        )

    class_pixel_counts = semantic_class_pixel_counts(arrays["semantic"])
    if class_pixel_counts.shape != (NUM_CLASSES,):
        raise ValueError(
            "Unexpected semantic class-count shape: "
            f"{class_pixel_counts.shape}; expected ({NUM_CLASSES},)."
        )
    missing_classes = [
        class_id
        for class_id in range(NUM_CLASSES)
        if int(class_pixel_counts[class_id]) == 0
    ]
    if missing_classes:
        print(
            f"WARNING: {split} contains no pixels for class IDs "
            f"{missing_classes}; their per-class metrics are undefined."
        )

    model = None
    evaluation_start = time.perf_counter()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        validate_model_output_shapes(model)

        semantic_report, instance_report = evaluate_model_arrays(
            model,
            arrays,
            indices,
            show_progress=True,
        )
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()

    elapsed_seconds = time.perf_counter() - evaluation_start
    if not isinstance(semantic_report, dict):
        raise TypeError("Semantic evaluation did not return a dictionary.")
    if not isinstance(instance_report, dict):
        raise TypeError("Instance evaluation did not return a dictionary.")

    def validated_unit_metric(value, name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(
                f"Evaluation returned invalid {name}={number}; expected [0,1]."
            )
        return number

    foreground_miou = validated_unit_metric(
        semantic_report["foreground_mean_iou"],
        "foreground mIoU",
    )
    overall_instance = instance_report["overall"]
    instance_precision = validated_unit_metric(
        overall_instance["precision"],
        "instance precision",
    )
    instance_recall = validated_unit_metric(
        overall_instance["recall"],
        "instance recall",
    )
    instance_f1 = validated_unit_metric(
        overall_instance["f1"],
        "instance F1",
    )
    mask_map50 = validated_unit_metric(
        instance_report["mask_map50"],
        "mask mAP50",
    )
    mask_map50_95 = validated_unit_metric(
        instance_report["mask_map50_95"],
        "mask mAP50-95",
    )

    evaluation_context: dict[str, object] = {
        "dataset_images": total_images,
        "images_evaluated": int(indices.size),
        "selection_policy": selection_policy,
        "evaluation_max_images_setting": maximum_images,
        "evaluated_indices": indices.tolist(),
        "complete_split": bool(evaluating_complete_split),
        "class_pixel_counts_complete_split": class_pixel_counts.tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "images_per_second": float(
            indices.size / max(elapsed_seconds, 1e-12)
        ),
        "mixed_precision_policy": (
            tf.keras.mixed_precision.global_policy().name
        ),
    }

    semantic_report = dict(semantic_report)
    instance_report = dict(instance_report)
    semantic_report["evaluation_context"] = evaluation_context
    instance_report["evaluation_context"] = evaluation_context

    output_dir = (
        Path(MODEL_OUTPUT_DIR)
        / "performance"
        / f"{split}_evaluation"
    )
    save_evaluation_reports(
        split,
        model_path,
        semantic_report,
        instance_report,
        output_dir,
    )

    summary = {
        "split": split,
        "model": str(model_path),
        "output_directory": str(output_dir.resolve(strict=False)),
        "evaluation_context": evaluation_context,
        "headline_metrics": {
            "semantic_foreground_miou": foreground_miou,
            "instance_precision": instance_precision,
            "instance_recall": instance_recall,
            "instance_f1": instance_f1,
            "mask_map50": mask_map50,
            "mask_map50_95": mask_map50_95,
        },
    }
    _atomic_replace_text(
        output_dir / f"{split}_evaluation_summary.json",
        json.dumps(
            _json_compatible(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )

    print(f"\n{split.upper()} EVALUATION COMPLETE")
    print(f"Semantic foreground mIoU: {foreground_miou:.4f}")
    print(f"Instance precision:       {instance_precision:.4f}")
    print(f"Instance recall:          {instance_recall:.4f}")
    print(f"Instance F1:              {instance_f1:.4f}")
    print(f"Mask mAP50:               {mask_map50:.4f}")
    print(f"Mask mAP50-95:            {mask_map50_95:.4f}")
    print("Evaluation output:", output_dir)
    return semantic_report, instance_report



# =============================================================================
# 16. SELF-TEST
# =============================================================================

def _selftest_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one synthetic image with three well-separated known objects."""
    image = np.full(
        (IMG_SIZE, IMG_SIZE, 3),
        float(LETTERBOX_FILL_VALUE) / 255.0,
        dtype=np.float32,
    )
    semantic = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    instance = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

    semantic[60:130, 60:170] = 1
    instance[60:130, 60:170] = 1

    semantic[300:400, 320:460] = 1
    instance[300:400, 320:460] = 2

    # Two objects of the same class sharing an edge. Nothing but the
    # boundary head can separate these, which is the entire reason the
    # decoder is more than connected components.
    semantic[200:250, 60:110] = 4
    instance[200:250, 60:110] = 4
    semantic[200:250, 110:160] = 4
    instance[200:250, 110:160] = 5

    disc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(disc, (150, 380), 45, 1, -1)
    semantic[disc > 0] = 3
    instance[disc > 0] = 3

    # An asymmetric intensity ramp so every dihedral transform is distinct.
    ramp = np.linspace(0.0, 1.0, IMG_SIZE, dtype=np.float32)
    image[..., 0] = ramp[None, :]
    image[..., 1] = ramp[:, None]
    return image, semantic, instance


def _selftest_probabilities(
    semantic: np.ndarray,
) -> np.ndarray:
    """Turn a hard semantic map into a plausible softmax output."""
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[semantic]
    probabilities = 0.02 + 0.92 * one_hot
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities.astype(np.float32)


def _selftest_best_iou(
    predicted_map: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[float, int]:
    """Return the best IoU against any single predicted instance."""
    best_iou = 0.0
    best_id = 0
    for predicted_id in np.unique(predicted_map):
        predicted_id = int(predicted_id)
        if predicted_id == 0:
            continue
        predicted_mask = predicted_map == predicted_id
        union = int(np.count_nonzero(predicted_mask | target_mask))
        if union == 0:
            continue
        iou = float(
            np.count_nonzero(predicted_mask & target_mask) / union
        )
        if iou > best_iou:
            best_iou = iou
            best_id = predicted_id
    return best_iou, best_id


def run_selftest() -> None:
    """Check the decode, target, ranking and augmentation contracts.

    This is the oracle ablation in miniature and needs no dataset, GPU or
    checkpoint: ground-truth targets are fed straight into the decoder, which
    must hand back the objects it was given. A failure here means the decoder
    is broken by construction and no amount of training can compensate.
    """
    validate_spatial_configuration()
    validate_augmentation_configuration()

    image, semantic, instance = _selftest_scene()
    target_ids = [1, 2, 3, 4, 5]
    target_classes = {1: 1, 2: 1, 3: 3, 4: 4, 5: 4}

    # Instances 4 and 5 must be adjacent for the boundary check to mean
    # anything: if a gap ever creeps in, connected components would separate
    # them and the test would silently stop exercising the boundary head.
    assert np.any(
        (instance[:, :-1] == 4) & (instance[:, 1:] == 5)
    ), "the touching-object pair is no longer touching"
    assert (
        np.count_nonzero((instance == 4) | (instance == 5))
        == np.count_nonzero(
            cv2.connectedComponents(
                ((instance == 4) | (instance == 5)).astype(np.uint8)
            )[1]
        )
    ), "the touching pair must form a single connected component"

    # --- 1. targets round-trip through the decoder ---------------------------
    (
        semantic_target,
        distance_target,
        boundary_target,
    ) = build_all_targets(semantic, instance)

    assert distance_target.shape == (
        INSTANCE_HEAD_SIZE,
        INSTANCE_HEAD_SIZE,
        2,
    ), distance_target.shape
    # Every instance is normalised by its own maximum, so the field
    # reaches exactly 1.0 somewhere and never exceeds it. A max below 1
    # would mean the normalisation is not per-instance, which is the
    # single property the whole grouping design rests on.
    assert np.isclose(
        float(distance_target[..., 0].max()), 1.0
    ), float(distance_target[..., 0].max())
    assert float(distance_target[..., 0].min()) >= 0.0
    assert set(np.unique(distance_target[..., 1]).tolist()) <= {0.0, 1.0}

    (
        decoded_map,
        decoded_classes,
        _,
        decoded_scores,
    ) = decode_instances(
        _selftest_probabilities(semantic_target),
        distance_target[..., 0:1],
        boundary_target[..., 0],
    )

    assert len(decoded_classes) == len(target_ids), (
        f"decoded {len(decoded_classes)} instances from ground-truth "
        f"targets, expected {len(target_ids)}"
    )
    for target_id in target_ids:
        target_mask = instance == target_id
        iou, decoded_id = _selftest_best_iou(decoded_map, target_mask)
        assert iou >= 0.95, (
            f"ground-truth instance {target_id} decoded at IoU {iou:.3f}; "
            "the decoder cannot reproduce its own targets"
        )
        assert decoded_classes[decoded_id] == target_classes[target_id], (
            f"instance {target_id} decoded as class "
            f"{decoded_classes[decoded_id]}, expected "
            f"{target_classes[target_id]}"
        )
        # There is no fallback path left to rank against: every pixel
        # reaches a core, so nothing is minted from residue. The
        # replacement contract is that a perfectly decoded object
        # carries a high peak distance, which is what ranking now uses.
        assert decoded_scores[decoded_id] >= float(
            INNER_DISTANCE_CORE_THRESHOLD
        ), (
            f"instance {target_id} scored "
            f"{decoded_scores[decoded_id]:.4f}, below the core "
            "threshold, so a perfect detection would rank as a rim "
            "fragment"
        )

    # --- 2. ranking separates a centre-backed instance from a fallback -------
    summaries = summarise_instances(
        decoded_map,
        decoded_classes,
        _selftest_probabilities(semantic_target),
        decoded_scores,
    )
    assert len(summaries) == len(target_ids), len(summaries)

    fallback_scores = dict.fromkeys(
        decoded_classes, float(FALLBACK_INSTANCE_SCORE)
    )
    fallback_summaries = summarise_instances(
        decoded_map,
        decoded_classes,
        _selftest_probabilities(semantic_target),
        fallback_scores,
    )
    strongest_fallback = max(
        float(summary["confidence"]) for summary in fallback_summaries
    )
    weakest_detection = min(
        float(summary["confidence"]) for summary in summaries
    )
    assert weakest_detection > strongest_fallback, (
        f"weakest centre-backed detection {weakest_detection:.4f} does not "
        f"outrank the strongest fallback {strongest_fallback:.4f}; mask AP "
        "cannot order detections"
    )

    # --- 3. rim fragments do not become instances ----------------------------
    canvas = np.zeros((64, 64), dtype=bool)
    canvas[10:40, 10:40] = True  # 900 px object
    canvas[56:60, 56:60] = True  # 16 px rim fragment, above MIN_INSTANCE_AREA
    assert int(canvas[56:60, 56:60].sum()) > int(MIN_INSTANCE_AREA), (
        "the fragment must clear MIN_INSTANCE_AREA or this proves nothing"
    )
    fragment_map = np.zeros((64, 64), dtype=INSTANCE_ID_DTYPE)
    fragment_classes: dict[int, int] = {}
    fragment_scores: dict[int, float] = {}
    add_partitioned_instances(
        fragment_map,
        fragment_classes,
        fragment_scores,
        canvas,
        np.zeros((64, 64), dtype=np.float32),
        1,
        1,
        center_score=0.8,
    )
    assert len(fragment_classes) == 1, (
        f"a 900 px object with one 16 px fragment decoded into "
        f"{len(fragment_classes)} instances"
    )

    # --- 3b. the boundary ridge splits a merged candidate --------------------
    # With good centres the offsets already separate instances 4 and 5, so
    # boundary_partition only carries weight when two centres merge. Drive it
    # directly rather than leaving the split path untested.
    merged_candidate = (instance == 4) | (instance == 5)
    boundary_parts = boundary_partition(
        merged_candidate, boundary_target[..., 0]
    )
    assert len(boundary_parts) == 2, (
        f"the boundary ridge split a two-object candidate into "
        f"{len(boundary_parts)} parts, expected 2"
    )
    recovered = np.zeros_like(merged_candidate)
    for part in boundary_parts:
        assert not np.any(recovered & part), "boundary parts overlap"
        recovered |= part
    assert np.array_equal(recovered, merged_candidate), (
        "the boundary split lost or invented pixels"
    )
    for target_id in (4, 5):
        best = max(
            float(
                np.count_nonzero(part & (instance == target_id))
                / max(np.count_nonzero(part | (instance == target_id)), 1)
            )
            for part in boundary_parts
        )
        assert best >= 0.90, (
            f"boundary split matched instance {target_id} at IoU "
            f"{best:.3f}"
        )

    # --- 4. dihedral transforms are exact and label-preserving ---------------
    identity = apply_dihedral_transform(image, semantic, instance, 0)
    assert np.array_equal(identity[1], semantic)
    assert np.array_equal(identity[2], instance)

    rotated_image = image
    rotated_semantic = semantic
    rotated_instance = instance
    for _ in range(4):
        (
            rotated_image,
            rotated_semantic,
            rotated_instance,
        ) = apply_dihedral_transform(
            rotated_image, rotated_semantic, rotated_instance, 1
        )
    assert np.array_equal(rotated_image, image), (
        "four 90-degree rotations are not the identity"
    )
    assert np.array_equal(rotated_semantic, semantic)
    assert np.array_equal(rotated_instance, instance)

    seen: list[bytes] = []
    original_area = int(np.count_nonzero(instance))
    original_ids = set(np.unique(instance).tolist())
    for transform_index in range(8):
        (
            transformed_image,
            transformed_semantic,
            transformed_instance,
        ) = apply_dihedral_transform(
            image, semantic, instance, transform_index
        )
        assert (
            int(np.count_nonzero(transformed_instance)) == original_area
        ), transform_index
        assert (
            set(np.unique(transformed_instance).tolist()) == original_ids
        ), transform_index
        assert transformed_image.shape == image.shape
        seen.append(transformed_image.tobytes())

        # Targets rebuilt from a transformed instance map stay consistent.
        (
            _,
            transformed_distance,
            _,
        ) = build_all_targets(transformed_semantic, transformed_instance)
        # One peak per instance, wherever the transform put it. Counting
        # peaks rather than comparing rasters is what makes this a check
        # on label preservation instead of on interpolation.
        peak_count = int(np.count_nonzero(
            transformed_distance[..., 0] >= 1.0 - 1e-4
        ))
        assert peak_count >= len(target_ids), (
            transform_index, peak_count, len(target_ids)
        )
        assert np.all(np.isfinite(transformed_distance))
        # The field must stay inside its declared range under every transform.
        # A rotation that produced a value above 1.0 would mean the per-instance
        # normalisation had been applied before the transform rather than after,
        # and the core threshold would then mean something different per sample.
        assert 0.0 <= float(transformed_distance[..., 0].min())
        assert float(transformed_distance[..., 0].max()) <= 1.0 + 1e-6

    assert len(set(seen)) == 8, (
        f"the eight dihedral transforms produced {len(set(seen))} distinct "
        "images; they are not a faithful group action"
    )

    # --- 5. zoom-out actually shrinks the content ----------------------------
    zoom_in_matrix = zoom_to_fill_affine_matrix(
        IMG_SIZE,
        IMG_SIZE,
        angle_degrees=30.0,
        extra_zoom=1.0,
        allow_translation=False,
        rng=np.random.default_rng(0),
        fill_frame=True,
    )
    zoom_out_matrix = zoom_to_fill_affine_matrix(
        IMG_SIZE,
        IMG_SIZE,
        angle_degrees=30.0,
        extra_zoom=float(ZOOM_OUT_RANGE[0]),
        allow_translation=False,
        rng=np.random.default_rng(0),
        fill_frame=False,
    )
    zoom_in_scale = float(
        np.hypot(zoom_in_matrix[0, 0], zoom_in_matrix[0, 1])
    )
    zoom_out_scale = float(
        np.hypot(zoom_out_matrix[0, 0], zoom_out_matrix[0, 1])
    )
    assert zoom_in_scale >= 1.0, zoom_in_scale
    assert zoom_out_scale < 1.0, (
        f"zoom-out scale {zoom_out_scale:.4f} does not shrink the content"
    )

    zoomed_image, zoomed_semantic, zoomed_instance = (
        warp_image_semantic_and_instances(
            image, semantic, instance, zoom_out_matrix,
            fill_value=ZOOM_OUT_FILL_VALUE,
        )
    )
    assert int(np.count_nonzero(zoomed_instance)) < original_area, (
        "zoom-out did not reduce the labelled area"
    )
    assert np.all(zoomed_semantic[zoomed_instance == 0] == 0)
    # The uncovered corners must carry the zoom-out fill, not the letterbox
    # grey and not a replicated border pixel.
    expected_fill = float(ZOOM_OUT_FILL_VALUE) / 255.0
    for corner in (
        zoomed_image[0, 0], zoomed_image[0, -1],
        zoomed_image[-1, 0], zoomed_image[-1, -1],
    ):
        assert np.allclose(corner, expected_fill, atol=1e-6), (
            f"zoom-out padded with {corner} instead of {expected_fill}"
        )

    # Mirror-side: the output is only the mirror image -- pixels and labels
    # flipped together inside the board, same objects, same ids, bars intact.
    source_ids = np.unique(instance[instance > 0])
    for side in MIRROR_SIDES:
        mirror_image, mirror_semantic, mirror_instance = mirror_side_sample(
            image, semantic, instance, side
        )
        axis = 1 if side == "left" else 0
        top, bottom, left, right = letterboxed_board_bounds(image)
        window = (slice(top, bottom), slice(left, right))
        assert np.array_equal(
            mirror_image[window], np.flip(image[window], axis)
        ), f"mirror {side}: image is not the mirror of the board"
        assert np.array_equal(
            mirror_semantic[window], np.flip(semantic[window], axis)
        ), f"mirror {side}: class labels were not mirrored with the pixels"
        assert np.array_equal(
            mirror_instance[window], np.flip(instance[window], axis)
        ), f"mirror {side}: instance labels were not mirrored with the pixels"
        assert np.array_equal(
            np.unique(mirror_instance[mirror_instance > 0]), source_ids
        ), f"mirror {side}: objects were added or lost"

    # The ladder must be the requested one: 5% steps from 5% to 50% out.
    assert ZOOM_OUT_STEPS == tuple(
        round(1.0 - p / 100.0, 4) for p in range(5, 51, 5)
    ), ZOOM_OUT_STEPS
    assert len(ZOOM_OUT_STEPS) == 10, ZOOM_OUT_STEPS
    assert min(ZOOM_OUT_STEPS) == 0.5 and max(ZOOM_OUT_STEPS) == 0.95

    # The zoom-in ladder, and that it actually reaches the scale real boards
    # contain. Without the second assertion a silently narrowed range would
    # look healthy while reintroducing the exact defect it was added for.
    assert ZOOM_IN_STEPS == tuple(
        round(1.0 + p / 100.0, 4) for p in range(15, 151, 15)
    ), ZOOM_IN_STEPS
    assert len(ZOOM_IN_STEPS) == 10, ZOOM_IN_STEPS
    assert min(ZOOM_IN_STEPS) == 1.15 and max(ZOOM_IN_STEPS) == 2.5
    # 35.7 px is the measured train median for Rectangle and ~1.5 the
    # zoom-to-fill factor for a 3:2 source; 60 px was the old ceiling that
    # 30.1% of real Rectangles exceeded.
    assert 35.7 * 1.5 * max(ZOOM_IN_STEPS) > 60.0, (
        "zoom-in ladder no longer clears the ceiling it was added to clear"
    )

    # Zoom-in and zoom-out are mutually exclusive per sample, so their
    # probabilities must leave room for unzoomed samples as well.
    assert 0.0 <= ZOOM_IN_PROBABILITY <= 1.0, ZOOM_IN_PROBABILITY
    assert ZOOM_OUT_PROBABILITY + (
        1.0 - ZOOM_OUT_PROBABILITY
    ) * ZOOM_IN_PROBABILITY < 1.0, (
        "every realistic variant would be zoomed; none would be seen at "
        "labelled scale"
    )

    # --- 6. the batch loader produces usable batches in every mode ----------
    sample_count = 4
    arrays = {
        "images": np.repeat(
            (image * 255.0).astype(np.uint8)[None], sample_count, axis=0
        ),
        "semantic": np.repeat(
            semantic.astype(np.uint8)[None], sample_count, axis=0
        ),
        "instance": np.repeat(
            instance.astype(INSTANCE_ID_DTYPE)[None], sample_count, axis=0
        ),
    }
    sequence = InstanceArraySequence(
        arrays, batch_size=2, training=True, shuffle=False
    )

    modes_seen: set[int] = set()
    # Epoch 0 gives the original mode, 60 lands inside the realistic
    # variants, which is the only place the zoom-out branch can fire.
    for epoch_index in (0, 1, 46, 60, 94):
        sequence.epoch_index = epoch_index
        batch_images, batch_targets = sequence[0]
        modes_seen.add(
            sequence.augmentation_mode_for_sample(0)
        )

        assert batch_images.shape == (2, IMG_SIZE, IMG_SIZE, 3), (
            batch_images.shape
        )
        assert batch_images.dtype == np.float32
        assert np.all(np.isfinite(batch_images))
        assert 0.0 <= float(batch_images.min())
        assert float(batch_images.max()) <= 1.0

        expected_target_shapes = {
            "semantic": (2, IMG_SIZE, IMG_SIZE),
            "inner_distance": (
                2,
                INSTANCE_HEAD_SIZE,
                INSTANCE_HEAD_SIZE,
                2,
            ),
            "boundary": (2, IMG_SIZE, IMG_SIZE, 1),
        }
        assert set(batch_targets) == set(expected_target_shapes), sorted(
            batch_targets
        )
        for name, expected_shape in expected_target_shapes.items():
            assert batch_targets[name].shape == expected_shape, (
                name,
                batch_targets[name].shape,
            )
            assert np.all(np.isfinite(batch_targets[name])), name

        # The validity mask must mark exactly the supervised pixels, and the
        # distance must be zero wherever it does not. An unmasked non-zero
        # would be a target the loss ignores but the decoder would still
        # threshold, which is how a phantom core gets into background.
        distances = batch_targets["inner_distance"]
        mask = distances[..., 1]
        assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
        assert np.all(distances[..., 0][mask == 0.0] == 0.0)
        assert float(mask.sum()) > 0.0, "no supervised distance pixels"
        assert float(distances[..., 0].min()) >= 0.0
        assert float(distances[..., 0].max()) <= 1.0 + 1e-6

        semantic_batch = batch_targets["semantic"]
        assert semantic_batch.min() >= 0
        assert semantic_batch.max() < NUM_CLASSES

    assert len(modes_seen) == 5, sorted(modes_seen)

    # The dihedral stream must actually reach the batch, not merely exist.
    # Epoch 0 gives sample 0 the original mode, so anything that differs from
    # the stored image at that point can only have come from the dihedral
    # transform -- provided copy-paste, which was added to __getitem__ after
    # this check was written and also rewrites pixels, is held off for the one
    # batch being compared. Without this the check fails on every run and says
    # the dihedral stream is broken when it is not. Mirror-side
    # also rewrites pixels, so it is held off for the same batch.
    global USE_COPY_PASTE_AUGMENTATION, USE_MIRROR_SIDE_AUGMENTATION
    copy_paste_was_enabled = USE_COPY_PASTE_AUGMENTATION
    mirror_side_was_enabled = USE_MIRROR_SIDE_AUGMENTATION
    sequence.epoch_index = 0
    assert sequence.augmentation_mode_for_sample(0) == 0
    USE_COPY_PASTE_AUGMENTATION = False
    USE_MIRROR_SIDE_AUGMENTATION = False
    try:
        batch_images, _ = sequence[0]
    finally:
        USE_COPY_PASTE_AUGMENTATION = copy_paste_was_enabled
        USE_MIRROR_SIDE_AUGMENTATION = mirror_side_was_enabled
    raw_image = normalize_image(arrays["images"][0])

    if USE_DIHEDRAL_AUGMENTATION:
        transform_index = int(
            sequence.augmentation_rng_for_sample(0, salt=1).integers(8)
        )
        expected_image, _, _ = apply_dihedral_transform(
            raw_image, semantic, instance, transform_index
        )
        assert np.allclose(batch_images[0], expected_image, atol=1e-6), (
            "the dihedral stream is not wired into the batch loader"
        )
        # And the draw must vary with the sample, or it is a constant.
        drawn = {
            int(
                sequence.augmentation_rng_for_sample(
                    index, salt=1
                ).integers(8)
            )
            for index in range(64)
        }
        assert len(drawn) >= 6, (
            f"the dihedral draw only produced {sorted(drawn)} over 64 "
            "samples; the stream is not varying"
        )
    else:
        assert np.allclose(batch_images[0], raw_image, atol=1e-6)

    # --- 6b. copy-paste adds whole, single-class, target-able instances -----
    paste_rng = np.random.default_rng(11)
    host_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    host_semantic = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    host_instance = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    host_image[10:60, 10:60] = 0.5
    host_semantic[10:60, 10:60] = 1
    host_instance[10:60, 10:60] = 1

    donor_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    donor_semantic = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    donor_instance = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    donor_image[100:140, 200:280] = 1.0
    donor_semantic[100:140, 200:280] = 2
    donor_instance[100:140, 200:280] = 7
    donor_image[300:330, 300:330] = 0.25
    donor_semantic[300:330, 300:330] = 3
    donor_instance[300:330, 300:330] = 9

    pasted_image, pasted_semantic, pasted_instance = copy_paste_objects(
        host_image,
        host_semantic,
        host_instance,
        donor_image,
        donor_semantic,
        donor_instance,
        paste_rng,
        max_objects=2,
    )
    assert int(host_instance.max()) == 1 and int(host_semantic.max()) == 1, (
        "copy_paste_objects mutated its inputs; the memmap-backed caller "
        "would corrupt the dataset"
    )
    pasted_ids = [
        int(value) for value in np.unique(pasted_instance) if int(value) != 0
    ]
    assert len(pasted_ids) >= 2, (
        f"copy-paste produced no new instance: ids {pasted_ids}"
    )
    for pasted_id in pasted_ids:
        pasted_classes = np.unique(pasted_semantic[pasted_instance == pasted_id])
        assert pasted_classes.size == 1 and int(pasted_classes[0]) > 0, (
            f"instance {pasted_id} is not a single foreground class: "
            f"{pasted_classes.tolist()}"
        )
    _, paste_distance, _ = build_all_targets(
        pasted_semantic, pasted_instance
    )
    assert int((paste_distance[..., 0] > 0.99).sum()) >= len(pasted_ids), (
        "pasted instances did not each produce an inner-distance peak"
    )

    # --- 7. the model actually builds -------------------------------------
    # Keras rejects duplicate operation names, and C3k2(x, ..., "foo") creates
    # a layer called "foo_concat" internally. A skip connection named
    # "foo_concat" beside it therefore makes the whole model unbuildable, and
    # nothing before this point would notice.
    import collections as _collections

    model = build_model_v7_instance()
    layer_names = [layer.name for layer in model.layers]
    duplicates = sorted(
        name
        for name, count in _collections.Counter(layer_names).items()
        if count > 1
    )
    assert not duplicates, f"duplicate layer names: {duplicates}"
    validate_model_output_shapes(model)
    parameter_count = int(model.count_params())
    assert parameter_count > 0
    del model

    # --- 8. the progressive-transfer freeze contract actually holds --------
    # A frozen group is frozen by scaling its gradients to zero inside
    # apply_gradients. If Keras ever stops routing through apply_gradients, or
    # the optimizer fails to construct, the "frozen" V5 layers train from
    # epoch 0 and the logs look identical either way. Prove it on a two-layer
    # stand-in rather than trusting that it still works.
    probe_input = tf.keras.Input((4,))
    probe_frozen = tf.keras.layers.Dense(8, name="sppf_reduce_conv")(
        probe_input
    )
    probe_head = tf.keras.layers.Dense(1, name="new_head")(probe_frozen)
    probe = tf.keras.Model(probe_input, probe_head)
    probe_paths = {
        _transfer_variable_path(variable)
        for variable in probe.layers[1].trainable_variables
    }
    probe_optimizer = ProgressiveTransferAdamW(
        transfer_variable_groups={"deep": tuple(probe_paths)},
        initial_transfer_multipliers={"deep": 0.0},
        learning_rate=0.1,
        weight_decay=1e-2,
        gradient_accumulation_steps=2,
    )
    probe_optimizer.exclude_from_weight_decay(
        var_names=["bias"],
        var_list=list(probe.layers[1].trainable_variables),
    )
    probe.compile(optimizer=probe_optimizer, loss="mse")

    probe_rng = np.random.default_rng(0)
    probe_x = probe_rng.standard_normal((64, 4)).astype(np.float32)
    probe_y = probe_rng.standard_normal((64, 1)).astype(np.float32)

    frozen_before = probe.get_layer("sppf_reduce_conv").get_weights()[0].copy()
    head_before = probe.get_layer("new_head").get_weights()[0].copy()
    probe.fit(probe_x, probe_y, epochs=3, batch_size=8, verbose=0)
    assert np.array_equal(
        probe.get_layer("sppf_reduce_conv").get_weights()[0], frozen_before
    ), (
        "a transfer group with multiplier 0.0 still changed; the freeze "
        "contract is not being applied"
    )
    assert not np.array_equal(
        probe.get_layer("new_head").get_weights()[0], head_before
    ), "the untransferred head did not train, so the probe proves nothing"

    probe_optimizer.set_transfer_multipliers({"deep": 1.0})
    unfrozen_before = (
        probe.get_layer("sppf_reduce_conv").get_weights()[0].copy()
    )
    probe.fit(probe_x, probe_y, epochs=3, batch_size=8, verbose=0)
    assert not np.array_equal(
        probe.get_layer("sppf_reduce_conv").get_weights()[0], unfrozen_before
    ), "a transfer group with multiplier 1.0 did not train"
    del probe, probe_optimizer

    print("Self-test passed:")
    print("  decoder reproduces its own targets at IoU >= 0.95")
    print("  centre-backed detections outrank fallbacks")
    print("  rim fragments are rejected")
    print("  edge-adjacent same-class objects are separated")
    print("  the boundary ridge splits a merged candidate losslessly")
    print("  all 8 dihedral transforms are exact and label-preserving")
    print(
        "  zoom-out ladder is 5%-50% in 5% steps and pads with "
        f"{ZOOM_OUT_FILL_VALUE}"
    )
    print(
        "  mirror-side (left/top) flips labels with the pixels"
    )
    print("  the batch loader emits valid four-head targets in every mode")
    print("  copy-paste adds whole single-class instances with centre peaks")
    print(
        f"  the model builds: {len(layer_names)} layers, "
        f"{parameter_count:,} parameters, no duplicate names"
    )
    print("  the progressive-transfer freeze contract holds both ways")


# =============================================================================
# 15. TRAINING REPORTS
# =============================================================================

def _read_training_log(path: Path) -> list[dict[str, float]]:
    """Read the per-epoch CSV log into float rows, skipping unusable ones.

    Rows written before ``InstanceF1Checkpoint.LOG_KEYS`` were always published
    can be narrower than the header: Keras re-derives the CSV column set after
    every resume, so a resume onto a non-evaluating epoch wrote rows without
    those columns under a header that has them. ``csv.DictReader`` maps
    positionally, which would silently shift every value after the gap onto the
    wrong name. Short rows are therefore matched against the header with those
    optional names removed, which is exactly the order Keras wrote them in.
    """
    path = Path(path)
    if not path.is_file():
        return []

    optional = set(InstanceF1Checkpoint.LOG_KEYS)
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
        narrow_header = [name for name in header if name not in optional]

        for values in reader:
            if not values or values[0] == header[0]:
                continue  # a repeated header from an appended resume
            if len(values) == len(header):
                names = header
            elif len(values) == len(narrow_header):
                names = narrow_header
            else:
                continue  # unrecognisable width; guessing would fabricate data
            row: dict[str, float] = {}
            for key, value in zip(names, values):
                if key is None or value in (None, "", "NA"):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    row[key] = number
            if "epoch" in row:
                rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    return rows


def _read_instance_history(path: Path) -> list[dict[str, object]]:
    """Read the instance-evaluation history, tolerating a partial write."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    records = [
        record
        for record in loaded
        if isinstance(record, dict) and "epoch" in record
    ]
    records.sort(key=lambda record: int(record["epoch"]))
    return records


def _trend(values: list[float], window: int = 5) -> float:
    """Least-squares slope per evaluation over the last ``window`` points."""
    tail = [value for value in values[-window:] if np.isfinite(value)]
    if len(tail) < 2:
        return 0.0
    x = np.arange(len(tail), dtype=np.float64)
    y = np.asarray(tail, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def diagnose_training(
    log_rows: list[dict[str, float]],
    history: list[dict[str, object]],
) -> list[str]:
    """Turn the raw logs into the specific things worth acting on.

    Every finding below is a threshold on a logged number, not an opinion.
    An empty list means nothing crossed a threshold, which is good news.
    """
    findings: list[str] = []
    if not log_rows:
        return ["No epochs logged yet."]

    latest = log_rows[-1]

    # 1. Semantic overfitting.
    train_miou = latest.get("semantic_foreground_miou")
    val_miou = latest.get("val_semantic_foreground_miou")
    if train_miou is not None and val_miou is not None:
        gap = train_miou - val_miou
        if gap > 0.05:
            findings.append(
                f"OVERFITTING: foreground mIoU is {train_miou:.3f} on train "
                f"but {val_miou:.3f} on validation, a gap of {gap:.3f}. "
                "More augmentation or more data closes this; more epochs "
                "will not."
            )

    # 2. Per-head train/validation divergence.
    for head in ("center", "offset", "boundary", "semantic"):
        train_loss = latest.get(f"{head}_loss")
        val_loss = latest.get(f"val_{head}_loss")
        if not train_loss or not val_loss or train_loss <= 0.0:
            continue
        ratio = val_loss / train_loss
        if ratio > 3.0:
            findings.append(
                f"The {head} head is memorising: validation loss "
                f"{val_loss:.4f} is {ratio:.1f}x its training loss "
                f"{train_loss:.4f}."
            )

    # 3. Progress on the selection metric.
    scores = [
        float(record["selection_score"])
        for record in history
        if isinstance(record.get("selection_score"), (int, float))
    ]
    if len(scores) >= 3:
        metric_name = str(history[-1].get("selection_metric", "score"))
        slope = _trend(scores)
        best = max(scores)
        best_index = scores.index(best)
        since_best = len(scores) - 1 - best_index
        if slope > 1e-4:
            findings.append(
                f"Still improving: {metric_name} is rising about "
                f"{slope:.4f} per evaluation over the last 5."
            )
        elif since_best >= 3:
            findings.append(
                f"PLATEAU: {metric_name} peaked at {best:.4f} and has not "
                f"improved for {since_best} evaluations. Either the learning "
                "rate floor is reached or the data is the limit."
            )

    # 4. Precision versus recall, which decides what to tune next.
    if history:
        latest_record = history[-1]
        precision = latest_record.get("precision")
        recall = latest_record.get("recall")
        if isinstance(precision, (int, float)) and isinstance(
            recall, (int, float)
        ):
            if recall - precision > 0.05:
                findings.append(
                    f"False positives dominate: precision {precision:.3f} "
                    f"versus recall {recall:.3f}. Raise "
                    "CENTER_CONFIDENCE_THRESHOLD or "
                    "MIN_COMPONENT_AREA_FRACTION; do not add capacity."
                )
            elif precision - recall > 0.05:
                findings.append(
                    f"Missed objects dominate: recall {recall:.3f} versus "
                    f"precision {precision:.3f}. Lower "
                    "CENTER_CONFIDENCE_THRESHOLD or MIN_INSTANCE_AREA."
                )

        # 5. The weakest class, by instance F1.
        per_class = latest_record.get("per_class")
        if isinstance(per_class, dict) and per_class:
            scored = [
                (str(name), float(metrics["f1"]))
                for name, metrics in per_class.items()
                if isinstance(metrics, dict)
                and isinstance(metrics.get("f1"), (int, float))
            ]
            if scored:
                worst_name, worst_f1 = min(scored, key=lambda item: item[1])
                mean_f1 = float(np.mean([value for _, value in scored]))
                if mean_f1 - worst_f1 > 0.10:
                    findings.append(
                        f"Weakest class is {worst_name} at F1 {worst_f1:.3f} "
                        f"against a {mean_f1:.3f} mean. Check how many "
                        "training instances it actually has before "
                        "changing the model."
                    )

    # 6. Optimisation health.
    learning_rate = latest.get("learning_rate")
    if learning_rate is not None and learning_rate <= MIN_LEARNING_RATE * 1.01:
        findings.append(
            f"The learning rate has reached its floor ({learning_rate:.2e}); "
            "remaining epochs will change very little."
        )
    total_loss = latest.get("loss")
    if total_loss is not None and not np.isfinite(total_loss):
        findings.append("TRAINING DIVERGED: the loss is not finite.")

    return findings


def _report_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_training_report(run_output_dir: Path) -> str:
    """Render everything known about a run as readable Markdown."""
    run_output_dir = Path(run_output_dir)
    log_rows = _read_training_log(run_output_dir / "training_log.csv")
    history = _read_instance_history(
        run_output_dir / "instance_checkpoint_history.json"
    )

    lines: list[str] = [
        f"# Training report - {run_output_dir.name}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]

    if not log_rows:
        lines += ["No epochs have been logged yet.", ""]
        return "\n".join(lines)

    latest = log_rows[-1]
    epoch = int(latest["epoch"])
    lines += [
        "## Where the run is",
        "",
        f"- Epochs completed: **{epoch + 1}** of {EPOCHS}",
        f"- Learning rate: {latest.get('learning_rate', float('nan')):.3e}",
        f"- Train loss: {latest.get('loss', float('nan')):.4f}"
        f"  |  validation loss: {latest.get('val_loss', float('nan')):.4f}",
        f"- Foreground mIoU: train "
        f"{latest.get('semantic_foreground_miou', float('nan')):.4f}"
        f"  |  validation "
        f"{latest.get('val_semantic_foreground_miou', float('nan')):.4f}",
        "",
    ]

    if history:
        best = max(
            history,
            key=lambda record: float(record.get("selection_score", -1.0)),
        )
        metric_name = str(best.get("selection_metric", "selection score"))
        lines += [
            "## Best checkpoint so far",
            "",
            f"- Selected on **{metric_name}** = "
            f"**{float(best.get('selection_score', float('nan'))):.4f}** "
            f"at epoch {int(best['epoch'])}",
            f"- Instance precision {float(best.get('precision', 0)):.4f}, "
            f"recall {float(best.get('recall', 0)):.4f}, "
            f"F1 {float(best.get('f1', 0)):.4f} at IoU "
            f"{INSTANCE_EVALUATION_IOU:.2f}",
            f"- Mask mAP50 {float(best.get('mask_map50', 0)):.4f}, "
            f"mAP50-95 {float(best.get('mask_map50_95', 0)):.4f}",
            f"- TP {int(best.get('tp', 0)):,} / "
            f"FP {int(best.get('fp', 0)):,} / "
            f"FN {int(best.get('fn', 0)):,}",
            "",
        ]

        per_class = best.get("per_class")
        per_class_ap = best.get("per_class_ap") or {}
        if isinstance(per_class, dict) and per_class:
            rows = []
            for name, metrics in per_class.items():
                ap = per_class_ap.get(name, {}) or {}
                def fmt(value):
                    return (
                        f"{float(value):.4f}"
                        if isinstance(value, (int, float))
                        else "-"
                    )
                rows.append([
                    str(name),
                    f"{int(metrics.get('tp', 0)):,}",
                    f"{int(metrics.get('fp', 0)):,}",
                    f"{int(metrics.get('fn', 0)):,}",
                    fmt(metrics.get("precision")),
                    fmt(metrics.get("recall")),
                    fmt(metrics.get("f1")),
                    fmt(ap.get("ap50")),
                    fmt(ap.get("ap50_95")),
                ])
            lines += ["### Per class, at the best checkpoint", ""]
            lines += _report_table(
                ["class", "TP", "FP", "FN", "P", "R", "F1", "AP50", "AP50-95"],
                rows,
            )
            lines += [""]

        lines += ["### Evaluation history", ""]
        lines += _report_table(
            ["epoch", "P", "R", "F1", "mAP50", "mAP50-95", "saved"],
            [
                [
                    str(int(record["epoch"])),
                    f"{float(record.get('precision', 0)):.4f}",
                    f"{float(record.get('recall', 0)):.4f}",
                    f"{float(record.get('f1', 0)):.4f}",
                    f"{float(record.get('mask_map50', 0)):.4f}",
                    f"{float(record.get('mask_map50_95', 0)):.4f}",
                    "yes" if record.get("checkpoint_saved") else "",
                ]
                for record in history[-20:]
            ],
        )
        lines += [""]

    findings = diagnose_training(log_rows, history)
    lines += ["## What the numbers say", ""]
    if findings:
        lines += [f"- {finding}" for finding in findings]
    else:
        lines += ["- Nothing has crossed a warning threshold."]
    lines += [""]

    lines += ["## Semantic IoU by class (last epoch)", ""]
    class_rows = []
    for class_id in range(1, NUM_CLASSES):
        suffix = "".join(
            character if character.isalnum() else "_"
            for character in str(CLASS_NAMES[class_id]).lower()
        ).strip("_")
        train_value = latest.get(f"semantic_iou_{suffix}")
        val_value = latest.get(f"val_semantic_iou_{suffix}")
        class_rows.append([
            str(CLASS_NAMES[class_id]),
            f"{train_value:.4f}" if train_value is not None else "-",
            f"{val_value:.4f}" if val_value is not None else "-",
        ])
    lines += _report_table(["class", "train IoU", "val IoU"], class_rows)
    lines += [""]

    return "\n".join(lines)


def write_training_report(
    run_output_dir: Path,
    milestone_epoch: int | None = None,
) -> Path:
    """Write the live report, and snapshot it when a milestone is reached."""
    run_output_dir = Path(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    text = build_training_report(run_output_dir)
    live_path = run_output_dir / "TRAINING_REPORT.md"
    _atomic_replace_text(live_path, text)

    if milestone_epoch is not None:
        milestone_dir = run_output_dir / "reports"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        _atomic_replace_text(
            milestone_dir / f"epoch_{int(milestone_epoch):04d}.md",
            text,
        )
    return live_path


class TrainingReportCallback(tf.keras.callbacks.Callback):
    """Keep TRAINING_REPORT.md current, and snapshot it every N epochs.

    Report generation only reads files that other callbacks already wrote, so
    it costs no GPU time and cannot perturb training. A failure to write the
    report must never take down a run that has been going for days, so every
    error here is caught and printed.
    """

    def __init__(
        self,
        run_output_dir: Path,
        milestone_every_n_epochs: int = MILESTONE_ANALYSIS_EVERY_N_EPOCHS,
    ):
        super().__init__()
        self.run_output_dir = Path(run_output_dir)
        self.milestone_every_n_epochs = int(milestone_every_n_epochs)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        del logs
        epoch_number = int(epoch) + 1
        milestone = (
            epoch_number
            if (
                self.milestone_every_n_epochs > 0
                and epoch_number % self.milestone_every_n_epochs == 0
            )
            else None
        )
        try:
            path = write_training_report(self.run_output_dir, milestone)
        except Exception as error:  # never kill a multi-day run over a report
            print(f"WARNING: could not write the training report: {error}")
            return
        if milestone is not None:
            print(f"\nMilestone analysis at epoch {milestone}:")
            for finding in diagnose_training(
                _read_training_log(self.run_output_dir / "training_log.csv"),
                _read_instance_history(
                    self.run_output_dir / "instance_checkpoint_history.json"
                ),
            ):
                print(f"  - {finding}")
            print(f"  written to {path}\n")


def main() -> None:
    """Validate configuration and dispatch exactly one requested run mode."""
    validate_spatial_configuration()
    mode = str(RUN_MODE).strip().lower()
    allowed_modes = {
        "selftest",
        "report",
        "prepare",
        "preview",
        "train",
        "transfer",
        "fine_tune",
        "predict",
        "evaluate",
    }
    if mode not in allowed_modes:
        raise ValueError(
            f"Unsupported RUN_MODE={RUN_MODE!r}. Choose one of "
            f"{sorted(allowed_modes)}."
        )

    print("RUN_MODE:", mode)
    if mode == "selftest":
        run_selftest()
        return
    if mode == "report":
        # Read-only: regenerate the report from whatever is already on disk.
        # Safe to run from cron while training is in progress.
        for candidate in (
            MODEL_OUTPUT_DIR,
            TRANSFER_OUTPUT_DIR,
            FINE_TUNE_OUTPUT_DIR,
        ):
            if (candidate / "training_log.csv").is_file():
                path = write_training_report(candidate)
                print(build_training_report(candidate))
                print("Written to:", path)
                return
        raise FileNotFoundError(
            "No training_log.csv found in MODEL_OUTPUT_DIR, "
            "TRANSFER_OUTPUT_DIR or FINE_TUNE_OUTPUT_DIR."
        )
    if mode == "prepare":
        prepare_dataset()
        save_dataset_preview()
        print("\nDataset preparation complete.")
        print(
            "Next step: use RUN_MODE='transfer' for the audited V5 warm start "
            "or RUN_MODE='train' for a random-initialization baseline."
        )
        return
    if mode == "preview":
        save_dataset_preview()
        return
    if mode == "train":
        train(fine_tune=False, transfer=False)
        return
    if mode == "transfer":
        train(fine_tune=False, transfer=True)
        return
    if mode == "fine_tune":
        train(fine_tune=True, transfer=False)
        return
    if mode == "predict":
        predict()
        return

    # Add EVALUATION_SPLIT = "test" in settings when an independent labelled
    # test split is ready. It defaults to validation for backward compatibility.
    evaluation_split = str(globals().get("EVALUATION_SPLIT", "val"))
    evaluate(evaluation_split)


if __name__ == "__main__":
    main()
