# Training report - Model_v7_finetune_new3000

Generated 2026-10-04T03:22:15+00:00

## Where the run is

- Epochs completed: **49** of 120
- Learning rate: 5.535e-05
- Train loss: 0.1145  |  validation loss: 0.1864
- Foreground mIoU: train 0.9300  |  validation 0.8659

## Best checkpoint so far

- Selected on **mask_map50_95** = **0.8327** at epoch 48
- Instance precision 0.9378, recall 0.9700, F1 0.9536 at IoU 0.50
- Mask mAP50 0.9538, mAP50-95 0.8327
- TP 69,232 / FP 4,593 / FN 2,144

### Per class, at the best checkpoint

| class | TP | FP | FN | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| Rectangle | 25,982 | 1,942 | 610 | 0.9305 | 0.9771 | 0.9532 | 0.9820 | 0.8603 |
| Rectangle_concave | 749 | 314 | 79 | 0.7046 | 0.9046 | 0.7922 | 0.9020 | 0.8619 |
| circle | 10,754 | 1,073 | 718 | 0.9093 | 0.9374 | 0.9231 | 0.9517 | 0.8116 |
| circle_full | 31,747 | 1,264 | 737 | 0.9617 | 0.9773 | 0.9694 | 0.9794 | 0.7970 |

### Evaluation history

| epoch | P | R | F1 | mAP50 | mAP50-95 | saved |
|---|---|---|---|---|---|---|
| 1 | 0.8730 | 0.9568 | 0.9130 | 0.9387 | 0.7893 | yes |
| 3 | 0.8905 | 0.9618 | 0.9248 | 0.9498 | 0.8114 | yes |
| 6 | 0.9086 | 0.9656 | 0.9363 | 0.9567 | 0.8224 | yes |
| 9 | 0.9172 | 0.9669 | 0.9414 | 0.9572 | 0.8245 | yes |
| 12 | 0.9195 | 0.9673 | 0.9428 | 0.9573 | 0.8245 | yes |
| 15 | 0.9241 | 0.9689 | 0.9460 | 0.9582 | 0.8269 | yes |
| 18 | 0.9277 | 0.9684 | 0.9476 | 0.9563 | 0.8261 |  |
| 21 | 0.9272 | 0.9696 | 0.9479 | 0.9554 | 0.8269 |  |
| 24 | 0.9306 | 0.9700 | 0.9499 | 0.9565 | 0.8277 | yes |
| 27 | 0.9304 | 0.9690 | 0.9493 | 0.9535 | 0.8267 |  |
| 30 | 0.9337 | 0.9696 | 0.9513 | 0.9547 | 0.8282 | yes |
| 33 | 0.9339 | 0.9697 | 0.9515 | 0.9559 | 0.8296 | yes |
| 36 | 0.9347 | 0.9697 | 0.9519 | 0.9563 | 0.8298 | yes |
| 39 | 0.9372 | 0.9707 | 0.9537 | 0.9554 | 0.8312 | yes |
| 42 | 0.9352 | 0.9702 | 0.9524 | 0.9531 | 0.8303 |  |
| 45 | 0.9365 | 0.9696 | 0.9528 | 0.9538 | 0.8311 |  |
| 48 | 0.9378 | 0.9700 | 0.9536 | 0.9538 | 0.8327 | yes |

## What the numbers say

- OVERFITTING: foreground mIoU is 0.930 on train but 0.866 on validation, a gap of 0.064. More augmentation or more data closes this; more epochs will not.
- Still improving: mask_map50_95 is rising about 0.0006 per evaluation over the last 5.
- Weakest class is Rectangle_concave at F1 0.792 against a 0.909 mean. Check how many training instances it actually has before changing the model.

## Semantic IoU by class (last epoch)

| class | train IoU | val IoU |
|---|---|---|
| Rectangle | - | - |
| Rectangle_concave | - | - |
| circle | - | - |
| circle_full | - | - |
