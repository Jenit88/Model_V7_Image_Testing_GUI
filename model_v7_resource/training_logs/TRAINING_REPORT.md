# Training report - Model_v7_finetune_new3000

Generated 2026-10-05T21:53:07+00:00

## Where the run is

- Epochs completed: **68** of 120
- Learning rate: 2.575e-05
- Train loss: 0.1068  |  validation loss: 0.1904
- Foreground mIoU: train 0.9360  |  validation 0.8653

## Best checkpoint so far

- Selected on **mask_map50_95** = **0.8351** at epoch 54
- Instance precision 0.9393, recall 0.9696, F1 0.9542 at IoU 0.50
- Mask mAP50 0.9571, mAP50-95 0.8351
- TP 69,204 / FP 4,473 / FN 2,172

### Per class, at the best checkpoint

| class | TP | FP | FN | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| Rectangle | 25,971 | 1,820 | 621 | 0.9345 | 0.9766 | 0.9551 | 0.9822 | 0.8621 |
| Rectangle_concave | 767 | 301 | 61 | 0.7182 | 0.9263 | 0.8091 | 0.9151 | 0.8674 |
| circle | 10,729 | 1,079 | 743 | 0.9086 | 0.9352 | 0.9217 | 0.9520 | 0.8139 |
| circle_full | 31,737 | 1,273 | 747 | 0.9614 | 0.9770 | 0.9692 | 0.9792 | 0.7968 |

### Evaluation history

| epoch | P | R | F1 | mAP50 | mAP50-95 | saved |
|---|---|---|---|---|---|---|
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
| 51 | 0.9380 | 0.9700 | 0.9537 | 0.9549 | 0.8324 |  |
| 54 | 0.9393 | 0.9696 | 0.9542 | 0.9571 | 0.8351 | yes |
| 57 | 0.9419 | 0.9691 | 0.9553 | 0.9549 | 0.8327 |  |
| 60 | 0.9407 | 0.9690 | 0.9547 | 0.9530 | 0.8320 |  |
| 63 | 0.9403 | 0.9704 | 0.9551 | 0.9547 | 0.8335 |  |
| 66 | 0.9408 | 0.9698 | 0.9550 | 0.9536 | 0.8332 |  |
| 69 | 0.9396 | 0.9702 | 0.9547 | 0.9533 | 0.8328 |  |

## What the numbers say

- OVERFITTING: foreground mIoU is 0.936 on train but 0.865 on validation, a gap of 0.071. More augmentation or more data closes this; more epochs will not.
- Still improving: mask_map50_95 is rising about 0.0001 per evaluation over the last 5.

## Semantic IoU by class (last epoch)

| class | train IoU | val IoU |
|---|---|---|
| Rectangle | - | - |
| Rectangle_concave | - | - |
| circle | - | - |
| circle_full | - | - |
