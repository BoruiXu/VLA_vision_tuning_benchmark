# Recomputed results

Success rates are percentages; changes are percentage points. Overall pools episodes. Off-target is the unweighted mean over six dimensions.

| Run | Episodes | Successes | Overall (%) |
|---|---:|---:|---:|
| oft-plus-stock | 10030 | 6853 | 68.325 |
| oft-plus-camera_only | 10030 | 7603 | 75.803 |
| oft-plus-noise_only | 10030 | 7194 | 71.725 |
| pi05-plus-stock | 10030 | 8428 | 84.028 |
| pi05-plus-camera_only | 10030 | 8727 | 87.009 |
| pi05-plus-noise_only | 10030 | 8460 | 84.347 |
| internvla15-plus-stock | 10030 | 8399 | 83.739 |
| internvla15-plus-camera_only | 10030 | 7976 | 79.521 |
| internvla15-plus-noise_only | 10030 | 8142 | 81.176 |
| nora15-plus-stock | 10030 | 4757 | 47.428 |
| nora15-plus-camera_only | 10030 | 4389 | 43.759 |
| nora15-plus-noise_only | 10030 | 5381 | 53.649 |
| xvla-plus-stock | 10030 | 6950 | 69.292 |
| xvla-plus-camera_only | 10030 | 7652 | 76.291 |
| xvla-plus-noise_only | 10030 | 7275 | 72.532 |
| oft-plus-mixed | 10030 | 7892 | 78.684 |
| pi05-plus-mixed | 10030 | 8853 | 88.265 |
| xvla-plus-mixed | 10030 | 8315 | 82.901 |
| nora15-plus-mixed | 10030 | 5491 | 54.746 |
| internvla15-plus-mixed | 10030 | 8188 | 81.635 |
| oft-plus-camera-connector_lora | 10030 | 7511 | 74.885 |
| oft-plus-camera-backbone_lora | 10030 | 7569 | 75.464 |
| oft-plus-camera-delta | 10030 | 5190 | 51.745 |
| oft-plus-camera-wiseft_025 | 10030 | 7164 | 71.426 |
| oft-plus-camera-wiseft_050 | 10030 | 7374 | 73.519 |
| oft-plus-camera-wiseft_075 | 10030 | 7641 | 76.181 |
| oft-screen-step0 | 560 | 379 | 67.679 |
| oft-screen-step2000 | 560 | 404 | 72.143 |
| oft-screen-step4000 | 560 | 406 | 72.500 |
| oft-screen-step8000 | 560 | 421 | 75.179 |
| oft-screen-step12000 | 560 | 419 | 74.821 |
| oft-screen-seed43 | 560 | 429 | 76.607 |
| oft-screen-seed44 | 560 | 418 | 74.643 |
| xvla-plus-camera-connector_lora | 10030 | 6925 | 69.043 |
| xvla-plus-camera-backbone_lora | 10030 | 7712 | 76.889 |
| xvla-plus-camera-delta | 10030 | 8046 | 80.219 |
| xvla-plus-camera-wiseft_025 | 10030 | 7209 | 71.874 |
| xvla-plus-camera-wiseft_050 | 10030 | 7551 | 75.284 |
| xvla-plus-camera-wiseft_075 | 10030 | 7897 | 78.734 |
| xvla-screen-step0 | 560 | 394 | 70.357 |
| xvla-screen-step2000 | 560 | 416 | 74.286 |
| xvla-screen-step4000 | 560 | 431 | 76.964 |
| xvla-screen-step8000 | 560 | 431 | 76.964 |
| xvla-screen-step12000 | 560 | 431 | 76.964 |
| xvla-screen-seed43 | 560 | 429 | 76.607 |
| xvla-screen-seed44 | 560 | 416 | 74.286 |
| oft-original-stock | 2000 | 1944 | 97.200 |
| oft-original-camera-only | 2000 | 1933 | 96.650 |
| oft-original-noise-only | 2000 | 1932 | 96.600 |
| pi05-original-stock | 2000 | 1947 | 97.350 |
| pi05-original-camera-only | 2000 | 1933 | 96.650 |
| pi05-original-noise-only | 2000 | 1921 | 96.050 |
| internvla15-original-stock | 2000 | 1965 | 98.250 |
| internvla15-original-camera-only | 2000 | 1959 | 97.950 |
| internvla15-original-noise-only | 2000 | 1946 | 97.300 |
| nora15-original-stock | 2000 | 1894 | 94.700 |
| nora15-original-camera-only | 2000 | 1626 | 81.300 |
| nora15-original-noise-only | 2000 | 1855 | 92.750 |
| xvla-original-stock | 2000 | 1950 | 97.500 |
| xvla-original-camera-only | 2000 | 1874 | 93.700 |
| xvla-original-noise-only | 2000 | 1929 | 96.450 |

## Targeted tuning

| Model | Training | Target base → tuned | Off-target change [95% CI] | Overall base → tuned |
|---|---|---:|---:|---:|
| oft | camera_only | 55.22 → 94.50 | +1.40 [+0.17, +2.77] | 68.33 → 75.80 |
| oft | noise_only | 72.77 → 92.57 | +0.32 [-1.11, +1.68] | 68.33 → 71.72 |
| pi05 | camera_only | 70.67 → 92.68 | -0.34 [-1.67, +1.18] | 84.03 → 87.01 |
| pi05 | noise_only | 86.51 → 96.69 | -1.35 [-2.06, -0.65] | 84.03 → 84.35 |
| internvla15 | camera_only | 80.80 → 83.43 | -5.22 [-7.34, -3.48] | 83.74 → 79.52 |
| internvla15 | noise_only | 93.69 → 91.76 | -2.52 [-3.43, -1.62] | 83.74 → 81.18 |
| nora15 | camera_only | 6.00 → 34.71 | -10.14 [-13.90, -6.73] | 47.43 → 43.76 |
| nora15 | noise_only | 15.99 → 61.27 | -1.01 [-3.45, +1.36] | 47.43 → 53.65 |
| xvla | camera_only | 19.57 → 88.99 | -4.65 [-7.38, -2.07] | 69.29 → 76.29 |
| xvla | noise_only | 60.90 → 93.69 | -2.21 [-3.76, -0.70] | 69.29 → 72.53 |

## Training-seed sensitivity (fixed 560 episodes)

| Model | Seed | Target gain | Off-target mean change | Overall gain |
|---|---:|---:|---:|---:|
| xvla | 42 | +70.00 | -3.96 | +6.61 |
| xvla | 43 | +73.75 | -5.00 | +6.25 |
| xvla | 44 | +70.00 | -7.08 | +3.93 |
| oft | 42 | +38.75 | +1.88 | +7.14 |
| oft | 43 | +40.00 | +3.75 | +8.93 |
| oft | 44 | +41.25 | +1.25 | +6.96 |
