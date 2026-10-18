# Split proposals (group = video; nothing has been moved)

# split variant A: 70/15/15, images and balanced class equally important (seed 0)

| split | images | share | videos | helmet | motorcycle | no-helmet | plate |
|---|---|---|---|---|---|---|---|
| train | 1026 | 70% | 7 | 2105 | 2102 | 662 | 2263 |
| valid | 214 | 15% | 4 | 468 | 471 | 140 | 558 |
| test | 220 | 15% | 6 | 262 | 318 | 143 | 283 |

videos: train: frame_1, frame_12, frame_2, frame_3, frame_5, frame_7, frame_8; valid: frame_10, frame_11, frame_14, frame_15; test: frame_13, frame_16, frame_4, frame_6, frame_9, other:testing_images
manifest sha256: `fe103d4edf2352e15a22e0b61cdea03af6b65e34e6341e606f5c8a9694d92279`

# split variant B: 70/15/15, balanced class first (split sizes may drift) (seed 0)

| split | images | share | videos | helmet | motorcycle | no-helmet | plate |
|---|---|---|---|---|---|---|---|
| train | 1014 | 70% | 8 | 2116 | 2082 | 660 | 2260 |
| valid | 226 | 16% | 3 | 457 | 491 | 142 | 561 |
| test | 220 | 15% | 6 | 262 | 318 | 143 | 283 |

videos: train: frame_1, frame_11, frame_12, frame_15, frame_2, frame_3, frame_7, frame_8; valid: frame_10, frame_14, frame_5; test: frame_13, frame_16, frame_4, frame_6, frame_9, other:testing_images
manifest sha256: `9e01a09fadf64ff49c75ec9f18ee9530d0f7642946ee0abe29103e0f2198d00d`

# split variant C: 60/20/20, larger val/test for lower metric variance (seed 0)

| split | images | share | videos | helmet | motorcycle | no-helmet | plate |
|---|---|---|---|---|---|---|---|
| train | 875 | 60% | 7 | 1543 | 1568 | 565 | 1642 |
| valid | 294 | 20% | 6 | 570 | 608 | 187 | 669 |
| test | 291 | 20% | 4 | 722 | 715 | 193 | 793 |

videos: train: frame_1, frame_11, frame_12, frame_15, frame_16, frame_8, frame_9; valid: frame_10, frame_13, frame_14, frame_4, frame_5, frame_6; test: frame_2, frame_3, frame_7, other:testing_images
manifest sha256: `18e80b7a8f56265a49f110ba880a243baaa66c79482c2232da691745ae368c30`
