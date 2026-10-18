# split variant C: 60/20/20, larger val/test for lower metric variance (seed 0)

| split | images | share | videos | helmet | motorcycle | no-helmet | plate |
|---|---|---|---|---|---|---|---|
| train | 875 | 60% | 7 | 1543 | 1568 | 565 | 1642 |
| valid | 294 | 20% | 6 | 570 | 608 | 187 | 669 |
| test | 291 | 20% | 4 | 722 | 715 | 193 | 793 |

videos: train: frame_1, frame_11, frame_12, frame_15, frame_16, frame_8, frame_9; valid: frame_10, frame_13, frame_14, frame_4, frame_5, frame_6; test: frame_2, frame_3, frame_7, other:testing_images
manifest sha256: `18e80b7a8f56265a49f110ba880a243baaa66c79482c2232da691745ae368c30`
