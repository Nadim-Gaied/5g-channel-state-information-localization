# CSI-Based User Localization in 5G Networks

## Problem

Channel State Information (CSI) describes the complex response between a user and
a base station across antennas and OFDM subcarriers. Reflections, diffraction,
path delays and arrival angles make this response dependent on position. This
project treats localization as supervised regression from CSI to horizontal user
coordinates.

For antenna `m` and subcarrier `k`, a simplified multipath channel is

```text
H[m, k] = sum_l alpha_l a_m(theta_l) exp(-j 2 pi f_k tau_l)
```

Each path has complex gain `alpha_l`, delay `tau_l` and arrival angle `theta_l`.
The CSI matrix is an antenna-by-frequency observation at a base station; it is not
a spatial grid of wave propagation.

All channels are generated from DeepMIMO ray-tracing scenarios. They represent
ideal simulated CSI. Controlled complex AWGN is added in one experiment, but pilot
contamination, RF calibration, mobility and structured interference are outside
the evaluation.

## Known-map experiment

The New York scenario contains 17,337 active user positions. Spatial tiles rather
than individual samples are assigned to train, validation and test, producing
12,162 / 2,582 / 2,593 examples. Each complex `32 × 64` CSI matrix is represented
by real and imaginary channels. Feature and coordinate normalization is fitted on
the training split only.

The residual CNN contains 451,762 parameters. It uses three convolutional stages,
residual blocks and a coordinate-regression head. The 460,322-parameter ViT uses
`4 × 8` CSI patches, a 96-dimensional embedding, four Transformer layers and six
attention heads. Both models use AdamW, identical data splits, 35 epochs and the
same three seeds.

On clean test CSI, the CNN achieved 11.50 ± 1.24 m mean error, compared with
17.32 ± 0.19 m for the ViT. Global attention did not improve accuracy under this
data budget.

### Noise experiment

Complex AWGN was applied before normalization, with noise power derived separately
from each sample's CSI power. With clean-only training, the ViT degraded less at
0 dB SNR: 25.01 m mean error versus 27.32 m for the CNN. The CNN remained better
at all less severe tested SNRs.

Both models were then retrained with SNR sampled uniformly from 0 to 30 dB while
20% of examples remained clean. The augmented CNN reached 14.21 m at 0 dB, a 48%
reduction from clean-only training, while retaining 11.75 m clean error. It then
outperformed the augmented ViT at every tested SNR. Changing the training
distribution mattered more than choosing global attention in this experiment.

## Generalization to unseen maps

A CSI-only model trained on one environment has no explicit knowledge of a new
coordinate system, building layout or station geometry. The map-conditioned model
therefore receives CSI from every available station, station coordinates, and a
building occupancy and height raster.

A shared CNN encodes each station's CSI and an MLP encodes its coordinates. A
masked Transformer aggregates a variable set of station tokens. A separate CNN
turns the map and explicit x/y coordinate channels into spatial tokens. The radio
summary queries these tokens through cross-attention, and the final head predicts
normalized coordinates. Station masking makes padding inert, and the set encoder
is invariant to station ordering.

The corpus contains 41 maps and 83,968 users. Entire maps are held out: 29 maps
for training, six for validation and six for test. The reference model achieved
35.46 m mean and 21.46 m median error on all 12,288 users from the unseen test
maps without target-map retraining.

On a smaller common benchmark of 1,894 positions, the general model achieved
38.24 m mean error. Separate CSI-only CNNs trained on 70% of each target map
achieved 50.73 m. This compares complete systems: the general model has multiple
links, geometry inputs, more parameters and broader training data, so the 24.6%
difference cannot be attributed to the map encoder alone.

## Fusion ablation

The per-station combination of CSI and position embeddings was tested with
element-wise addition, a shared concatenation MLP and a sigmoid gate. All
downstream layers and training controls were fixed. Each condition was trained
with seeds 42, 43 and 44.

Addition had the lowest validation mean at 38.02 m. The concatenation MLP was the
better learned alternative and was selected before test evaluation. On the test
maps it reduced mean error from 36.16 to 33.87 m and median error from 22.09 to
18.27 m. It also added 5.8% more parameters and slightly increased P90 and P95.
The validation/test ranking reversal and three-seed sample make this a descriptive
trade-off, not evidence that learned fusion is uniformly better.

## Limitations

- The study uses simulated channels and one fixed radio configuration.
- AWGN does not represent real interference or channel-estimation errors.
- The multi-map model assumes the target map and station positions are known.
- One physical-map split and three seeds do not support significance claims.
- The multimodal baseline does not isolate the contribution of each input.

The next useful tests are modality-removal ablations, several held-out map splits,
missing-station and interference experiments, and evaluation on measured CSI.

## References

- A. Alkhateeb, “DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave
  and Massive MIMO Applications,” ITA, 2019.
- K. He et al., “Deep Residual Learning for Image Recognition,” CVPR, 2016.
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.
- A. Dosovitskiy et al., “An Image Is Worth 16x16 Words,” ICLR, 2021.
- A. Foliadis et al., “CSI-Based Localization with CNNs Exploiting Phase
  Information,” WCNC, 2021.
