# CSI and station-position fusion

For station `b`, the shared CSI encoder produces `c_b` and the position MLP
produces `p_b`, both with dimension 128. Three fusion rules were tested:

- addition: `s_b = c_b + p_b`;
- concatenation: `s_b = MLP([c_b; p_b])`;
- gating: `g_b = sigmoid(MLP([c_b; p_b]))`, then
  `s_b = g_b * c_b + (1 - g_b) * p_b`.

The learned MLP is shared across stations and has dimensions `256 → 128 → 128`.
Only this per-station operation changes. The radio token, station-set Transformer,
map encoder, cross-attention and regression head remain fixed.

Addition registers no extra parameters and remains compatible with the original
checkpoint. Each learned rule adds 49,408 parameters. Initialization of the extra
MLP uses a forked CPU random-number context so matched seeds retain the same shared
weights, data order and station sampling.

Tests cover the three formulas, gradients, masked padding, station permutation,
invalid empty station sets and strict loading of the original additive checkpoint.
The gate is inspired by Arevalo et al., *Gated Multimodal Units for Information
Fusion* (2017); this implementation is a small ablation rather than a reproduction
of that architecture.
