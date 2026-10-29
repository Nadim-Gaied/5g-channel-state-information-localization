# Multi-map dataset

The map-conditioned experiment uses 41 DeepMIMO scenarios at 3.5 GHz. Each
scenario contributes 2,048 sampled user positions with at least two active base
stations. The split is made by physical map: 29 maps for training, six for
validation and six for test. Variants of the same city are kept in one split.

Each processed scenario stores complex CSI with shape `[users, stations, 32, 64]`,
station coordinates and availability, user coordinates, a two-channel building
raster and metric map bounds.

Real and imaginary CSI statistics are fitted only on active links from the 29
training maps. The saved statistics are reused unchanged for validation and test.
The radio setup is fixed: 3.5 GHz, 20 MHz bandwidth, a 32-element half-wavelength
array, 64 selected OFDM subcarriers and up to 25 stored propagation paths.

## Preparation

The commands assume that the DeepMIMO scenarios are available under
`deepmimo_scenarios/`.

```powershell
.\.venv\Scripts\python.exe tools\inventory_multimap_scenarios.py
.\.venv\Scripts\python.exe tools\process_multimap_batch.py --config configs\multimap\full_city_batch.json
.\.venv\Scripts\python.exe tools\compute_multimap_normalization.py --config configs\multimap\full_city_batch.json --output data\multimap\normalization\full_train_stats.npz
```

Processing skips valid existing outputs. Use `--scenario NAME` to rebuild one
scenario or `--overwrite` to replace an existing processed file.
