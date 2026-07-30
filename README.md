## Data Preparation

1. Download the daily dataset archive, `2_LamaH-CE_daily.tar.gz`, from the [official LamaH-CE data repository](https://doi.org/10.5281/zenodo.4525244).
2. Place the archive in the `dataset/` directory:

```text
dataset/
└── 2_LamaH-CE_daily.tar.gz
```

3. Run the preprocessing script:

```bash
python preprocess_lamah_ce.py \
  --archive dataset/2_LamaH-CE_daily.tar.gz \
  --out-dir dataset/processed_lamah_ce
```

The default configuration uses:

- the `B_basins_intermediate_all` basin delineation;
- daily data from January 1, 1981, to December 31, 2017;
- 1981–2009 for training, 2010–2013 for validation, and 2014–2017 for testing;
- five dynamic features: `qobs`, `prec`, `2m_temp_mean`, `total_et`, and `swe`;
- a 30-day input window and forecast horizons of 1, 3, and 7 days;
- normalization statistics computed exclusively from the training set to prevent data leakage.

## Quick Start

### 1. Build the Granular-Ball Graph

```bash
python build_bfs_farthest_variance_seed_topology_penalty_balls.py
```

### 2. Train and Evaluate

```bash
python train_gb_stgnn_variance_seed_topology_penalty.py
```

If a GPU is not available, replace `--device cuda` with `--device cpu`. The training script selects and saves the best model based on validation MAE. After training, it automatically reports the overall test metrics and the metrics for each forecast horizon.

To explicitly model missing input observations, add:

```bash
--use_missing_mask_channel
```
