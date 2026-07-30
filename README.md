## GB-STGNN

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
