from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lamah_dataset import LamaHCEDataset
from metrics import masked_kge, masked_mae, masked_nse, masked_rmse
from models.gb_stgnn import GBSTGNNDiffusion


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GB-STGNN with variance-seed topology-penalty balls."
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("dataset/processed_lamah_ce"))
    parser.add_argument("--balls_dir", type=Path, default=Path("outputs/variance_seed_bfs_farthest_topology_penalty_sqrtN"))
    parser.add_argument("--input_len", type=int, default=30)
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_gru_layers", type=int, default=2)
    parser.add_argument("--num_ball_diffusion_layers", type=int, default=2)
    parser.add_argument("--diffusion_steps", type=int, default=2)
    parser.add_argument(
        "--use_missing_mask_channel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append an observation-mask channel to the model input and use "
            "mask-weighted granular-ball mean pooling."
        ),
    )
    parser.add_argument("--theta_q", type=float, default=0.80)
    parser.add_argument("--min_adaptive_balls", type=int, default=4)
    parser.add_argument("--min_adaptive_ball_size", type=int, default=1)
    parser.add_argument("--max_single_node_ball_pct", type=float, default=0.05)
    parser.add_argument("--topology_cut_weight", type=float, default=1.0)
    parser.add_argument("--river_weight", type=float, default=1.0)
    parser.add_argument("--quality_connectivity_weight", type=float, default=0.0)
    parser.add_argument("--quality_synchrony_weight", type=float, default=1.0)
    parser.add_argument("--lambda_sim", type=float, default=0.1)
    parser.add_argument("--topk_sim", type=int, default=1)
    parser.add_argument("--isolated_sim_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_site_missing_rate", type=float, default=0.0)
    parser.add_argument("--train_noise_std", type=float, default=0.0)
    parser.add_argument("--impute_value", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_dir", type=Path, default=Path("runs/gb_stgnn_bfs_farthest_topology_penalty"))
    parser.add_argument("--loss", choices=["mae", "huber"], default="huber")
    parser.add_argument("--beta_ball", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"--horizons must contain positive integers, got {value!r}")
    return horizons


def choose_device(requested: str, gpu_id: int | None) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda" and gpu_id is not None:
        count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= count:
            raise ValueError(f"--gpu_id must be in [0, {count - 1}], got {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(requested)


def load_array(directory: Path, name: str, dtype=None) -> np.ndarray:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    values = np.load(path, allow_pickle=True)
    return values.astype(dtype) if dtype is not None else values


def load_quality_definition(balls_dir: Path) -> str:
    """Load the quality definition recorded for the prebuilt balls."""
    metadata_path = balls_dir / "metadata.json"
    if not metadata_path.exists():
        return "(connectivity + synchrony) / 2"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(
        metadata.get("quality_definition", "(connectivity + synchrony) / 2")
    )


def ensure_balls_match_training_args(args: argparse.Namespace) -> None:
    """Build granular balls when the cached files do not match this run."""
    metadata_path = args.balls_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = None

    expected_weights = {
        "connectivity": float(args.quality_connectivity_weight),
        "synchrony": float(args.quality_synchrony_weight),
    }
    expected = {
        "theta_q": float(args.theta_q),
        "min_ball_size": int(args.min_adaptive_ball_size),
        "topology_cut_weight": float(args.topology_cut_weight),
        "river_weight": float(args.river_weight),
        "lambda_sim": float(args.lambda_sim),
        "topk_sim": int(args.topk_sim),
    }
    required_files = (
        "node_to_ball.npy", "ball_sizes.npy", "ball_quality.npy",
        "hydro_dist_to_ball_center.npy", "A_ball.npy", "A_phy.npy", "A_sim.npy",
    )

    def values_match() -> bool:
        if metadata is None or not all(
            (args.balls_dir / name).exists() for name in required_files
        ):
            return False
        try:
            for key, expected_value in expected.items():
                actual = metadata.get(key)
                if isinstance(expected_value, float):
                    if actual is None or not np.isclose(float(actual), expected_value):
                        return False
                elif actual != expected_value:
                    return False
            actual_weights = metadata.get("quality_weights")
            if not isinstance(actual_weights, dict):
                return False
            return all(
                key in actual_weights
                and np.isclose(float(actual_weights[key]), expected_value)
                for key, expected_value in expected_weights.items()
            )
        except (TypeError, ValueError):
            return False

    if values_match():
        print(
            f"Reusing granular balls from {args.balls_dir} | "
            f"theta_q={args.theta_q:g} | num_balls={metadata.get('num_balls')}"
        )
        return

    builder_path = Path(__file__).resolve().parent / (
        "build_bfs_farthest_variance_seed_topology_penalty_balls.py"
    )
    if not builder_path.exists():
        raise FileNotFoundError(f"Missing granular-ball builder: {builder_path}")
    print(
        f"Building granular balls in {args.balls_dir} for theta_q={args.theta_q:g}; "
        "cached metadata is missing or does not match this run."
    )
    command = [
        sys.executable,
        str(builder_path),
        "--processed_dir", str(args.processed_dir.resolve()),
        "--output_dir", str(args.balls_dir.resolve()),
        "--theta_q", str(args.theta_q),
        "--min_ball_size", str(args.min_adaptive_ball_size),
        "--topology_cut_weight", str(args.topology_cut_weight),
        "--river_weight", str(args.river_weight),
        "--quality_connectivity_weight", str(args.quality_connectivity_weight),
        "--quality_synchrony_weight", str(args.quality_synchrony_weight),
        "--lambda_sim", str(args.lambda_sim),
        "--topk_sim", str(args.topk_sim),
    ]
    result = subprocess.run(
        command,
        cwd=builder_path.parent,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    rebuilt = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not np.isclose(float(rebuilt.get("theta_q", -1.0)), args.theta_q):
        raise RuntimeError(
            f"Built granular-ball metadata has theta_q={rebuilt.get('theta_q')}, "
            f"expected {args.theta_q}"
        )


def prepare_prebuilt_balls(args: argparse.Namespace):
    node_to_ball = load_array(args.balls_dir, "node_to_ball.npy", np.int64)
    A_ball = load_array(args.balls_dir, "A_ball.npy", np.float32)
    A_phy = load_array(args.balls_dir, "A_phy.npy", np.float32)
    A_sim = load_array(args.balls_dir, "A_sim.npy", np.float32)
    ball_sizes = load_array(args.balls_dir, "ball_sizes.npy", np.int64)
    ball_quality = load_array(args.balls_dir, "ball_quality.npy", np.float32)
    hydro_dist = load_array(args.balls_dir, "hydro_dist_to_ball_center.npy", np.float32)
    static = load_array(args.processed_dir, "S.npy")
    num_nodes, num_balls = len(static), len(ball_sizes)
    if node_to_ball.shape != (num_nodes,) or hydro_dist.shape != (num_nodes,):
        raise ValueError("Station-level granular-ball arrays do not match the processed node count")
    if any(matrix.shape != (num_balls, num_balls) for matrix in (A_ball, A_phy, A_sim)):
        raise ValueError(f"Ball adjacency matrices must have shape {(num_balls, num_balls)}")
    if ball_quality.shape != (num_balls,):
        raise ValueError("ball_quality.npy does not match the ball count")
    if not np.array_equal(np.bincount(node_to_ball, minlength=num_balls), ball_sizes):
        raise ValueError("ball_sizes.npy does not match node_to_ball.npy")
    return node_to_ball, A_ball, A_phy, A_sim, ball_sizes, ball_quality, hydro_dist


def prepare_model_inputs(x, x_mask, use_missing_mask_channel):
    if not use_missing_mask_channel:
        return x
    if x_mask is None:
        mask_channel = torch.ones(
            (*x.shape[:-1], 1), dtype=x.dtype, device=x.device
        )
    else:
        x_mask = x_mask.to(device=x.device)
        if x_mask.shape != x.shape:
            raise ValueError(
                f"x_mask must match x, got {tuple(x_mask.shape)} and {tuple(x.shape)}"
            )
        mask_channel = x_mask.to(dtype=x.dtype).amin(dim=-1, keepdim=True)
    return torch.cat([x, mask_channel], dim=-1)


def augment_training_inputs(x, x_mask, site_missing_rate, noise_std, impute_value):
    x = x.clone()
    mask = x_mask.clone() if x_mask is not None else torch.ones_like(x)
    if site_missing_rate > 0:
        missing = (torch.rand(x.shape[0], x.shape[2], device=x.device) < site_missing_rate)[:, None, :, None]
        x = torch.where(missing, x.new_full((), impute_value), x)
        mask = torch.where(missing, torch.zeros_like(mask), mask)
    if noise_std > 0:
        x = torch.where(mask.bool(), x + torch.randn_like(x) * noise_std, x)
    return x, mask


def masked_loss(y_pred, y_true, mask, loss_name):
    valid = mask.bool() & torch.isfinite(y_pred) & torch.isfinite(y_true)
    if not torch.any(valid):
        raise RuntimeError("Batch has no valid target values after applying y_mask.")
    if loss_name == "mae":
        return torch.mean(torch.abs(y_pred[valid] - y_true[valid]))
    return F.smooth_l1_loss(y_pred[valid], y_true[valid])


def ball_consistency_loss(y_pred, y_true, mask, node_to_ball, ball_quality, eps=1e-8):
    valid = mask.bool() & torch.isfinite(y_pred) & torch.isfinite(y_true)
    valid_float = valid.to(y_pred.dtype)
    residual = torch.where(valid, y_pred - y_true, torch.zeros_like(y_pred))
    batch_size, num_nodes, num_horizons = residual.shape
    num_balls = int(ball_quality.numel())
    index = node_to_ball.view(1, num_nodes, 1).expand(batch_size, -1, num_horizons)

    residual_sum = residual.new_zeros(batch_size, num_balls, num_horizons)
    residual_sum.scatter_add_(1, index, residual * valid_float)
    valid_count = residual.new_zeros(batch_size, num_balls, num_horizons)
    valid_count.scatter_add_(1, index, valid_float)
    mean_residual = residual_sum / valid_count.clamp_min(eps)

    station_mean = mean_residual.gather(1, index)
    deviation = torch.abs(residual - station_mean) * valid_float
    deviation_sum = residual.new_zeros(batch_size, num_balls, num_horizons)
    deviation_sum.scatter_add_(1, index, deviation)

    per_ball_deviation = deviation_sum.sum(dim=(0, 2))
    per_ball_count = valid_count.sum(dim=(0, 2))
    active = (per_ball_count > 0).to(y_pred.dtype)
    per_ball_loss = ball_quality.to(y_pred.dtype) * per_ball_deviation / per_ball_count.clamp_min(eps)
    return (per_ball_loss * active).sum() / active.sum().clamp_min(1.0)


def combined_training_loss(y_pred, y_true, mask, loss_name, beta_ball, node_to_ball, ball_quality):
    pred_loss = masked_loss(y_pred, y_true, mask, loss_name)
    if beta_ball <= 0:
        zero = y_pred.new_tensor(0.0)
        return pred_loss, pred_loss, zero
    ball_loss = ball_consistency_loss(y_pred, y_true, mask, node_to_ball, ball_quality)
    return pred_loss + beta_ball * ball_loss, pred_loss, ball_loss


def metrics(pred, true, mask):
    return {
        "mae": float(masked_mae(pred, true, mask).item()),
        "rmse": float(masked_rmse(pred, true, mask).item()),
        "nse": float(masked_nse(pred, true, mask).item()),
        "kge": float(masked_kge(pred, true, mask).item()),
    }


@torch.no_grad()
def evaluate_arrays(
    model, loader, device, node_to_ball, hydro_dist, A_ball, ball_quality,
):
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        x = prepare_model_inputs(
            x, batch.get("x_mask"), model.use_missing_mask_channel
        )
        predictions.append(model(
            x, batch["static"].to(device), node_to_ball, hydro_dist, A_ball,
            ball_quality,
        ).cpu())
        targets.append(batch["y"])
        masks.append(batch["y_mask"])
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def evaluate(
    model, loader, device, node_to_ball, hydro_dist, A_ball, ball_quality,
):
    return metrics(*evaluate_arrays(
        model, loader, device, node_to_ball, hydro_dist, A_ball, ball_quality,
    ))


def evaluate_horizons(
    model, loader, device, node_to_ball, hydro_dist, A_ball, ball_quality, horizons,
):
    pred, true, mask = evaluate_arrays(
        model, loader, device, node_to_ball, hydro_dist, A_ball, ball_quality,
    )
    per_horizon = {
        horizon: metrics(pred[:, :, index:index + 1], true[:, :, index:index + 1], mask[:, :, index:index + 1])
        for index, horizon in enumerate(horizons)
    }
    return metrics(pred, true, mask), per_horizon


def make_loader(processed_dir, split, input_len, horizons, batch_size, shuffle, num_workers):
    dataset = LamaHCEDataset(
        processed_dir=processed_dir, split=split, input_len=input_len,
        horizons=horizons, target_mode="raw", return_static=True, return_adj=False,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def count_nonzero_offdiag(matrix):
    values = matrix.detach().cpu().numpy() if isinstance(matrix, torch.Tensor) else np.asarray(matrix)
    if values.ndim == 3:
        values = values[0]
    mask = values > 0
    np.fill_diagonal(mask, False)
    return int(mask.sum())


def print_graph_statistics(node_to_ball, A_phy, A_sim):
    print("--------------------------------------------------")
    print("Graph Statistics (Test Phase):")
    print(f"M (granular balls) = {len(np.unique(node_to_ball))}")
    print(f"|E_phy| = {count_nonzero_offdiag(A_phy)}")
    print(f"|E_sim| = {count_nonzero_offdiag(A_sim)}")
    print("--------------------------------------------------")


def print_graph_propagation_profile(model, A_ball, sample_count):
    profile = model.graph_propagation_profile()
    total, calls = float(profile["seconds"]), int(profile["calls"])
    print(
        f"Graph propagation time: total={total:.4f}s | "
        f"per_batch={1000 * total / max(calls, 1):.3f} ms/batch | "
        f"per_sample={1000 * total / max(sample_count, 1):.6f} ms/sample | "
        f"batches={calls} | samples={sample_count} | graph_nodes={A_ball.shape[-1]} | "
        f"graph_edges={count_nonzero_offdiag(A_ball)}"
    )


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.train_site_missing_rate <= 1:
        raise ValueError("--train_site_missing_rate must be in [0, 1]")
    if args.train_noise_std < 0:
        raise ValueError("--train_noise_std must be non-negative")
    if not 0 <= args.theta_q <= 1:
        raise ValueError("--theta_q must be in [0, 1]")
    if args.min_adaptive_ball_size < 1:
        raise ValueError("--min_adaptive_ball_size must be at least 1")
    if args.topology_cut_weight < 0 or args.river_weight < 0:
        raise ValueError("Topology and river weights must be non-negative")
    if args.quality_connectivity_weight < 0 or args.quality_synchrony_weight < 0:
        raise ValueError("Quality weights must be non-negative")
    if not np.isclose(
        args.quality_connectivity_weight + args.quality_synchrony_weight, 1.0
    ):
        raise ValueError("Quality connectivity and synchrony weights must sum to 1")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    horizons = parse_horizons(args.horizons)
    device = choose_device(args.device, args.gpu_id)
    ensure_balls_match_training_args(args)
    features = load_array(args.processed_dir, "features.npy")
    static = load_array(args.processed_dir, "S.npy")
    raw_input_dim = len(features)
    input_dim = raw_input_dim + int(args.use_missing_mask_channel)
    static_dim, horizon_dim = static.shape[1], len(horizons)
    node_np, A_ball_np, A_phy_np, A_sim_np, sizes, quality_np, hydro_np = prepare_prebuilt_balls(args)
    quality_definition = load_quality_definition(args.balls_dir)
    num_balls = int(len(sizes))
    single_node_balls = int(np.count_nonzero(sizes == 1))
    single_node_ball_pct = 100.0 * single_node_balls / num_balls if num_balls else 0.0
    print(
        "Adaptive granular balls | "
        f"num_balls={num_balls} | avg_size={sizes.mean():.2f} | "
        f"max_size={sizes.max()} | min_size={sizes.min()} | avg_quality={quality_np.mean():.4f}"
    )
    print(
        "Single-node granular balls | "
        f"count={single_node_balls} | pct_of_balls={single_node_ball_pct:.2f}%"
    )
    node = torch.from_numpy(node_np).long().to(device)
    A_ball = torch.from_numpy(A_ball_np).float().to(device)
    hydro = torch.from_numpy(hydro_np).float().to(device)
    quality = torch.from_numpy(quality_np).float().to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best_gb_stgnn.pt"
    config = vars(args).copy()
    config.update({
        "processed_dir": str(args.processed_dir), "balls_dir": str(args.balls_dir),
        "save_dir": str(args.save_dir), "horizons": horizons, "input_dim": input_dim,
        "raw_input_dim": raw_input_dim, "static_dim": static_dim, "horizon_dim": horizon_dim,
        "use_missing_mask_channel": bool(args.use_missing_mask_channel),
        "use_masked_ball_pool": bool(args.use_missing_mask_channel),
        "ball_representation": {
            "aggregation": (
                "masked_mean_pool" if args.use_missing_mask_channel else "mean_pool"
            ),
            "use_masked_ball_pool": bool(args.use_missing_mask_channel),
            "mask_source": (
                "input mask channel averaged over the input window"
                if args.use_missing_mask_channel else None
            ),
            "description": (
                "Mask-weighted mean of time-encoded node hidden states inside each granular ball; hydro_dist_to_ball is not used for ball pooling."
                if args.use_missing_mask_channel
                else "Mean of time-encoded node hidden states inside each granular ball; hydro_dist_to_ball is not used for ball pooling."
            ),
        },
        "quality_weighted_adjacency": {
            "formula": "A_quality[i,j] = Q[i] * Q[j] * A_ball[i,j] before forward/reverse diffusion normalization.",
            "ball_quality_definition": quality_definition,
            "applied_in": "models.gb_stgnn.GBSTGNNDiffusion.forward",
            "recompute_on_missing_ball": False, "repartition_on_missing": False,
        },
        "training_augmentation": {
            "site_missing_rate": args.train_site_missing_rate,
            "noise_std": args.train_noise_std, "impute_value": args.impute_value,
        },
        "ball_size_summary": {
            "num_balls": num_balls, "avg_size": float(sizes.mean()),
            "max_size": int(sizes.max()), "min_size": int(sizes.min()),
            "single_node_balls": single_node_balls,
            "single_node_ball_pct": single_node_ball_pct,
            "avg_quality": float(quality_np.mean()),
        },
    })
    (args.save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_loader = make_loader(args.processed_dir, "train", args.input_len, horizons, args.batch_size, True, args.num_workers)
    val_loader = make_loader(args.processed_dir, "val", args.input_len, horizons, args.batch_size, False, args.num_workers)
    test_loader = make_loader(args.processed_dir, "test", args.input_len, horizons, args.batch_size, False, args.num_workers)
    model = GBSTGNNDiffusion(
        input_dim=input_dim, static_dim=static_dim, hidden_dim=args.hidden_dim,
        num_gru_layers=args.num_gru_layers,
        num_ball_diffusion_layers=args.num_ball_diffusion_layers,
        diffusion_steps=args.diffusion_steps, horizon_dim=horizon_dim,
        dropout=args.dropout,
        use_missing_mask_channel=args.use_missing_mask_channel,
        use_masked_ball_pool=args.use_missing_mask_channel,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(device)})"
    print(
        f"Training GBSTGNNDiffusion on {device_label} | input_dim={input_dim}, static_dim={static_dim}, "
        f"horizons={horizons}, train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}"
    )
    print(f"batch_size={args.batch_size}, hidden_dim={args.hidden_dim}, beta_ball={args.beta_ball}, num_workers={args.num_workers}")
    print(f"theta_q={args.theta_q}, topk_sim={args.topk_sim}, lambda_sim={args.lambda_sim},isolated_sim_fallback={args.isolated_sim_fallback}")
    print(f"train_site_missing_rate={args.train_site_missing_rate}, train_noise_std={args.train_noise_std}")
    print(
        f"use_missing_mask_channel={args.use_missing_mask_channel}, "
        f"use_masked_ball_pool={args.use_missing_mask_channel}, "
        "recompute_missing_quality=False"
    )

    best_val_mae = float("inf")
    training_epoch_times = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        total_loss = total_pred = total_ball = count = 0.0
        for batch in train_loader:
            x_raw = batch["x"].to(device)
            x_mask = batch.get("x_mask")
            x_mask = x_mask.to(device) if x_mask is not None else None
            x_raw, x_mask = augment_training_inputs(
                x_raw, x_mask, args.train_site_missing_rate, args.train_noise_std, args.impute_value
            )
            x = prepare_model_inputs(
                x_raw, x_mask, args.use_missing_mask_channel
            )
            y, y_mask = batch["y"].to(device), batch["y_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x, batch["static"].to(device), node, hydro, A_ball, quality
            )
            loss, pred_loss, ball_loss = combined_training_loss(
                prediction, y, y_mask, args.loss, args.beta_ball, node, quality
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = x.shape[0]
            total_loss += loss.item() * batch_count
            total_pred += pred_loss.item() * batch_count
            total_ball += ball_loss.item() * batch_count
            count += batch_count
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_epoch_times.append(time.perf_counter() - start)
        val = evaluate(
            model, val_loader, device, node, hydro, A_ball, quality,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"Epoch {epoch:03d} | total_loss={total_loss/max(count,1):.6f} | "
            f"pred_loss={total_pred/max(count,1):.6f} | ball_loss={total_ball/max(count,1):.6f} | "
            f"val_MAE={val['mae']:.6f} | val_RMSE={val['rmse']:.6f} | "
            f"val_NSE={val['nse']:.6f} | val_KGE={val['kge']:.6f} | epoch_time={time.perf_counter()-start:.2f}s"
        )
        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "val_metrics": val, "epoch": epoch, "node_to_ball": node_np,
                "A_ball": A_ball_np, "A_phy": A_phy_np, "A_sim": A_sim_np,
                "ball_sizes": sizes, "ball_quality": quality_np,
                "hydro_dist_to_ball": hydro_np,
            }, best_path)
            print(f"Saved best model to {best_path}")

    print(
        f"Average training time over {len(training_epoch_times)} epochs: "
        f"{sum(training_epoch_times) / max(len(training_epoch_times), 1):.4f}s/epoch "
        f"| total_training_time={sum(training_epoch_times):.4f}s "
        f"| excludes_validation_and_test=True"
    )

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.reset_graph_propagation_timer()
    model.set_graph_propagation_profiling(True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_start = time.perf_counter()
    test, test_horizons = evaluate_horizons(
        model, test_loader, device, node, hydro, A_ball, quality, horizons,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.set_graph_propagation_profiling(False)
    elapsed = time.perf_counter() - test_start
    samples = len(test_loader.dataset)
    print(f"Test Overall | MAE={test['mae']:.6f} | RMSE={test['rmse']:.6f} | NSE={test['nse']:.6f} | KGE={test['kge']:.6f}")
    print(f"Test inference time: total={elapsed:.4f}s | per_sample={elapsed/max(samples,1):.6f}s ({1000*elapsed/max(samples,1):.3f} ms/sample) | samples={samples}")
    print_graph_propagation_profile(model, A_ball_np, samples)
    print_graph_statistics(node_np, A_phy_np, A_sim_np)
    for horizon in horizons:
        result = test_horizons[horizon]
        print(f"Test Horizon h={horizon} | MAE={result['mae']:.6f} | RMSE={result['rmse']:.6f} | NSE={result['nse']:.6f} | KGE={result['kge']:.6f}")


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "training.log").open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            run(args)


if __name__ == "__main__":
    main()
