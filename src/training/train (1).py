# src/training/train.py

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time

from src.data_handling.data import load_dataset
from src.data_handling.decomposition import split_trend_residual
from src.data_handling.residual_dataset import ResidualWindowDataset
from src.data_handling.trend_dataset import TrendWindowDataset
from src.models.model import DiffusionModel
from src.models.diffusion_common.diffusion import cosine_beta_schedule

def train_diffusion_model(
    series_list,
    dataset_cls,
    config_section: dict,
    shared_training_cfg: dict,
    device: torch.device,
    alpha_bars: torch.Tensor,
    save_prefix: str
):
    """
    Train a DiffusionModel on series_list using dataset_cls.
    Reads all dataset‐specific & model hyperparameters from config_section,
    and shared training settings from shared_training_cfg.
    Checkpoints are saved with save_prefix (e.g., "residual" or "trend").
    """
    # Dataset hyperparameters
    window_size = config_section['window_size']
    # If this is TrendWindowDataset, also read sma_window
    sma_window = config_section.get('sma_window', 1)

    # Instantiate the dataset, passing any extra args from config_section
    if dataset_cls is TrendWindowDataset:
        dataset = dataset_cls(
            series_list=series_list,
            window_size=window_size,
            sma_window=sma_window
        )
    else:  # ResidualWindowDataset
        dataset = dataset_cls(
            series_list=series_list,
            window_size=window_size,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=shared_training_cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=shared_training_cfg.get('num_workers', 4)
    )

    # Model hyperparameters
    model_cfg = config_section['model']
    model = DiffusionModel(
        window_size=window_size,
        feature_dim=series_list[0].shape[-1],
        base_channels=model_cfg.get('base_channels', 64),
        emb_dim=model_cfg.get('time_emb_dim', 128),
        n_res_blocks_per_level=model_cfg.get('n_res_blocks', 4)
    ).to(device)

    # Optimizer hyperparameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(shared_training_cfg['lr'])
    )

    # Training settings
    epochs = shared_training_cfg['epochs']
    T = shared_training_cfg['diffusion_steps']
    batch_size = shared_training_cfg['batch_size']
    save_interval = shared_training_cfg.get('save_interval', 10)
    ckpt_dir = shared_training_cfg.get('ckpt_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ────────── Diagnostic Report ──────────
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*60)
    print(f"Starting {save_prefix.capitalize()} Model Training")
    print(f"  Number of raw series provided: {len(series_list)}")
    if dataset_cls is TrendWindowDataset:
        print(f"  (Trend)    window_size = {window_size}, sma_window = {sma_window}")
    else:
        print(f"  (Residual) window_size = {window_size}")
    print(f"  ⇒ Dataset length (num. windows) = {len(dataset)}")
    print(f"  batch_size = {batch_size}, epochs = {epochs}, diffusion_steps = {T}")
    print(f"  Model parameter count: {total_params:,}")
    print(f"  Checkpoints will be saved to: {ckpt_dir}")
    print("="*60 + "\n")

    # ───── Epoch loop with tqdm ─────
    for epoch in tqdm(range(1, epochs + 1),
                      desc=f"{save_prefix.capitalize()} Epochs",
                      unit="ep"):
        model.train()
        running_loss = 0.0
        running_denoise = 0.0
        running_stat = 0.0

        # Batch‐level progress bar
        batch_bar = tqdm(dataloader,
                         desc=f" Epoch {epoch}/{epochs} batches",
                         leave=False,
                         unit="batch",
                         miniters=10)

        total_data_time = 0.0
        total_forward_time = 0.0

        for batch in batch_bar:
            t0 = time.time()

            # Data loading + transfer timing
            x_raw = batch['window'].to(device)          # (B, W, D)
            start_idx = batch['start_idx'].to(device)   # (B,)
            series_len = batch['series_len'].to(device) # (B,)
            data_time = time.time() - t0
            total_data_time += data_time

            B = x_raw.size(0)
            t = torch.randint(0, T, (B,), device=device)  # (B,)
            noise = torch.randn_like(x_raw)               # (B, W, D)
            alpha_bar_t = alpha_bars[t]                   # (B,)

            t1 = time.time()
            optimizer.zero_grad()
            total_loss, denoise_loss, stat_loss, \
                mu_true, sigma_true, mu_pred, sigma_pred = \
                model(x_raw, start_idx, series_len, t,
                      noise=noise, alpha_bar_t=alpha_bar_t)
            total_loss.backward()
            optimizer.step()
            forward_time = time.time() - t1
            total_forward_time += forward_time

            running_loss += total_loss.item() * B
            running_denoise += denoise_loss.item() * B
            running_stat += stat_loss.item() * B

            # Update batch‐level progress with key stats
            batch_bar.set_postfix({
                "loss":    f"{(running_loss/len(dataset)):.4e}",
                "data(s)": f"{data_time:.3f}",
                "fwd(s)":  f"{forward_time:.3f}"
            }, refresh=False)

        batch_bar.close()

        avg_loss = running_loss / len(dataset)
        avg_denoise = running_denoise / len(dataset)
        avg_stat = running_stat / len(dataset)
        print(
            f"Epoch {epoch}/{epochs} — "
            f"Total Loss: {avg_loss:.6f}  "
            f"(Denoise: {avg_denoise:.6f}  Stat: {avg_stat:.6f})  "
            f"[data-loading: {total_data_time:.2f}s, fwd/back: {total_forward_time:.2f}s]"
        )

        # Save checkpoint at intervals
        if epoch % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{save_prefix}_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    # Final save
    final_path = os.path.join(ckpt_dir, f"{save_prefix}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"{save_prefix.capitalize()} training complete. Model saved to {final_path}\n")


def main(config):
    # Device & seed
    device = torch.device(
        config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    torch.manual_seed(config.get('seed', 42))

    # Load & decompose
    data = load_dataset(config['name'])
    trend_list, resid_list = split_trend_residual(data)

    # Precompute diffusion schedule
    T = config['diffusion']['steps']
    betas, alphas, alpha_bars = cosine_beta_schedule(T)
    alpha_bars = alpha_bars.to(device)

    # Shared training hyperparameters
    shared_training_cfg = {
        'batch_size': config['training']['batch_size'],
        'lr': config['training']['lr'],
        'epochs': config['training']['epochs'],
        'num_workers': config['training'].get('num_workers', 4),
        'save_interval': config['training'].get('save_interval', 10),
        'ckpt_dir': config['training'].get('ckpt_dir', 'checkpoints'),
        'diffusion_steps': T
    }

    # 1) Residual model
    train_diffusion_model(
        series_list=resid_list,
        dataset_cls=ResidualWindowDataset,
        config_section=config['residual'],
        shared_training_cfg=shared_training_cfg,
        device=device,
        alpha_bars=alpha_bars,
        save_prefix="residual"
    )

    # 2) Trend model
    train_diffusion_model(
        series_list=trend_list,
        dataset_cls=TrendWindowDataset,
        config_section=config['trend'],
        shared_training_cfg=shared_training_cfg,
        device=device,
        alpha_bars=alpha_bars,
        save_prefix="trend"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
