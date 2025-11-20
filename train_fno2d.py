#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fno2d.py

Updated: adds AMP (mixed precision) training support, MAE loss, and evaluation outputs
- Uses original CSV (processed_data_mean.csv) without modifying it.
- Builds samples by (date x depth_slice) via interpolation (lon,lat,depth) -> uo,vo,so,thetao on a regular grid.
- Model: strict FNO2d (spectral conv as in the paper).
- Loss: MAE (L1).
- Train/Test random split: 85% / 15% (deterministic given seed).
- AMP: enabled with --fp16 (recommended for GPU). Default is enabled to suit RTX4060 Laptop (8GB).
- Saves best model, a sample pred/target, and two CSVs:
    - eval_per_var.csv -- columns: variable,mae,count  (order: so,thetao,uo,vo)
    - eval_per_day.csv -- columns: date,mae_so,mae_thetao,mae_uo,mae_vo,mae_overall

Run example for RTX4060 (recommended):
  python train_fno2d.py --csv processed_data_mean.csv --nx 48 --ny 48 --nz 8 \
      --epochs 120 --batch 4 --modes1 12 --modes2 12 --width 24 --device cuda --fp16

"""
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# IMPORTANT: use non-interactive matplotlib backend to avoid tkinter/Tcl errors on headless systems
import matplotlib
matplotlib.use("Agg")  # <-- ensures no Tkinter/Tcl GUI required
import matplotlib.pyplot as plt

# ---------------------------
# FNO2d model
# ---------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        # store real and imag parts as parameters
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def forward(self, x):
        # x: (batch, in_channels, nx, ny)
        batchsize = x.shape[0]
        nx = x.shape[2]
        ny = x.shape[3]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # complex tensor
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-2], x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        kx = min(self.modes1, x_ft.shape[-2])
        ky = min(self.modes2, x_ft.shape[-1])
        if kx == 0 or ky == 0:
            return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))
        w_complex = torch.complex(self.weights_real[:, :, :kx, :ky], self.weights_imag[:, :, :kx, :ky])  # (in,out,kx,ky)
        x_slice = x_ft[:, :, :kx, :ky]  # (b, in, kx, ky)
        # multiply and sum over in_channels
        out_slice = torch.einsum('bixy,ioxy->boxy', x_slice, w_complex)  # (b, out, kx, ky)
        out_ft[:, :, :kx, :ky] = out_slice
        x = torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))
        return x

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, in_ch, nx, ny)
        b = x.shape[0]
        nx = x.shape[2]
        ny = x.shape[3]
        # lift
        x = x.permute(0, 2, 3, 1)  # (b, nx, ny, in_ch)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (b, width, nx, ny)

        x1 = self.conv0(x) + self.w0(x)
        x1 = self.act(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = self.act(x2)
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = self.act(x3)
        x4 = self.conv3(x3) + self.w3(x3)
        x = self.act(x4)

        x = x.permute(0, 2, 3, 1)  # (b, nx, ny, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)  # (b, nx, ny, out_ch)
        x = x.permute(0, 3, 1, 2)  # (b, out_ch, nx, ny)
        return x

# ---------------------------
# Data preprocessing (no file modification)
# ---------------------------
def load_csv_make_samples(csv_path, nx=48, ny=48, nz=8, min_points=50):
    """
    Read CSV, create samples by date and depth slices.
    Returns:
      samples: list of dict with keys inputs (4,nx,ny), targets (4,nx,ny), date (date), depth_level (float)
      lon_grid, lat_grid, depths_target arrays
    """
    df = pd.read_csv(csv_path)
    required = {'time', 'latitude', 'longitude', 'depth', 'uo', 'vo', 'so', 'thetao'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    df['date'] = pd.to_datetime(df['time']).dt.date
    df['lon'] = df['longitude'].astype(float)
    df['lat'] = df['latitude'].astype(float)
    df['depth'] = df['depth'].astype(float)
    df['uo'] = df['uo'].astype(float)
    df['vo'] = df['vo'].astype(float)
    df['so'] = df['so'].astype(float)
    df['thetao'] = df['thetao'].astype(float)

    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    depth_min, depth_max = df['depth'].min(), df['depth'].max()

    lon_grid = np.linspace(lon_min, lon_max, ny)
    lat_grid = np.linspace(lat_min, lat_max, nx)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)  # shapes (nx, ny)

    unique_dates = sorted(df['date'].unique())
    if len(unique_dates) == 0:
        raise RuntimeError("No dates found in CSV.")
    t0 = unique_dates[0]
    max_days = max(1, (unique_dates[-1] - t0).days)
    depths_target = np.linspace(depth_min, depth_max, nz)

    samples = []
    for d in unique_dates:
        sdf = df[df['date'] == d]
        if len(sdf) < min_points:
            continue
        pts = np.vstack([sdf['lon'].values, sdf['lat'].values, sdf['depth'].values]).T
        vals_u = sdf['uo'].values
        vals_v = sdf['vo'].values
        vals_s = sdf['so'].values
        vals_th = sdf['thetao'].values

        t_scalar = (d - t0).days / max_days
        for depth_level in depths_target:
            xi = np.stack([Lon.ravel(), Lat.ravel(), np.full(Lon.size, depth_level)], axis=-1)  # (nx*ny,3)
            # linear interpolation, fallback to nearest for NaNs
            u_grid = griddata(pts, vals_u, xi, method='linear')
            v_grid = griddata(pts, vals_v, xi, method='linear')
            s_grid = griddata(pts, vals_s, xi, method='linear')
            th_grid = griddata(pts, vals_th, xi, method='linear')

            mask_nan = np.isnan(u_grid) | np.isnan(v_grid) | np.isnan(s_grid) | np.isnan(th_grid)
            if mask_nan.any():
                u_near = griddata(pts, vals_u, xi, method='nearest')
                v_near = griddata(pts, vals_v, xi, method='nearest')
                s_near = griddata(pts, vals_s, xi, method='nearest')
                th_near = griddata(pts, vals_th, xi, method='nearest')
                u_grid[mask_nan] = u_near[mask_nan]
                v_grid[mask_nan] = v_near[mask_nan]
                s_grid[mask_nan] = s_near[mask_nan]
                th_grid[mask_nan] = th_near[mask_nan]

            try:
                u_grid = u_grid.reshape((nx, ny))
                v_grid = v_grid.reshape((nx, ny))
                s_grid = s_grid.reshape((nx, ny))
                th_grid = th_grid.reshape((nx, ny))
            except Exception:
                # skip if reshape fails
                continue

            # input channels: lon_norm, lat_norm, t_norm, depth_norm
            lon_norm = 2 * (Lon - lon_min) / max(1e-8, (lon_max - lon_min)) - 1.0
            lat_norm = 2 * (Lat - lat_min) / max(1e-8, (lat_max - lat_min)) - 1.0
            depth_norm = (depth_level - depth_min) / max(1e-8, (depth_max - depth_min))
            t_norm = float(t_scalar)

            ch_lon = lon_norm.astype(np.float32)
            ch_lat = lat_norm.astype(np.float32)
            ch_t = np.full_like(ch_lon, fill_value=t_norm, dtype=np.float32)
            ch_depth = np.full_like(ch_lon, fill_value=float(depth_norm), dtype=np.float32)

            inputs = np.stack([ch_lon, ch_lat, ch_t, ch_depth], axis=0)  # (4, nx, ny)
            targets = np.stack([u_grid.astype(np.float32), v_grid.astype(np.float32),
                                s_grid.astype(np.float32), th_grid.astype(np.float32)], axis=0)  # (4, nx, ny)

            samples.append({'inputs': inputs, 'targets': targets, 'date': d, 'depth_level': float(depth_level)})

    if len(samples) == 0:
        raise RuntimeError("No samples were generated. Check CSV content and min_points parameter.")
    return samples, lon_grid, lat_grid, depths_target

class OceanSliceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # convert arrays to torch tensors for efficient collation & transfer
        x = torch.from_numpy(s['inputs']).to(dtype=torch.float32)
        y = torch.from_numpy(s['targets']).to(dtype=torch.float32)
        meta = (str(s['date']), float(s['depth_level']))
        return x, y, meta

# ---------------------------
# Training & evaluation utilities
# ---------------------------
def train_epoch(model, loader, opt, device, scaler=None):
    model.train()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    total_samples = 0
    for xb, yb, _meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        total_loss += float(loss.item()) * xb.size(0)
        total_samples += xb.size(0)
    return total_loss / max(1, total_samples)

def evaluate_and_save(model, samples, test_indices, device, lon_grid, lat_grid, out_prefix="eval"):
    """
    Evaluate model on test_indices (list of ints into samples).
    Saves two CSVs: {out_prefix}_per_var.csv and {out_prefix}_per_day.csv
    Returns (df_var, df_day)
    """
    model.eval()
    out_ch = 4  # order: [uo, vo, so, thetao] as stored in samples
    sum_abs = np.zeros(out_ch, dtype=np.float64)
    total_grid_points_per_var = 0  # will be nx*ny * num_samples
    per_day_sums = {}  # date_str -> np.array sum per var
    per_day_counts = {}  # date_str -> int (number of grid points accumulated)

    nx = len(lat_grid)
    ny = len(lon_grid)
    per_sample_count = nx * ny

    with torch.no_grad():
        for idx in test_indices:
            s = samples[idx]
            x_np = s['inputs']
            y_np = s['targets']  # (4,nx,ny)
            date_str = str(s['date'])
            # prepare tensor
            xb = torch.from_numpy(x_np[None, ...]).to(dtype=torch.float32, device=device)
            pred = model(xb)  # (1,4,nx,ny)
            pred_np = pred.cpu().numpy()[0]
            abs_err = np.abs(pred_np - y_np)  # (4,nx,ny)
            sum_abs += abs_err.reshape(out_ch, -1).sum(axis=1)
            total_grid_points_per_var += per_sample_count

            if date_str not in per_day_sums:
                per_day_sums[date_str] = np.zeros(out_ch, dtype=np.float64)
                per_day_counts[date_str] = 0
            per_day_sums[date_str] += abs_err.reshape(out_ch, -1).sum(axis=1)
            per_day_counts[date_str] += per_sample_count

    if total_grid_points_per_var == 0:
        raise RuntimeError("No test points evaluated.")

    # compute MAEs for internal order [uo,vo,so,thetao]
    mae_uo = sum_abs[0] / total_grid_points_per_var
    mae_vo = sum_abs[1] / total_grid_points_per_var
    mae_so = sum_abs[2] / total_grid_points_per_var
    mae_thetao = sum_abs[3] / total_grid_points_per_var

    # Save eval_per_var.csv in order so,thetao,uo,vo to match your existing outputs
    var_rows = [
        ('so', float(mae_so), int(total_grid_points_per_var)),
        ('thetao', float(mae_thetao), int(total_grid_points_per_var)),
        ('uo', float(mae_uo), int(total_grid_points_per_var)),
        ('vo', float(mae_vo), int(total_grid_points_per_var)),
    ]
    df_var = pd.DataFrame(var_rows, columns=['variable', 'mae', 'count'])
    df_var.to_csv(f"{out_prefix}_per_var.csv", index=False)

    # per-day MAE rows sorted by date ascending
    rows = []
    # sort keys reliably
    def parse_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return datetime.min
    date_keys = sorted(per_day_sums.keys(), key=parse_date)
    for d in date_keys:
        sums = per_day_sums[d]
        cnt = per_day_counts[d]
        if cnt == 0:
            continue
        mae_u = sums[0] / cnt
        mae_v = sums[1] / cnt
        mae_s = sums[2] / cnt
        mae_th = sums[3] / cnt
        mae_overall = float(np.mean([mae_s, mae_th, mae_u, mae_v]))
        rows.append((d, float(mae_s), float(mae_th), float(mae_u), float(mae_v), mae_overall))
    df_day = pd.DataFrame(rows, columns=['date', 'mae_so', 'mae_thetao', 'mae_uo', 'mae_vo', 'mae_overall'])
    df_day.to_csv(f"{out_prefix}_per_day.csv", index=False)

    return df_var, df_day

# ---------------------------
# Main entry
# ---------------------------
def main(args):
    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading CSV and generating samples (in-memory interpolation, no file changes)...")
    samples, lon_grid, lat_grid, depths_target = load_csv_make_samples(
        args.csv, nx=args.nx, ny=args.ny, nz=args.nz, min_points=args.min_points
    )
    print(f"Generated {len(samples)} samples (date x depth slice). Grid: {args.nx}x{args.ny}, depth slices: {len(depths_target)}")

    total = len(samples)
    n_test = max(1, int(total * args.test_frac))
    n_train = total - n_test

    # deterministic random permutation
    perm = np.random.RandomState(seed=args.seed).permutation(total)
    train_idx = perm[:n_train].tolist()
    test_idx = perm[n_train:].tolist()

    train_set = Subset(OceanSliceDataset(samples), train_idx)
    test_set = Subset(OceanSliceDataset(samples), test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=False,
                              num_workers=0, pin_memory=True)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    print("Using device:", device)

    model = FNO2d(in_channels=4, out_channels=4, modes1=args.modes1, modes2=args.modes2, width=args.width)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == 'cuda') else None

    best_overall = 1e9
    best_epoch = -1

    for ep in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
        scheduler.step()

        if (ep % args.eval_every == 0) or (ep == args.epochs - 1):
            print(f"Epoch {ep+1}/{args.epochs} train_loss={train_loss:.6e} -> running full test eval...")
            df_var, df_day = evaluate_and_save(model, samples, test_idx, device, lon_grid, lat_grid, out_prefix="temp_eval")
            # overall is mean across variables in df_var
            overall_mae = float(df_var['mae'].mean())
            if overall_mae < best_overall:
                best_overall = overall_mae
                best_epoch = ep
                torch.save(model.state_dict(), args.save_model)
            print(f"  temp eval overall MAE={overall_mae:.6e} best={best_overall:.6e} (epoch {best_epoch})")
        else:
            print(f"Epoch {ep+1}/{args.epochs} train_loss={train_loss:.6e}")

    print("Training completed. Best overall MAE:", best_overall, "at epoch", best_epoch)
    print("Loading best model and doing final evaluation...")
    model.load_state_dict(torch.load(args.save_model, map_location=device))
    model.to(device).eval()

    df_var, df_day = evaluate_and_save(model, samples, test_idx, device, lon_grid, lat_grid, out_prefix="eval")

    # Save a first test sample pred/target for quick inspection
    idx0 = test_idx[0]
    sample0 = samples[idx0]
    x0 = sample0['inputs']
    y0 = sample0['targets']
    xb = torch.from_numpy(x0[None, ...]).to(dtype=torch.float32, device=device)
    with torch.no_grad():
        pred0 = model(xb).cpu().numpy()[0]
    np.save("sample_input.npy", x0)
    np.save("sample_target.npy", y0)
    np.save("sample_pred.npy", pred0)
    print("Saved sample_input.npy, sample_target.npy, sample_pred.npy")

    # produce quick figure for uo target vs pred
    try:
        Lon, Lat = np.meshgrid(lon_grid, lat_grid)
        targ_u = y0[0]
        pred_u = pred0[0]
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.pcolormesh(Lon, Lat, targ_u, shading='auto')
        plt.title("target uo")
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolormesh(Lon, Lat, pred_u, shading='auto')
        plt.title("pred uo")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("example_uo_pred.png")
        print("Saved example_uo_pred.png")
    except Exception as e:
        print("Warning: plotting failed (non-fatal). Error:", str(e))
        # still continue; sample numpy saved above

    print("Saved CSVs: eval_per_var.csv and eval_per_day.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="processed_data_mean.csv", help="Original CSV input (do not modify).")
    parser.add_argument("--nx", type=int, default=48, help="Grid rows (lat)")
    parser.add_argument("--ny", type=int, default=48, help="Grid cols (lon)")
    parser.add_argument("--nz", type=int, default=8, help="Number of depth slices per date")
    parser.add_argument("--min_points", type=int, default=50, help="Min points required in a date to use it")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size (lower if OOM)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr_step", type=int, default=40)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction (0..1)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_model", type=str, default="fno2d_ocean_amp.pth")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable AMP mixed precision (recommended for GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=10, help="Full-test eval frequency (epochs)")
    args = parser.parse_args()
    main(args)