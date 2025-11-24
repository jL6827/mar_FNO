#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fno2d.py

FNO2d training + evaluation script with optional AMP and explicit train/test CSV support.

This version includes a fallback for the case where the user provided explicit
train/test CSVs but those CSVs were created by random row-wise splitting
(i.e., many dates have too few points). In that case the script will automatically
combine train+test CSVs, build samples by date+depth from the combined data, and
then perform a random 80/20 sample-level split to produce train/test sets.

CSV header supported (your format):
  date,t_numeric,latitude,longitude,depth,so,thetao,uo,vo

Modes:
  1) Explicit train/test CSVs (preferred for fixed splits):
     --train_csv processed_data_mean_train.csv --test_csv processed_data_mean_test.csv

  2) Single CSV (legacy) with random 80/20 split:
     --csv processed_data_mean.csv  (uses --test_frac, default 0.20)

If explicit CSVs are provided but result in no test samples (e.g. because
each date has too few points), the script will fall back to combined random-split mode.

Recommended (RTX 4060 Laptop 8GB - conservative):
  python train_fno2d.py --train_csv processed_data_mean_train.csv --test_csv processed_data_mean_test.csv \
      --nx 48 --ny 48 --nz 8 --epochs 120 --batch 4 --modes1 12 --modes2 12 --width 24 --device cuda --fp16

Quick evaluation-only (if you already have model saved):
  python train_fno2d.py --train_csv processed_data_mean_train.csv --test_csv processed_data_mean_test.csv \
      --nx 48 --ny 48 --nz 8 --epochs 0 --batch 1 --device cuda --save_model fno2d_ocean_amp.pth

"""
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# Use non-interactive matplotlib backend to avoid tkinter/Tcl errors on headless systems
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Model: SpectralConv2d + FNO2d
# ---------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def forward(self, x):
        # x: (batch, in_chan, nx, ny)
        b, c, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # complex tensor (b, c, nx, nyf)
        out_ft = torch.zeros(b, self.out_channels, x_ft.shape[-2], x_ft.shape[-1],
                             dtype=x_ft.dtype, device=x.device)
        kx = min(self.modes1, x_ft.shape[-2])
        ky = min(self.modes2, x_ft.shape[-1])
        if kx == 0 or ky == 0:
            return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))
        w_complex = torch.complex(self.weights_real[:, :, :kx, :ky], self.weights_imag[:, :, :kx, :ky])  # (in,out,kx,ky)
        x_slice = x_ft[:, :, :kx, :ky]  # (b, in, kx, ky)
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
        # x: (b, in_ch, nx, ny)
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
# Data preprocessing utilities
# ---------------------------
def compute_global_domain(csv_paths):
    """Compute global lon/lat/depth bounds from list of CSV file paths."""
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            # try look in script directory as fallback
            alt = os.path.join(os.path.dirname(__file__), os.path.basename(p))
            if os.path.exists(alt):
                p = alt
            else:
                raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    # accept either 'longitude'/'latitude' naming
    if 'longitude' in df_all.columns:
        df_all['lon'] = df_all['longitude'].astype(float)
    elif 'lon' in df_all.columns:
        df_all['lon'] = df_all['lon'].astype(float)
    else:
        raise ValueError("No longitude column found in CSVs.")
    if 'latitude' in df_all.columns:
        df_all['lat'] = df_all['latitude'].astype(float)
    elif 'lat' in df_all.columns:
        df_all['lat'] = df_all['lat'].astype(float)
    else:
        raise ValueError("No latitude column found in CSVs.")
    if 'depth' in df_all.columns:
        df_all['depth_f'] = df_all['depth'].astype(float)
    else:
        raise ValueError("No depth column found in CSVs.")
    lon_min, lon_max = df_all['lon'].min(), df_all['lon'].max()
    lat_min, lat_max = df_all['lat'].min(), df_all['lat'].max()
    depth_min, depth_max = df_all['depth_f'].min(), df_all['depth_f'].max()
    return lon_min, lon_max, lat_min, lat_max, depth_min, depth_max

def make_samples_from_df(df, lon_grid, lat_grid, depths_target, min_points=50):
    """
    Given a dataframe and fixed grid/depths, generate samples (in-memory interpolation).
    Returns list of sample dicts with keys 'inputs','targets','date','depth_level'.
    """
    df = df.copy()
    # determine date column: prefer 'date' if present, else 'time' or fallback to generated
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'time' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['time']).dt.date
    else:
        # if no date-like column, create a pseudo-date grouping by integer division of row index to get slices
        # but primary fallback is to let caller handle random splitting; here set each row's date as its index
        df['date_parsed'] = pd.to_datetime(df.index).date

    # ensure lon/lat/depth numeric columns exist (accept 'longitude'/'latitude' naming)
    if 'longitude' in df.columns:
        df['lon'] = df['longitude'].astype(float)
    elif 'lon' in df.columns:
        df['lon'] = df['lon'].astype(float)
    else:
        raise ValueError("No longitude-like column found in CSV.")

    if 'latitude' in df.columns:
        df['lat'] = df['latitude'].astype(float)
    elif 'lat' in df.columns:
        df['lat'] = df['lat'].astype(float)
    else:
        raise ValueError("No latitude-like column found in CSV.")

    if 'depth' in df.columns:
        df['depth_f'] = df['depth'].astype(float)
    else:
        raise ValueError("No depth column found in CSV.")

    # ensure target variables exist
    for col in ['uo', 'vo', 'so', 'thetao']:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    unique_dates = sorted(df['date_parsed'].unique())
    if len(unique_dates) == 0:
        return []

    t0 = unique_dates[0]
    max_days = max(1, (unique_dates[-1] - t0).days)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    nx, ny = Lat.shape[0], Lon.shape[1]
    samples = []
    for d in unique_dates:
        sdf = df[df['date_parsed'] == d]
        if len(sdf) < min_points:
            # skip date if too few raw points for robust interpolation
            continue
        pts = np.vstack([sdf['lon'].values, sdf['lat'].values, sdf['depth_f'].values]).T
        vals_u = sdf['uo'].values
        vals_v = sdf['vo'].values
        vals_s = sdf['so'].values
        vals_th = sdf['thetao'].values
        t_scalar = (d - t0).days / max_days

        for depth_level in depths_target:
            xi = np.stack([Lon.ravel(), Lat.ravel(), np.full(Lon.size, depth_level)], axis=-1)
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
                continue

            # normalization channels (lon/lat -> [-1,1], t_norm in [0,1], depth_norm in [0,1])
            lon_min = lon_grid.min()
            lon_max = lon_grid.max()
            lat_min = lat_grid.min()
            lat_max = lat_grid.max()
            lon_norm = 2 * (Lon - lon_min) / max(1e-8, (lon_max - lon_min)) - 1.0
            lat_norm = 2 * (Lat - lat_min) / max(1e-8, (lat_max - lat_min)) - 1.0
            depth_min = depths_target[0]
            depth_max = depths_target[-1]
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
    return samples

class OceanSliceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.from_numpy(s['inputs']).to(dtype=torch.float32)
        y = torch.from_numpy(s['targets']).to(dtype=torch.float32)
        meta = (str(s['date']), float(s['depth_level']))
        return x, y, meta

# ---------------------------
# Train / Eval functions
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
    Evaluate model on samples[test_indices] and save eval CSVs.
    samples: list of sample dicts (with 'inputs','targets','date')
    test_indices: list of indices into samples for evaluation
    """
    model.eval()
    out_ch = 4  # internal order [uo,vo,so,thetao]
    sum_abs = np.zeros(out_ch, dtype=np.float64)
    total_grid_points_per_var = 0
    per_day_sums = {}
    per_day_counts = {}

    nx = len(lat_grid)
    ny = len(lon_grid)
    per_sample_count = nx * ny

    with torch.no_grad():
        for idx in test_indices:
            s = samples[idx]
            x_np = s['inputs']
            y_np = s['targets']
            date_str = str(s['date'])
            xb = torch.from_numpy(x_np[None, ...]).to(dtype=torch.float32, device=device)
            pred = model(xb)
            pred_np = pred.cpu().numpy()[0]
            abs_err = np.abs(pred_np - y_np)
            sum_abs += abs_err.reshape(out_ch, -1).sum(axis=1)
            total_grid_points_per_var += per_sample_count
            if date_str not in per_day_sums:
                per_day_sums[date_str] = np.zeros(out_ch, dtype=np.float64)
                per_day_counts[date_str] = 0
            per_day_sums[date_str] += abs_err.reshape(out_ch, -1).sum(axis=1)
            per_day_counts[date_str] += per_sample_count

    if total_grid_points_per_var == 0:
        raise RuntimeError("No test points evaluated.")

    # compute MAE in internal order [uo,vo,so,thetao]
    mae_uo = sum_abs[0] / total_grid_points_per_var
    mae_vo = sum_abs[1] / total_grid_points_per_var
    mae_so = sum_abs[2] / total_grid_points_per_var
    mae_thetao = sum_abs[3] / total_grid_points_per_var

    # Save eval_per_var.csv in order so,thetao,uo,vo to match prior format
    var_rows = [
        ('so', float(mae_so), int(total_grid_points_per_var)),
        ('thetao', float(mae_thetao), int(total_grid_points_per_var)),
        ('uo', float(mae_uo), int(total_grid_points_per_var)),
        ('vo', float(mae_vo), int(total_grid_points_per_var)),
    ]
    df_var = pd.DataFrame(var_rows, columns=['variable', 'mae', 'count'])
    df_var.to_csv(f"{out_prefix}_per_var.csv", index=False)

    # per-day CSV
    rows = []
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
# Main
# ---------------------------
def main(args):
    # reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # explicit mode if both train_csv and test_csv provided
    if args.train_csv and args.test_csv:
        print("Mode: explicit train/test CSVs")
        print("Train CSV:", args.train_csv)
        print("Test CSV :", args.test_csv)
        # compute global domain from both to ensure consistent grid
        try:
            lon_min, lon_max, lat_min, lat_max, depth_min, depth_max = compute_global_domain([args.train_csv, args.test_csv])
        except FileNotFoundError as e:
            # try fallback to script dir already handled in compute_global_domain
            raise

        lon_grid = np.linspace(lon_min, lon_max, args.ny)
        lat_grid = np.linspace(lat_min, lat_max, args.nx)
        depths_target = np.linspace(depth_min, depth_max, args.nz)

        # load dataframes
        df_train = pd.read_csv(args.train_csv)
        df_test = pd.read_csv(args.test_csv)

        # build samples from each CSV separately (this is the original explicit behavior)
        train_samples = make_samples_from_df(df_train, lon_grid, lat_grid, depths_target, min_points=args.min_points)
        test_samples = make_samples_from_df(df_test, lon_grid, lat_grid, depths_target, min_points=args.min_points)

        # If test_samples is empty (common when the CSVs are random row-wise splits),
        # fall back to combined random-split mode: build samples from concatenated df and then randomly split samples.
        if len(test_samples) == 0:
            print("Warning: no test samples generated from test CSV (likely because per-date point counts < min_points).")
            print("Falling back to combined random-split mode: building samples from train+test combined and performing a random 80/20 split of samples.")
            df_combined = pd.concat([df_train, df_test], ignore_index=True)
            # recompute domain using combined dataframe (already similar), but reuse lon_grid/lat_grid/depths_target
            combined_samples = make_samples_from_df(df_combined, lon_grid, lat_grid, depths_target, min_points= max(1, args.min_points//2))
            if len(combined_samples) == 0:
                raise RuntimeError("Fallback combined sample generation failed - no samples created. Consider lowering --min_points further.")
            print(f"Combined samples created: {len(combined_samples)}. Now performing random 80/20 split.")
            total = len(combined_samples)
            n_test = max(1, int(total * 0.20))  # 80/20 fixed split for fallback mode
            n_train = total - n_test
            perm = np.random.RandomState(seed=args.seed).permutation(total)
            train_idx = perm[:n_train].tolist()
            test_idx = perm[n_train:].tolist()
            train_samples = [combined_samples[i] for i in train_idx]
            test_samples = [combined_samples[i] for i in test_idx]
            print(f"Fallback split: train {len(train_samples)} samples, test {len(test_samples)} samples.")
        else:
            print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

        # create DataLoader for training from train_samples
        train_dataset = OceanSliceDataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

        samples_for_eval = test_samples
        test_indices = list(range(len(test_samples)))

    else:
        # legacy single CSV mode with random split (default 80/20)
        print("Mode: single CSV with random split (legacy). CSV:", args.csv)
        samples, lon_grid, lat_grid, depths_target = make_samples_legacy(args.csv, nx=args.nx, ny=args.ny, nz=args.nz, min_points=args.min_points)
        total = len(samples)
        if total == 0:
            raise RuntimeError("No samples generated from CSV.")
        n_test = max(1, int(total * args.test_frac))
        n_train = total - n_test
        perm = np.random.RandomState(seed=args.seed).permutation(total)
        train_idx = perm[:n_train].tolist()
        test_idx = perm[n_train:].tolist()
        print(f"Total samples: {total}. Train: {len(train_idx)}, Test: {len(test_idx)}")
        train_loader = DataLoader(Subset(OceanSliceDataset(samples), train_idx), batch_size=args.batch, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
        samples_for_eval = samples
        test_indices = test_idx

    # build model and optimizer
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    print("Using device:", device)
    model = FNO2d(in_channels=4, out_channels=4, modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == 'cuda') else None

    best_overall = 1e9
    best_epoch = -1

    # training loop (skip if epochs == 0)
    if args.epochs > 0:
        for ep in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
            scheduler.step()
            if (ep % args.eval_every == 0) or (ep == args.epochs - 1):
                print(f"Epoch {ep+1}/{args.epochs} train_loss={train_loss:.6e} -> running full test eval...")
                df_var, df_day = evaluate_and_save(model, samples_for_eval, test_indices, device, lon_grid, lat_grid, out_prefix="temp_eval")
                overall_mae = float(df_var['mae'].mean())
                if overall_mae < best_overall:
                    best_overall = overall_mae
                    best_epoch = ep
                    torch.save(model.state_dict(), args.save_model)
                print(f"  temp eval overall MAE={overall_mae:.6e} best={best_overall:.6e} (epoch {best_epoch})")
            else:
                print(f"Epoch {ep+1}/{args.epochs} train_loss={train_loss:.6e}")
    else:
        print("Skipping training (epochs=0).")

    # load best model if available, otherwise use current model
    if os.path.exists(args.save_model):
        print("Loading model from", args.save_model)
        model.load_state_dict(torch.load(args.save_model, map_location=device))
        model.to(device).eval()
    else:
        print("Warning: saved model not found at", args.save_model, "- using current model weights for evaluation")

    # final eval and saving
    df_var, df_day = evaluate_and_save(model, samples_for_eval, test_indices, device, lon_grid, lat_grid, out_prefix="eval")

    # save example sample outputs (first test sample)
    ex_idx = test_indices[0]
    ex_sample = samples_for_eval[ex_idx]
    x0 = ex_sample['inputs']
    y0 = ex_sample['targets']
    xb = torch.from_numpy(x0[None, ...]).to(dtype=torch.float32, device=device)
    with torch.no_grad():
        pred0 = model(xb).cpu().numpy()[0]
    np.save("sample_input.npy", x0)
    np.save("sample_target.npy", y0)
    np.save("sample_pred.npy", pred0)
    print("Saved sample_input.npy, sample_target.npy, sample_pred.npy")

    # example plot (protected)
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

    print("Saved CSVs: eval_per_var.csv and eval_per_day.csv")
    return

# Legacy helper (kept for compatibility)
def make_samples_legacy(csv_path, nx=48, ny=48, nz=8, min_points=50):
    """Old mode: build grid and samples from a single CSV (kept for backward compatibility)."""
    if not os.path.exists(csv_path):
        # fallback to script dir
        alt = os.path.join(os.path.dirname(__file__), os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # handle potential 'date' column or 'time' column as used earlier
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'time' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['time']).dt.date
    else:
        df['date_parsed'] = pd.to_datetime(df.index).date

    df['lon'] = df['longitude'].astype(float) if 'longitude' in df.columns else df['lon'].astype(float)
    df['lat'] = df['latitude'].astype(float) if 'latitude' in df.columns else df['lat'].astype(float)
    df['depth_f'] = df['depth'].astype(float)
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    depth_min, depth_max = df['depth_f'].min(), df['depth_f'].max()
    lon_grid = np.linspace(lon_min, lon_max, ny)
    lat_grid = np.linspace(lat_min, lat_max, nx)
    depths_target = np.linspace(depth_min, depth_max, nz)
    samples = make_samples_from_df(df, lon_grid, lat_grid, depths_target, min_points=min_points)
    return samples, lon_grid, lat_grid, depths_target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="processed_data_mean.csv", help="Single CSV input (legacy random-split mode).")
    parser.add_argument("--train_csv", type=str, default="", help="Explicit train CSV (use with --test_csv for fixed split).")
    parser.add_argument("--test_csv", type=str, default="", help="Explicit test CSV (use with --train_csv for fixed split).")
    parser.add_argument("--nx", type=int, default=48, help="Grid rows (lat)")
    parser.add_argument("--ny", type=int, default=48, help="Grid cols (lon)")
    parser.add_argument("--nz", type=int, default=8, help="Number of depth slices per date")
    parser.add_argument("--min_points", type=int, default=50, help="Minimum raw points per date to use that date")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size (lower if OOM)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr_step", type=int, default=40)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--test_frac", type=float, default=0.20, help="Test fraction for legacy single-csv split (default 0.20 = 80/20)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_model", type=str, default="fno2d_ocean_amp.pth")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable AMP mixed precision (use --fp16 to enable on GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=10, help="Full-test eval frequency (epochs)")
    args = parser.parse_args()
    main(args)