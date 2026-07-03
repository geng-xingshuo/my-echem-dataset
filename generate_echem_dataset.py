#!/usr/bin/env python3
"""
generate_echem_dataset.py
Generate synthetic electrochemical plots (CV, charge-discharge, Nyquist, multi-line, scatter)
and produce paired ground-truth CSVs + metadata.

Usage:
  python generate_echem_dataset.py --num 100 --out ./generated_echem --seed 42

Outputs:
  generated_echem/
    images/              # png/jpeg plot images
    csvs/                # csv files per image (one or more per image, named <imgid>_curve<n>.csv)
    metadata.csv         # manifest with image -> csv mapping and flags
    generated_echem.zip  # optional archive if --zip provided
"""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import Image

plt.style.use('seaborn-whitegrid')

def ensure_dirs(out):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "csvs").mkdir(parents=True, exist_ok=True)

def save_csv_pairs(out_csv_dir, img_id, curves):
    # curves: list of dicts {id:int, x:np.array, y:np.array}
    files = []
    for c in curves:
        fname = f"{img_id}_curve{c['id']}.csv"
        path = out_csv_dir / fname
        np.savetxt(path, np.column_stack((c['x'], c['y'])), delimiter=",", header="x,y", comments='')
        files.append(str(path.name))
    return files

def add_legend_occlusion(ax, loc='upper right', size_fraction=0.12, color=None):
    # Draw a rectangle patch over the legend area to simulate occlusion
    # loc string ignored; just cover upper-right-ish area
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = (xlim[1] - xlim[0]) * size_fraction
    h = (ylim[1] - ylim[0]) * size_fraction
    x = xlim[1] - w - (xlim[1] - xlim[0]) * 0.02
    y = ylim[1] - h - (ylim[1] - ylim[0]) * 0.02
    if color is None:
        color='white'
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor=color, zorder=10)
    ax.add_patch(rect)

def simulate_cv(seed=None, cycles=2, points=1000, noise=0.01):
    t = np.linspace(-0.5, 0.5, points)
    # Base shape: combination of broad peaks and a slow baseline
    base = np.zeros_like(t)
    for k in range(cycles):
        base += 0.6 * np.exp(-((t - (k - cycles/2)*0.02)**2) * 60) * np.sin(2*np.pi*(k+1)*(t+0.1))
    baseline = 0.02 * t + 0.05 * np.sin(2*np.pi*0.5*t)
    y = base + baseline + noise * np.random.randn(points)
    return t, y

def simulate_cd(points=400, noise=0.01):
    x = np.linspace(0, 1, points)
    charge = 4.2 - 0.9 * x**1.1 + 0.02*np.sin(10*x) + noise*np.random.randn(points)
    discharge = 3.8 - 0.7 * x**1.05 + 0.02*np.cos(8*x) + noise*np.random.randn(points)
    return x, charge, discharge

def simulate_nyquist(points=500):
    freq = np.logspace(5, -1, points)
    R0 = 10 + np.random.rand()*5
    Rp = 50 + np.random.rand()*30
    C = 1e-5 * (0.5 + np.random.rand())
    w = 2 * np.pi * freq
    Zc = 1 / (1j * w * C)
    Zp = 1 / (1/Rp + 1/Zc)
    Z = R0 + Zp
    warburg = 0.5*(1 - 1j) / np.sqrt(freq)
    Z = Z + warburg * (0.5 + np.random.rand())
    x = Z.real
    y = -Z.imag
    return x, y

def simulate_multi_line(num_curves=3, points=600, noise=0.01):
    x = np.linspace(0, 1, points)
    curves = []
    for i in range(num_curves):
        phase = np.random.rand()*2*np.pi
        amp = 0.1 + np.random.rand()*0.5
        y = (0.5*i/num_curves) + amp * np.sin(2*np.pi*(1 + i*0.3)*x + phase) + 0.02*np.random.randn(points)
        curves.append({'id': i+1, 'x': x.copy(), 'y': y})
    return curves

def maybe_apply_twin_axis(ax, x, y1, y2):
    # create twin y-axis for same x, return twin flag
    twin_flag = random.random() < 0.15
    if twin_flag:
        ax2 = ax.twinx()
        ax2.plot(x, y2, color='tab:orange', linestyle='--', linewidth=1.2, label='twin')
        ax2.set_ylabel("Secondary")
    return twin_flag

def sample_style():
    # random style choices
    return {
        'dpi': random.choice([100, 150, 200]),
        'figsize': random.choice([(5,4), (6,4), (6,5)]),
        'font_size': random.choice([8,9,10,11,12]),
        'grid': random.random() < 0.9,
        'line_width': random.choice([0.8,1.0,1.4,2.0]),
        'save_as_jpeg': random.random() < 0.2
    }

def generate_one(img_id, out_dir, config):
    images_dir = out_dir / "images"
    csvs_dir = out_dir / "csvs"
    meta = {'image': None, 'chart_type': None, 'csv_files': [], 'flags': {}, 'style': {}}

    chart_types = ['cv', 'cd', 'nyquist', 'multi_line', 'scatter']
    ctype = random.choice(chart_types)
    meta['chart_type'] = ctype
    style = sample_style()
    meta['style'] = style

    fig = plt.figure(figsize=style['figsize'])
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': style['font_size']})

    if ctype == 'cv':
        t, y = simulate_cv(noise=config['max_noise'] * random.random())
        ax.plot(t, y, color='tab:blue', linewidth=style['line_width'])
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (A)")
        curves = [{'id':1, 'x': t, 'y': y}]
        # maybe add a second cycle curve overlay
        if random.random() < 0.35:
            t2 = t
            y2 = y*0.6 + 0.02*np.random.randn(len(t))
            ax.plot(t2, y2, color='tab:green', linewidth=style['line_width'], linestyle='--')
            curves.append({'id':2, 'x': t2, 'y': y2})
    elif ctype == 'cd':
        x, chg, dis = simulate_cd(noise=config['max_noise'] * random.random())
        ax.plot(x, chg, label='Charge', color='tab:red', linewidth=style['line_width'])
        ax.plot(x, dis, label='Discharge', color='tab:green', linewidth=style['line_width'], linestyle='--')
        ax.set_xlabel("Capacity (mAh)")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        curves = [{'id':1,'x':x,'y':chg}, {'id':2,'x':x,'y':dis}]
    elif ctype == 'nyquist':
        x, y = simulate_nyquist()
        ax.plot(x, y, color='tab:purple', linewidth=style['line_width'])
        ax.set_xlabel("Z' (Ohm)")
        ax.set_ylabel("-Z'' (Ohm)")
        curves = [{'id':1,'x':x,'y':y}]
    elif ctype == 'multi_line':
        curves = simulate_multi_line(num_curves=random.randint(2,5), noise=config['max_noise'] * random.random())
        colors = plt.cm.tab10(np.linspace(0,1,len(curves)))
        for i,c in enumerate(curves):
            ax.plot(c['x'], c['y'], color=colors[i], linewidth=style['line_width'], label=f'c{i+1}')
        if random.random() < 0.7:
            ax.legend(ncol=1, fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    elif ctype == 'scatter':
        x = np.linspace(0,1,200)
        y = 0.5*np.sin(4*np.pi*x) + 0.2*np.random.randn(len(x))
        ax.scatter(x, y, s=8, color='tab:blue')
        curves = [{'id':1,'x':x,'y':y}]

    # random variations: log axis, secondary axis, annotate, rotate ticks
    flags = {'legend_occluded': False, 'has_intersection': False, 'log_x': False, 'log_y': False, 'twin_axis': False}
    if random.random() < 0.12:
        ax.set_xscale('log')
        flags['log_x'] = True
    if random.random() < 0.08:
        ax.set_yscale('log')
        flags['log_y'] = True
    if ctype == 'multi_line':
        # detect intersections roughly
        # crude count: check sign changes in pairwise differences
        ints = 0
        for i in range(len(curves)):
            for j in range(i+1, len(curves)):
                d = curves[i]['y'] - curves[j]['y']
                ints += np.sum(np.diff(np.sign(d)) != 0)
        flags['has_intersection'] = ints > 0
        flags['intersection_count'] = int(min(ints, 9999))
    # maybe twin axis for cd/multi_line
    if random.random() < 0.12 and ctype in ('cd','multi_line','scatter'):
        # for CD, create twin y using one of curves or derived
        if ctype == 'cd':
            flags['twin_axis'] = maybe_apply_twin_axis(ax, x, chg, dis)
        else:
            # synthetic twin using a transformed average
            avg_y = np.mean([c['y'] for c in curves], axis=0)
            maybe_apply_twin_axis(ax, curves[0]['x'], avg_y, avg_y*0.8 + 0.02*np.random.randn(len(avg_y)))
            flags['twin_axis'] = True

    # maybe occlude legend by drawing patch
    if ax.get_legend() is not None and random.random() < 0.25:
        add_legend_occlusion(ax)
        flags['legend_occluded'] = True

    if not style['grid']:
        ax.grid(False)

    # occasional annotations
    if random.random() < 0.15:
        # add text annotation overlapping curve
        xann = 0.2 + 0.6*random.random()
        yann = 0.2 + 0.6*random.random()
        ax.text(xann, yann, "note", fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    # save image and csvs
    img_fname = f"{img_id}.png"
    img_path = images_dir / img_fname
    dpi = style['dpi']
    fig.tight_layout()
    fig.savefig(img_path, dpi=dpi)
    # optional jpeg variant to simulate lossy: save some as jpeg
    if style['save_as_jpeg']:
        im = Image.open(img_path)
        jpeg_name = f"{img_id}.jpg"
        jpeg_path = images_dir / jpeg_name
        im.convert('RGB').save(jpeg_path, quality=85)
        img_path.unlink()  # remove original png, keep jpeg as canonical
        img_fname = jpeg_name

    csv_files = save_csv_pairs(csvs_dir, img_id, curves)
    meta['image'] = img_fname
    meta['csv_files'] = csv_files
    meta['flags'] = flags
    plt.close(fig)
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100, help='number of images to generate')
    parser.add_argument('--out', type=str, default='generated_echem', help='output directory')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--zip', action='store_true', help='create zip archive at end')
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    out = Path(args.out)
    ensure_dirs(out)
    metas = []
    config = {'max_noise': 0.03}
    print(f"Generating {args.num} samples into {out.resolve()}")
    for i in range(args.num):
        img_id = f"img_{i:04d}"
        meta = generate_one(img_id, out, config)
        meta['id'] = img_id
        metas.append(meta)
        if (i+1) % 10 == 0:
            print(f"  generated {i+1}/{args.num}")
    # write metadata CSV / JSON
    meta_df_rows = []
    for m in metas:
        row = {
            'id': m['id'],
            'image': m['image'],
            'chart_type': m['chart_type'],
            'csv_files': ";".join(m['csv_files']),
            'flags': json.dumps(m['flags']),
            'style': json.dumps(m['style'])
        }
        meta_df_rows.append(row)
    meta_df = pd.DataFrame(meta_df_rows)
    meta_df.to_csv(out / "metadata.csv", index=False)
    # optional zip
    if args.zip:
        import shutil
        archive_name = shutil.make_archive(str(out), 'zip', root_dir=str(out))
        print("Archive created:", archive_name)
    print("Done. Metadata saved to", str(out / "metadata.csv"))

if __name__ == "__main__":
    main()
