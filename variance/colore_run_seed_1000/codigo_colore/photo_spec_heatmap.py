import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--catalogue', type=str, required=True,
                    help='Path to the galaxy catalogue CSV file')
parser.add_argument('--output', type=str, default='photo_spec_heatmap.png',
                    help='Output filename for the plot')
parser.add_argument('--chunksize', type=int, default=2_000_000,
                    help='Chunksize for reading the CSV in chunks')
parser.add_argument('--bins', type=int, default=500,
                    help='Number of bins per axis for the 2D histogram')
args = parser.parse_args()

# --- Pass 1: find global min/max for axis limits without loading everything ---
print('Pass 1: scanning redshift range...')
zs_min, zs_max = np.inf, -np.inf
zp_min, zp_max = np.inf, -np.inf

for chunk in pd.read_csv(args.catalogue, chunksize=args.chunksize,
                         usecols=['SPEC_Z', 'PHOTO_Z']):
    zs_min = min(zs_min, chunk['SPEC_Z'].min())
    zs_max = max(zs_max, chunk['SPEC_Z'].max())
    zp_min = min(zp_min, chunk['PHOTO_Z'].min())
    zp_max = max(zp_max, chunk['PHOTO_Z'].max())

print(f'  SPEC_Z  range: [{zs_min:.3f}, {zs_max:.3f}]')
print(f'  PHOTO_Z range: [{zp_min:.3f}, {zp_max:.3f}]')

zmin = max(0, min(zs_min, zp_min))
zmax = max(zs_max, zp_max)

# --- Pass 2: accumulate 2D histogram in chunks ---
print('Pass 2: building 2D histogram...')
bins_s = np.linspace(zmin, zmax, args.bins + 1)
bins_p = np.linspace(zmin, zmax, args.bins + 1)
H      = np.zeros((args.bins, args.bins), dtype=np.float64)

for i, chunk in enumerate(pd.read_csv(args.catalogue, chunksize=args.chunksize,
                                       usecols=['SPEC_Z', 'PHOTO_Z'])):
    h, _, _ = np.histogram2d(chunk['SPEC_Z'].values,
                              chunk['PHOTO_Z'].values,
                              bins=[bins_s, bins_p])
    H += h
    if (i + 1) % 5 == 0:
        print(f'  Processed {(i+1)*args.chunksize:,} rows...')

print(f'  Total galaxies: {int(H.sum()):,}')

# --- Plot ---
print('Plotting...')
fig, ax = plt.subplots(figsize=(7, 6))

# Mask empty bins so they show as white background
H_masked = np.ma.masked_where(H.T == 0, H.T)

# Log-normalised colormap matching the reference image style
norm  = mcolors.LogNorm(vmin=max(1, H_masked.min()), vmax=H_masked.max())
cmap  = plt.cm.get_cmap('inferno_r').reversed()   # hot colours = high density
cmap.set_bad('white')

im = ax.pcolormesh(bins_s, bins_p, H_masked,
                   norm=norm, cmap='hot_r', shading='auto')

plt.colorbar(im, ax=ax, label='Number of galaxies')

# --- Overlay lines ---
z_line = np.linspace(zmin, zmax, 500)

# Equality line z_p = z_s
ax.plot(z_line, z_line, 'r-', lw=1.2, label=r'$z_p = z_s$')

# sigma lines: z_p = z_s ± 0.05(1+z_s)
sigma = 0.05 * (1 + z_line)
ax.plot(z_line, z_line + sigma, color='cyan', lw=1.0, ls='--',
        label=r'$\pm 0.05(1+z_s)$')
ax.plot(z_line, z_line - sigma, color='cyan', lw=1.0, ls='--')

ax.set_xlim(zmin, zmax)
ax.set_ylim(zmin, zmax)
ax.set_xlabel(r'$z_s$ (spectroscopic)', fontsize=13)
ax.set_ylabel(r'$z_p$ (photometric)',   fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(args.output, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {args.output}')
