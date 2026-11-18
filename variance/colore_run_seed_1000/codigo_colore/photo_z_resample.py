import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
seed = args.seed
directory = (f"/home/roger02/project/projects/variance/colore_run_seed_{seed}/")
keyword = "_srcs_"

matching_files = [filename for filename in os.listdir(directory) if keyword in filename]

filename = 'big_beautiful_full_sky_catalogue.csv'
pd.DataFrame(columns=["SPEC_Z", "PHOTO_Z", "RA", "DEC"]).to_csv(directory+filename, index=False)
spec_file = 'spec_full_sky_0.7-0.75_catalogue.csv'
photo_file = 'photo_full_sky_0.7-0.75_catalogue.csv'
pd.DataFrame(columns=["SPEC_Z", "PHOTO_Z", "RA", "DEC"]).to_csv(directory+spec_file, index=False)
pd.DataFrame(columns=["SPEC_Z", "PHOTO_Z", "RA", "DEC"]).to_csv(directory+photo_file, index=False)

sample_spec  = []
sample_photo = []

for fname in matching_files:
    print("Processing:", fname)
    
    with fits.open(os.path.join(directory, fname), memmap=True) as hdul:
        cat    = hdul[1].data
        SPEC_Z = np.array(cat['Z_COSMO'], dtype=np.float32)
        RA     = np.array(cat['RA'],      dtype=np.float32)
        DEC    = np.array(cat['DEC'],     dtype=np.float32)


    PHOTO_Z = np.random.normal(SPEC_Z, 0.05 * (1 + SPEC_Z)).astype(np.float32)
    neg_mask = PHOTO_Z < 0
    while neg_mask.any():
        PHOTO_Z[neg_mask] = np.random.normal(SPEC_Z[neg_mask], 0.05 * (1 + SPEC_Z[neg_mask])).astype(np.float32)
        neg_mask = PHOTO_Z < 0

    df = pd.DataFrame({'SPEC_Z': SPEC_Z, 'PHOTO_Z': PHOTO_Z, 'RA': RA, 'DEC':DEC})
    df.to_csv(directory+filename, mode='a', header=False, index=False)
    del df

    spec_mask = (SPEC_Z >= 0.7) & (SPEC_Z <= 0.75)
    photo_mask = (PHOTO_Z >= 0.7) & (PHOTO_Z <= 0.75)

    df_spec = pd.DataFrame({'SPEC_Z': SPEC_Z[spec_mask], 'PHOTO_Z': PHOTO_Z[spec_mask], 'RA': RA[spec_mask], 'DEC':DEC[spec_mask]})
    df_spec.to_csv(directory+spec_file, mode='a', header=False, index=False)
    del df_spec

    df_photo = pd.DataFrame({'SPEC_Z': SPEC_Z[photo_mask], 'PHOTO_Z': PHOTO_Z[photo_mask], 'RA': RA[photo_mask], 'DEC':DEC[photo_mask]})
    df_photo.to_csv(directory+photo_file, mode='a', header=False, index=False)
    del df_photo


    nsample = int(0.2 * len(SPEC_Z))
    idx = np.random.choice(len(SPEC_Z), nsample, replace = False)
    sample_spec.append(SPEC_Z[idx].copy())
    sample_photo.append(PHOTO_Z[idx].copy())

    del SPEC_Z, PHOTO_Z, RA, DEC, spec_mask, photo_mask, neg_mask

sample_spec = np.concatenate(sample_spec)
sample_photo = np.concatenate(sample_photo)

plt.figure(figsize=(7, 6))
plt.scatter(sample_spec, sample_photo, s=1, marker='.', alpha=0.3)
zmin = min(sample_spec.min(), sample_photo.min())
zmax = max(sample_spec.max(), sample_photo.max())
plt.plot([zmin, zmax], [zmin, zmax], 'r--', lw=1, label='z_photo = z_spec')
plt.axhline(0.7, color='k', lw=0.2, ls='--')
plt.axhline(0.75, color='k', lw=0.2, ls='--')
plt.axvline(x=0.75, color='y', lw=0.2, ls='--')
plt.axvline(x=0.7, color='y', lw=0.2, ls='--')
plt.ylabel('Photometric redshifts')
plt.xlabel('Spectroscopic redshifts')
plt.legend()
plt.tight_layout()
plt.savefig(directory+'photo_spec_zs.png', dpi=300)
plt.close()
