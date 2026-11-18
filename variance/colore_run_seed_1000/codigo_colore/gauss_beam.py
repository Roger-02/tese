import os
import numpy as np
import healpy as hp

# Load frequency table and derived quantities
nu_table   = np.loadtxt('./nuTable.txt')           # shape (N, 2), in MHz, increasing freq
redshift   = (1420.0 / nu_table) - 1              # shape (N, 2)
wavelengths = 3e8 / (nu_table * 1e6)              # in metres, shape (N, 2)

fwhm_rad = np.array([1.16 * np.mean(wavelengths[i]) / 13.5 
                     for i in range(len(wavelengths))])   # shape (N,), in radians

# sigma of the Gaussian in radians: sigma = FWHM / (2 * sqrt(2 * ln(2)))
sigma_rad = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))     # shape (N,), in radians

# --- Define redshift range to process ---
lower_z = 0.6
upper_z = 1.0
nside   = 1024

lower_z_index = int(abs(redshift[::-1] - lower_z).argmin() / 2)
upper_z_index = int(abs(redshift[::-1] - upper_z).argmin() / 2)

print(f'Processing channels {lower_z_index} to {upper_z_index}')

for i in range(lower_z_index, upper_z_index + 1):
    print(f'Smoothing {i}th map')
    true_map = hp.read_map(f'./colore_seed_1000_imap_s1_nu{i:03}.fits')

    smoothed_map = hp.smoothing(true_map, sigma=sigma_rad[i], verbose=False)

    hp.write_map(f'./beamed_colore_seed_1000_imap_s1_nu{i:03}.fits', smoothed_map, overwrite=True)
    print(f'Channel {i:03}: nu={np.mean(nu_table[i]):.1f} MHz, '
          f'FWHM={np.degrees(fwhm_rad[i]):.3f} deg, '
          f'sigma={np.degrees(sigma_rad[i]):.3f} deg — done')