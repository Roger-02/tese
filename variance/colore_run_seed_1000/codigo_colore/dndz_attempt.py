import sys, argparse, resource, gc
import numpy as np
import healpy as hp
from scipy.special import erf

import camb
from camb import model, initialpower
import treecorr
import pandas as pd
from camb.sources import SplinedSourceWindow
import os
os.environ.setdefault("PYSM_LOCAL_DATA", "/home/roger02/ScienceCodes/pysm_data")

from multiprocessing import Process, Queue
import time, psutil

_start_time = time.time()

def print_mem(label):
    current_RSS_gb = psutil.Process().memory_info().rss / 1e9
    usage_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    time_passed = time.time() - _start_time
    hours, minutes, seconds = int(time_passed // 3600), int((time_passed % 3600) // 60), time_passed % 60
    print(f'[MEM] [{hours:02d}h{minutes:02d}m{seconds:05.2f}s] {label}: current RSS = {current_RSS_gb:.2f}; peak RSS so far = {usage_gb:.2f} GB', flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
seed = args.seed

print_mem('Start')

def _gen_random_catalogue_theta(RA_min, RA_max, DEC_min, DEC_max, Npoints, seed=None):
    rng = np.random.default_rng(seed)

    phi_min = RA_min/180*np.pi
    phi_max = RA_max/180*np.pi
    
    u_min = phi_min/(2*np.pi)
    u_max = phi_max/(2*np.pi)
    
    theta_min_2 = (90 - DEC_min)/180*np.pi
    theta_max_2 = (90 - DEC_max)/180*np.pi
    
    v_min = (np.cos(theta_min_2) + 1)/2
    v_max = (np.cos(theta_max_2) + 1)/2

    u = u_min + (u_max - u_min)*rng.uniform(0, 1, size=Npoints)
    v = v_min + (v_max - v_min)*rng.uniform(0, 1, size=Npoints)

    ra = 2*np.pi*u
    ra_deg = ra/np.pi*180
    dec = np.pi/2 - np.arccos(2*v - 1)
    dec_deg = dec/np.pi*180

    return pd.DataFrame({'RA': ra_deg % 360, 'DEC': dec_deg})

def PCAclean(M,N_fg,MeanCentre=True,return_eig=False):
    # N_fg: number of eigenmodes for PCA to remove
    nz,npix = np.shape(M)
    M = np.swapaxes(M,0,1) # [nz,npix]->[npix,nz]
    ### Mean Centring and axes configuration:
    if MeanCentre==True:
        M = M - np.mean(M,0) # Mean centre data
    M = np.swapaxes(M,0,1) # [Npix,Nz]->[Nz,Npix]
    ### Covariance calculation:
    C = np.cov(M)
    if MeanCentre==False: # do manual cov calc to avoid mean-centring in numpy:
        C = (M @ M.T) / M.shape[1]
    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    if return_eig==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        return eignumb,eigenval,V
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
    M_clean = (M - np.dot(A,S))
    return M_clean

def Get_PySM_FG_sky(nu=None,nside=256,coord='C'):
    # Frequency inputs in MHz
    npix = hp.nside2npix(nside)
    nu_GHz = nu/1e3
    import pysm3
    import pysm3.units as u
    sky = pysm3.Sky(nside=nside, preset_strings=["f1","s1","d1","a1","c1"])
    if isinstance(nu, np.ndarray): # run PySM for array of nu
        T_FG = np.zeros((len(nu),npix), dtype=np.float32)
        for i in range(len(nu)):
            T_FG[i] = sky.get_emission(nu[i]*u.MHz).value[0] + 2.73e6 # add on CMB monopole
            # print(i)
    if not isinstance(nu, np.ndarray): # run PySM for single nu
        T_FG = sky.get_emission(nu*u.MHz).value[0]
    T_FG /= 1e3 # Convert microK to mK

    if coord=='C' or coord=='E': # rotate sky to coordinate frame
        r = hp.Rotator(coord=['G',coord])
        if isinstance(nu, np.ndarray):
            for i in range(np.shape(T_FG)[0]):
                T_FG[i] = r.rotate_map_pixel(T_FG[i])
        if not isinstance(nu, np.ndarray):
            T_FG = r.rotate_map_pixel(T_FG)
    return T_FG # [in mK]


def nnnnn(galaxies_, randoms_):
    catalog_sources = treecorr.Catalog(ra = galaxies_['RA'], dec = galaxies_['DEC'], ra_units = 'deg', dec_units = 'deg')
    rand_cat = treecorr.Catalog(ra = randoms_['RA'], dec = randoms_['DEC'], ra_units = 'deg', dec_units = 'deg')
    
    print_mem('pos catalogues')

    DD = treecorr.NNCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear')
    RR = treecorr.NNCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear')
    DR = treecorr.NNCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear')
    RD = treecorr.NNCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear')
    print_mem('before process')
    DD.process(catalog_sources)
    RR.process(rand_cat)
    DR.process(catalog_sources, rand_cat)
    RD.process(rand_cat, catalog_sources)
    print_mem('feito')
    xi_sources_lin, _ = DD.calculateXi(rr=RR, dr=DR, rd=RD)
    r_sources_lin = DD.meanr

    del catalog_sources, rand_cat, DD, RR, RD, DR
    gc.collect()
    print_mem('after nnnnn dels')
    return r_sources_lin, xi_sources_lin

def kkkkk_(k_values, ra_window, dec_window):
    catalogue = treecorr.Catalog(ra=ra_window, dec=dec_window, k=k_values, ra_units='deg', dec_units='deg')
    KK_ = treecorr.KKCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear')
    KK_.process(catalogue)

    r, xi, _ = KK_.meanr.copy(), KK_.xi.copy(), KK_.varxi.copy()
    del catalogue, KK_
    gc.collect()
    return r, xi


def nknk_(galaxy_points, random_points, k_values, ra_window, dec_window):
    cat_sour = treecorr.Catalog(ra = galaxy_points['RA'], dec = galaxy_points['DEC'], ra_units = 'deg', dec_units = 'deg')
    cat_rands = treecorr.Catalog(ra = random_points['RA'], dec = random_points['DEC'], ra_units = 'deg', dec_units = 'deg')
    cat_field = treecorr.Catalog(ra=ra_window, dec=dec_window, k=k_values, ra_units='deg', dec_units='deg')
    
    NK_rand = treecorr.NKCorrelation(min_sep = 0.1, max_sep = 5, bin_size = 0.15, sep_units = 'degrees', bin_type = 'Linear')
    NK_rand.process(cat_rands, cat_field)
    
    NK = treecorr.NKCorrelation(min_sep = 0.1, max_sep = 5, bin_size = 0.15, sep_units = 'degrees', bin_type = 'Linear')
    NK.process(cat_sour, cat_field)
    xi, _ = NK.calculateXi(rk = NK_rand)
    r = NK.meanr
    del cat_sour, cat_rands, cat_field, NK_rand, NK
    gc.collect()
    return r, xi


nside = 512

nu_table = np.loadtxt(f'./nuTable.txt')
redshift = (1420.0 / nu_table) - 1

npix = hp.nside2npix(nside) # hp.get_map_size(T_obs_cleaned_full_sky[50]) #T_FG[50])
pixel_indices = np.arange(npix)
theta, phi = hp.pix2ang(nside, pixel_indices)
ra = np.degrees(phi)
dec = 90.0 - np.degrees(theta)

print_mem('after the collapsings and del big matrix')

def full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, beam, photo):

    if beam:
        beamed = 'beamed_'
    else:
        beamed = ''
    if photo:
        photo_zs = 'photo_'
    else:
        photo_zs = 'spec_'

    print_mem(f'run tal que beam:{beam} e photozs:{photo_zs}')

    lower_z_index = int(abs(redshift[::-1] - lower_z).argmin() / 2)
    upper_z_index = int(abs(redshift[::-1] - upper_z).argmin() / 2)

    maps = np.load(f"./{beamed}full_sky_clean.npy", mmap_mode='r')  # doesn't load into memory
    cleaned = np.mean(np.array(maps[lower_z_index:upper_z_index+1]), axis = 0)

    H1_thick_ = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    counter = 0
    for i in range(int(lower_z_index), int(upper_z_index) + 1):
        H1_thick_ += hp.ud_grade(hp.read_map(f'./{beamed}colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)
        counter += 1

    H1_thick_ = H1_thick_ / counter
    delta_H1 = H1_thick_ - np.mean(H1_thick_)
    delta_final = cleaned - np.mean(cleaned)

    print_mem('deltas created')
    
    mask_window_maps = ((ra >= lower_ra) & (ra <= upper_ra) & (dec >= lower_dec) & (dec <= upper_dec))
    # mask_window_gals = ((galaxies_['RA'] >= lower_ra) & (galaxies_['RA'] <= upper_ra) & (galaxies_['DEC'] >= lower_dec) & (galaxies_['DEC'] <= upper_dec))

    # print('tamanho anterior galaxias', np.shape(galaxies_), type(galaxies_))
    # print_mem('before gals')
    # galaxies_ = galaxies_[mask_window_gals]
    # print('tamanho posterior galaxias', np.shape(galaxies_), type(galaxies_))
    # print_mem('pos window gals')
    n_rands = 5 * len(galaxies_)
    print('numero de rands', n_rands)
    randoms_ = _gen_random_catalogue_theta(RA_min=lower_ra, RA_max=upper_ra, DEC_min=lower_dec, DEC_max=upper_dec, Npoints=n_rands, seed=7+115*10+10)
    print_mem('pos criar randoms grandes')
    print('tamanho dos rands', np.shape(randoms_), type(randoms_))
    delta_H1 = delta_H1[mask_window_maps]
    delta_final = delta_final[mask_window_maps]
    ra_window = ra[mask_window_maps]
    dec_window = dec[mask_window_maps]

    all_ws = pd.DataFrame()
    # print_mem('pre h1')
    # all_ws['r_h1'], all_ws['xi_h1'] = kkkkk_(delta_H1, ra_window, dec_window)
    # print_mem('pos h1, pre h1 clean')
    # all_ws['r_clean'], all_ws['xi_clean'] = kkkkk_(delta_final, ra_window, dec_window)
    # print_mem('pos clean, pre cross')
    # all_ws['r_gal'], all_ws['xi_gal'] = nnnnn(galaxies_, randoms_)
    # print_mem('pos gal, pre cross')
    all_ws['r_cross'], all_ws['xi_cross'] = nknk_(galaxies_, randoms_, delta_H1, ra_window, dec_window)
    print_mem('pos cross, pre cleaned cross')
    all_ws['r_cross_clean'], all_ws['xi_cross_clean'] = nknk_(galaxies_, randoms_, delta_final, ra_window, dec_window)
    print_mem('pos cross clean, pre limpeza')
    # del galaxies_
    gc.collect()
    # random_test = _gen_random_catalogue_theta(RA_min=lower_ra, RA_max=upper_ra, DEC_min=lower_dec, DEC_max=upper_dec, Npoints=int(len(randoms_)/5), seed=np.random.randint(1, 100))
    # print_mem('pos criar randoms de teste, pre cross rand')
    # all_ws['r_cross_rand'], all_ws['xi_cross_rand'] = nknk_(random_test, randoms_, delta_H1, ra_window, dec_window)
    # print_mem('pos cross rand, pre cross rand clean')
    # all_ws['r_cross_rand_clean'], all_ws['xi_cross_rand_clean'] = nknk_(random_test, randoms_, delta_final, ra_window, dec_window)

    print_mem('w thetas created')

    return all_ws

lower_ra = 100
upper_ra = 250
lower_dec = -90
upper_dec = -30

chunksize = 2000000
# janela_list = []
# for chunk in pd.read_csv(f"./big_beautiful_full_sky_catalogue.csv", chunksize=chunksize):
#     mask2 = ((chunk['SPEC_Z'] >= 0.6) & (chunk['SPEC_Z'] <= 0.9) & 
#              (chunk['RA'] >= lower_ra) & (chunk['RA'] <= upper_ra) & 
#              (chunk['DEC'] >= lower_dec) & (chunk['DEC'] <= upper_dec))
#     janela_list.append(chunk.loc[mask2])
#     del chunk
# galaxies_ = pd.concat(janela_list, ignore_index=True)
# del janela_list
# print(galaxies_.shape)

hi_bins = np.arange(0.5, 0.951, 0.01)
# for i in [0]:#range(len(hi_bins) - 1):
#     ws_NB_S = full_sky_cls(hi_bins[i], hi_bins[i+1], lower_ra, upper_ra, lower_dec, upper_dec, False, False)
#     ws_B_S = full_sky_cls(hi_bins[i], hi_bins[i+1], lower_ra, upper_ra, lower_dec, upper_dec, True, False)

#     ws_NB_S.to_csv(f'./dndz_/NB_S_hi-{hi_bins[i]}-{hi_bins[i+1]}.csv')
#     ws_B_S.to_csv(f'./dndz_/B_S_hi-{hi_bins[i]}-{hi_bins[i+1]}.csv')

# del galaxies_
janela_list = []
for chunk in pd.read_csv(f"./big_beautiful_full_sky_catalogue.csv", chunksize=chunksize):
    mask2 = ((chunk['PHOTO_Z'] >= 0.65) & (chunk['PHOTO_Z'] <= 0.8) & 
             (chunk['RA'] >= lower_ra) & (chunk['RA'] <= upper_ra) & 
             (chunk['DEC'] >= lower_dec) & (chunk['DEC'] <= upper_dec))
    janela_list.append(chunk.loc[mask2])
    del chunk
galaxies_ = pd.concat(janela_list, ignore_index=True)
del janela_list
print(galaxies_.shape)
print_mem('galaxies created')

for i in range(len(hi_bins) - 1):
    ws_NB_P = full_sky_cls(hi_bins[i], hi_bins[i+1], lower_ra, upper_ra, lower_dec, upper_dec, False, True)
    ws_B_P = full_sky_cls(hi_bins[i], hi_bins[i+1], lower_ra, upper_ra, lower_dec, upper_dec, True, True)

    ws_NB_P.to_csv(f'./dndz_here_we_go_again/NB_P_hi-{round(hi_bins[i], 3):.2f}-{round(hi_bins[i+1], 3):.2f}.csv')
    ws_B_P.to_csv(f'./dndz_here_we_go_again/B_P_hi-{round(hi_bins[i], 3):.2f}-{round(hi_bins[i+1], 2):.2f}.csv')
