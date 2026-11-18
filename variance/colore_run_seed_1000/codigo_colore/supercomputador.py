import sys, argparse, resource, gc
import numpy as np
import pyccl as ccl
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
parser.add_argument('--lower-z', type=float, required=True)
parser.add_argument('--upper-z', type=float, required=True)
args = parser.parse_args()
seed = args.seed

print_mem('Start')

def _worker(func, args, queue):
    result = func(*args)
    queue.put(result)

def run_isolated(func, *args):
    q = Queue()
    p = Process(target=_worker, args=(func, args, q))
    p.start()
    result = q.get()
    p.join()
    return result

def cosmology(omega_M, omega_B, omega_L, h, w, ns, sigma_8, As):
    
    csm = {'omega_M': omega_M,
       'omega_B': omega_B,
       'omega_L': omega_L,
       'h': h,
       'w': w,
       'ns': ns,
       'sigma_8': sigma_8,
       'As': As}
    
    return csm


def make_camb(csm, lmax, zs, Ws, bias, cross = False):

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=csm['h']*100, ombh2=csm['omega_B']*csm['h']**2, omch2=(csm['omega_M']-csm['omega_B'])*csm['h']**2)
    pars.InitPower.set_params(As=csm['As'], ns=csm['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    
    pars.Want_CMB, pars.Want_CMB_lensing, pars.DoLensing                                                    = False, False, False
    pars.SourceTerms.counts_velocity, pars.SourceTerms.counts_radial, pars.SourceTerms.counts_timedelay     = False, False, False
    pars.SourceTerms.counts_ISW, pars.SourceTerms.counts_potential, pars.SourceTerms.counts_evolve          = False, False, False
    pars.SourceTerms.line_phot_dipole, pars.SourceTerms.line_phot_quadrupole, pars.SourceTerms.line_basic   = False, False, False
    pars.SourceTerms.line_distortions, pars.SourceTerms.line_extra, pars.SourceTerms.line_reionization      = False, False, False
    pars.SourceTerms.use_21cm_mK                                                                            = False      
    pars.SourceTerms.counts_density, pars.SourceTerms.counts_redshift, pars.SourceTerms.counts_lensing      = True, False, False

    if cross:
        pars.SourceWindows = [SplinedSourceWindow(z = zs[0], W = Ws[0], bias_z = bias[0]),
                              SplinedSourceWindow(z = zs[1], W = Ws[1], bias_z = bias[1])]
        map_ = 'W1xW2'
    else:
        pars.SourceWindows = [SplinedSourceWindow(z = zs, W = Ws, bias_z = bias)]
        map_ = 'W1xW1'
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)[map_][2 : lmax + 1]
    ls = np.arange(2, lmax+1)

    return ls, cls

def make_w_theta(csm, ls, cls, theta):
    
    cosmo = ccl.Cosmology(Omega_c=csm['omega_M']-csm['omega_B'],
                      Omega_b=csm['omega_B'],
                      h=csm['h'],
                      sigma8=csm['sigma_8'],
                      n_s=csm['ns'],
                      transfer_function='eisenstein_hu')

    wtheta = ccl.correlation(cosmo, ell=ls, C_ell=cls, 
                         theta=theta, type='NN', method='Legendre')
    
    return wtheta

def dndz_des_alonso(zi, amp=22.357, alpha=1.044, z0=0.568, beta=1.742):
    return amp*(zi/z0)**alpha*np.exp(-(zi/z0)**beta)*3600

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

def get_pk(csm, z_in):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=csm['h']*100, ombh2=csm['omega_B']*csm['h']**2, omch2=(csm['omega_M']-csm['omega_B'])*csm['h']**2,omk=0,mnu=0)
    pars.set_dark_energy() #LCDM (default)
    pars.InitPower.set_params(ns=csm['ns'], r=0, As=csm['As'])
    pars.set_matter_power(redshifts=[z_in], kmax=2.0)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2.0, npoints = 500)
    
    csm['sigma_8'] = float(results.get_sigma8()[0])
    f = results.get_fsigma8()/csm['sigma_8']
    
    return csm, kh, pk[0], f[0]

def W_err(z,zc,sz,smooth):
    return (erf((zc+sz-z)/(np.sqrt(2)*smooth))-erf((zc-sz-z)/(np.sqrt(2)*smooth)))/2

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

def clean_T(lower_z, upper_z, beam, matrix):
    lower_z_index = abs(redshift[::-1] - lower_z).argmin() / 2
    upper_z_index = abs(redshift[::-1] - upper_z).argmin() / 2
    counter = 0

    if beam:
        T_obs_cleaned_masked_beamed = np.zeros(hp.nside2npix(nside), dtype=np.float32)
        for i in range(int(lower_z_index), int(upper_z_index) + 1):
            T_obs_cleaned_masked_beamed += matrix[i] # T_obs_cleaned_full_sky_beam[i]
            counter += 1
        return T_obs_cleaned_masked_beamed / counter
    else:
        T_obs_cleaned_masked_ = np.zeros(hp.nside2npix(nside), dtype=np.float32)
        for i in range(int(lower_z_index), int(upper_z_index) + 1):
            T_obs_cleaned_masked_ += matrix[i] # T_obs_cleaned_full_sky[i]
            counter += 1
        return T_obs_cleaned_masked_ / counter


nside = 512

nu_table = np.loadtxt(f'./nuTable.txt')
redshift = (1420.0 / nu_table) - 1
nu_for_psm = []
for i in nu_table:
    nu_for_psm.append(round(np.average(i), 4))
nu_for_psm = np.array(nu_for_psm)
lower_z, upper_z = args.lower_z, args.upper_z

print_mem('before cleaning')

T_FG = Get_PySM_FG_sky(nu_for_psm, nside, coord = 'G')
for i in range(len(T_FG)):
    hiim = hp.ud_grade(hp.read_map(f'./colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)
    T_FG[i] += hiim
T_obs_cleaned_full_sky = PCAclean(T_FG, 4)
del T_FG
T_obs_cleaned_masked_ = clean_T(lower_z, upper_z, False, T_obs_cleaned_full_sky)
del T_obs_cleaned_full_sky
gc.collect()
print_mem('after del sim FG')

T_FG_beam = Get_PySM_FG_sky(nu_for_psm, nside, coord = 'G')
for i in range(len(T_FG_beam)):
    hiim2 = hp.ud_grade(hp.read_map(f'./beamed_colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out=nside)
    T_FG_beam[i] += hiim2
    T_FG_beam[i] = hp.smoothing(...., T_FG_beam[i])
T_obs_cleaned_full_sky_beam = PCAclean(T_FG_beam, 4)
del T_FG_beam
T_obs_cleaned_beamed = clean_T(lower_z, upper_z, True, T_obs_cleaned_full_sky_beam)
del T_obs_cleaned_full_sky_beam
gc.collect()
print_mem('after del beam FG')

npix = hp.nside2npix(nside) # hp.get_map_size(T_obs_cleaned_full_sky[50]) #T_FG[50])
pixel_indices = np.arange(npix)
theta, phi = hp.pix2ang(nside, pixel_indices)
ra = np.degrees(phi)
dec = 90.0 - np.degrees(theta)

print_mem('after the collapsings and del big matrix')

def full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, beam, photo, cleaned):

    if beam:
        beamed = 'beamed_'
    else:
        beamed = ''
    if photo:
        photo_zs = 'photo_'
        redshifts = 'PHOTO_Z'
    else:
        photo_zs = 'spec_'
        redshifts = 'SPEC_Z'

    print_mem(f'run tal que beam:{beam} e photozs:{photo_zs}')

    lower_z_index = abs(redshift[::-1] - lower_z).argmin() / 2
    upper_z_index = abs(redshift[::-1] - upper_z).argmin() / 2

    H1_thick_ = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    counter = 0
    for i in range(int(lower_z_index), int(upper_z_index) + 1):
        H1_thick_ += hp.ud_grade(hp.read_map(f'./{beamed}colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)
        counter += 1

    H1_thick_ = H1_thick_ / counter
    delta_H1 = H1_thick_ - np.mean(H1_thick_)
    delta_final = cleaned - np.mean(cleaned) # T_obs_cleaned_masked_ - np.mean(T_obs_cleaned_masked_)

    print_mem('deltas created')

    chunksize = 2000000
    janela_list = []
    for chunk in pd.read_csv(f"./{photo_zs}full_sky_0.7-0.75_catalogue.csv", chunksize=chunksize):
        mask2 = ((chunk[redshifts] >= lower_z) & (chunk[redshifts] <= upper_z))
        janela_list.append(chunk.loc[mask2])
        del chunk
    galaxies_ = pd.concat(janela_list, ignore_index=True)

    print_mem('galaxies created')
    
    ra_for_cl  = galaxies_["RA"].values
    dec_for_cl = galaxies_["DEC"].values
    ipix = hp.ang2pix(nside, ra_for_cl, dec_for_cl, lonlat=True)
    binc = np.bincount(ipix, minlength = hp.nside2npix(nside))
    delta_galaxies = (binc - np.mean(binc)) / np.mean(binc)
    
    all_cls = pd.DataFrame()
    all_ws = pd.DataFrame()
    
    all_cls['cl_h1'] = hp.anafast(delta_H1, lmax = 400)
    all_cls['cl_clean'] = hp.anafast(delta_final, lmax = 400)
    all_cls['cl_gal'] = hp.anafast(delta_galaxies, lmax = 400)
    all_cls['cl_cross'] = hp.anafast(delta_H1, delta_galaxies, lmax = 400)
    all_cls['cl_cross_clean'] = hp.anafast(delta_final, delta_galaxies, lmax = 400)

    print_mem('cls created')
    
    mask_window_maps = ((ra >= lower_ra) & (ra <= upper_ra) & (dec >= lower_dec) & (dec <= upper_dec))
    mask_window_gals = ((galaxies_['RA'] >= lower_ra) & (galaxies_['RA'] <= upper_ra) & (galaxies_['DEC'] >= lower_dec) & (galaxies_['DEC'] <= upper_dec))

    print('tamanho anterior galaxias', np.shape(galaxies_), type(galaxies_))
    print_mem('before gals')
    galaxies_ = galaxies_[mask_window_gals]
    print('tamanho posterior galaxias', np.shape(galaxies_), type(galaxies_))
    print_mem('pos window gals')
    n_rands = 5 * len(galaxies_)
    print('numero de rands', n_rands)
    randoms_ = _gen_random_catalogue_theta(RA_min=lower_ra, RA_max=upper_ra, DEC_min=lower_dec, DEC_max=upper_dec, Npoints=n_rands, seed=7+115*10+10)
    print_mem('pos criar randoms grandes')
    print('tamanho dos rands', np.shape(randoms_), type(randoms_))
    delta_H1 = delta_H1[mask_window_maps]
    delta_final = delta_final[mask_window_maps]
    ra_window = ra[mask_window_maps]
    dec_window = dec[mask_window_maps]

    print_mem('pre h1')
    all_ws['r_h1'], all_ws['xi_h1'] = kkkkk_(delta_H1, ra_window, dec_window) # run_isolated(kkkkk_, delta_H1, ra_window, dec_window)
    print_mem('pos h1, pre h1 clean')
    all_ws['r_clean'], all_ws['xi_clean'] = kkkkk_(delta_final, ra_window, dec_window) # run_isolated(kkkkk_, delta_final, ra_window, dec_window)
    print_mem('pos clean, pre gal')
    all_ws['r_gal'], all_ws['xi_gal'] = nnnnn(galaxies_, randoms_) # run_isolated(nnnnn, galaxies_, randoms_)
    print_mem('pos gal, pre cross')
    all_ws['r_cross'], all_ws['xi_cross'] = nknk_(galaxies_, randoms_, delta_H1, ra_window, dec_window) #run_isolated(nknk_, galaxies_, randoms_, delta_H1, ra_window, dec_window)
    print_mem('pos cross, pre cleaned cross')
    all_ws['r_cross_clean'], all_ws['xi_cross_clean'] = nknk_(galaxies_, randoms_, delta_final, ra_window, dec_window) #run_isolated(nknk_, galaxies_, randoms_, delta_final, ra_window, dec_window)
    print_mem('pos cross clean, pre limpeza')
    del galaxies_
    gc.collect()
    random_test = _gen_random_catalogue_theta(RA_min=lower_ra, RA_max=upper_ra, DEC_min=lower_dec, DEC_max=upper_dec, Npoints=int(len(randoms_)/5), seed=np.random.randint(1, 100))
    print_mem('pos criar randoms de teste, pre cross rand')
    all_ws['r_cross_rand'], all_ws['xi_cross_rand'] = nknk_(random_test, randoms_, delta_H1, ra_window, dec_window) #run_isolated(nknk_, random_test, randoms_, delta_H1, ra_window, dec_window)
    print_mem('pos cross rand, pre cross rand clean')
    all_ws['r_cross_rand_clean'], all_ws['xi_cross_rand_clean'] = nknk_(random_test, randoms_, delta_final, ra_window, dec_window) #run_isolated(nknk_, random_test, randoms_, delta_final, ra_window, dec_window)

    print_mem('w thetas created')

    return all_cls, all_ws

lower_ra = 100
upper_ra = 240
lower_dec = -90
upper_dec = -30

suffix = {0.05:'0_05', 0.02:'0_02', 0.01:'0_01'}
dz = round(upper_z - lower_z, 6)

all_cls_sim_spec, all_ws_sim_spec = full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, False, False, T_obs_cleaned_masked_)
for width, name in suffix.items():
    if abs(width - dz) < 1e-6:
        all_cls_sim_spec.to_csv(f'./sim_spec_cls_bin_{name}.csv')
        all_ws_sim_spec.to_csv(f'./sim_spec_wt_bin_{name}.csv')
del all_cls_sim_spec, all_ws_sim_spec

all_cls_sim_photo, all_ws_sim_photo = full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, False, True, T_obs_cleaned_masked_)
for width, name in suffix.items():
    if abs(width - dz) < 1e-6:
        all_cls_sim_photo.to_csv(f'./sim_photo_cls_bin_{name}.csv')
        all_ws_sim_photo.to_csv(f'./sim_photo_wt_bin_{name}.csv')
del all_cls_sim_photo, all_ws_sim_photo

all_cls_beam_spec, all_ws_beam_spec = full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, True, False, T_obs_cleaned_beamed)
for width, name in suffix.items():
    if abs(width - dz) < 1e-6:
        all_cls_beam_spec.to_csv(f'./beamed_spec_cls_bin_{name}.csv')
        all_ws_beam_spec.to_csv(f'./beamed_spec_wt_bin_{name}.csv')
del all_cls_beam_spec, all_ws_beam_spec

all_cls_beam_photo, all_ws_beam_photo = full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec, True, True, T_obs_cleaned_beamed)
for width, name in suffix.items():
    if abs(width - dz) < 1e-6:
        all_cls_beam_photo.to_csv(f'./beamed_photo_cls_bin_{name}.csv')
        all_ws_beam_photo.to_csv(f'./beamed_photo_wt_bin_{name}.csv')
del all_cls_beam_photo, all_ws_beam_photo

# csm = cosmology(0.31, 0.05, 1-0.31, 0.6774, -1.0, 0.967, np.nan, 2.142e-9)
# csm, kh, pkh, f = get_pk(csm, 0.0)
# lmax = 5000
# zs_theory = np.arange(0.1, 3.0, 0.0005)
# bias_theory = np.ones_like(zs_theory)
# thetas = np.linspace(0.1, 5, 100)
# theory_cls = pd.DataFrame()
# theory_ws = pd.DataFrame()
# theory_ws['wts_theta'] = thetas


# theory_cls['ls_theory_h1'], theory_cls['cls_theory_h1'] = make_camb(csm, lmax, zs_theory, W_err(zs_theory, (lower_z+upper_z)/2, abs(lower_z-upper_z)/2, 0.001), 
#                                                               bias_theory, cross = False)
# theory_cls['ls_theory_gal'], theory_cls['cls_theory_gal'] = make_camb(csm, lmax, zs_theory, W_err(zs_theory, (lower_z+upper_z)/2, abs(lower_z-upper_z)/2, 0.001) * dndz_des_alonso(zs_theory), bias_theory, cross=False)
# theory_cls['ls_theory_cross'], theory_cls['cls_theory_cross'] = make_camb(csm, lmax, [zs_theory, zs_theory], 
#                                                             [W_err(zs_theory, (lower_z+upper_z)/2, abs(lower_z-upper_z)/2, 0.001), 
#                                                              W_err(zs_theory, (lower_z+upper_z)/2, abs(lower_z-upper_z)/2, 0.001) * dndz_des_alonso(zs_theory)], 
#                                                                     [bias_theory, bias_theory], cross = True)


# theory_ws['wts_theory_h1'] = make_w_theta(csm, theory_cls['ls_theory_h1'], theory_cls['cls_theory_h1'] * 0.05**2, thetas)
# theory_ws['wts_theory_gal'] = make_w_theta(csm, theory_cls['ls_theory_gal'], theory_cls['cls_theory_gal'], thetas)
# theory_ws['wts_theory_cross'] = make_w_theta(csm, theory_cls['ls_theory_cross'], theory_cls['cls_theory_cross'] * 0.05, thetas)


# for width, name in suffix.items():
#     if abs(dz - width) < 1e-6:
#        all_cls.to_csv(f'./sim_photo_cls_bin_{name}.csv')
#        all_ws.to_csv(f'./sim_photo_wt_bin_{name}.csv')
#         theory_cls.to_csv(f'./theory_cls_bin_{name}.csv')
#         theory_ws.to_csv(f'./theory_wt_bin_{name}.csv')
