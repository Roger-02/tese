import sys, argparse
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
os.environ["PYSM_LOCAL_DATA"] = "/home/roger02/ScienceCodes/pysm_data"


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--lower-z', type=float, required=True)
parser.add_argument('--upper-z', type=float, required=True)
args = parser.parse_args()
seed = args.seed

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
                         theta=theta, type='NN', method='FFTLog')
    
    return wtheta

def dndz_des_alonso(zi, amp=22.357, alpha=1.044, z0=0.568, beta=1.742):
    return amp*(zi/z0)**alpha*np.exp(-(zi/z0)**beta)*3600

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
        T_FG = np.zeros((len(nu),npix))
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


def kkkkk_(k_values, ra_window, dec_window):
    catalogue = treecorr.Catalog(ra=ra_window, dec=dec_window, k=k_values, ra_units='deg', dec_units='deg', npatch = 50)
    KK_ = treecorr.KKCorrelation(min_sep=0.1, max_sep=5, bin_size=0.15, sep_units='degrees', bin_type = 'Linear', var_method = 'jackknife')
    KK_.process(catalogue)
    return KK_.meanr, KK_.xi, KK_.varxi


nside = 512

nu_table = np.loadtxt('./nuTable.txt')
redshift = (1420.0 / nu_table) - 1
nu_for_psm = []
for i in nu_table:
    nu_for_psm.append(round(np.average(i), 4))
nu_for_psm = np.array(nu_for_psm)
T_FG = Get_PySM_FG_sky(nu_for_psm, nside, coord = 'G')

for i in range(len(T_FG)):
    hiim = hp.ud_grade(hp.read_map(f'./colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)
    T_FG[i] += + hiim

npix = hp.get_map_size(T_FG[50])
pixel_indices = np.arange(npix)
theta, phi = hp.pix2ang(nside, pixel_indices)
ra = np.degrees(phi)
dec = 90.0 - np.degrees(theta)

lower_z, upper_z = args.lower_z, args.upper_z

def full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec):

    lower_z_index = abs(redshift[::-1] - lower_z).argmin() / 2
    upper_z_index = abs(redshift[::-1] - upper_z).argmin() / 2

    H1_thick_ = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    for i in range(int(lower_z_index), int(upper_z_index) + 1):
        H1_thick_ += hp.ud_grade(hp.read_map(f'./colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)

    delta_H1 = H1_thick_ - np.mean(H1_thick_)

    print('deltas created')

    all_ws = pd.DataFrame()
    
    mask_window_maps = ((ra >= lower_ra) & (ra <= upper_ra) & (dec >= lower_dec) & (dec <= upper_dec))
    delta_H1 = delta_H1[mask_window_maps]
    ra_window = ra[mask_window_maps]
    dec_window = dec[mask_window_maps]


    all_ws['r_h1'], all_ws['xi_h1'], all_ws['varxi_h1'] = kkkkk_(delta_H1, ra_window, dec_window)

    return all_ws

lower_ra = 100
upper_ra = 200
lower_dec = -90
upper_dec = 10


all_ws  = full_sky_cls(lower_z, upper_z, lower_ra, upper_ra, lower_dec, upper_dec)


suffix = {0.1: '0_1', 0.05: '0_05', 0.02: '0_02', 0.01: '0_01'}
dz = round(upper_z - lower_z, 6)
for width, name in suffix.items():
    if abs(dz - width) < 1e-6:
        all_ws.to_csv(f'./wt_bin_{name}.csv')
