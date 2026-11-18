import sys, argparse, resource, gc
import numpy as np
import healpy as hp

import pandas as pd
import os
os.environ.setdefault("PYSM_LOCAL_DATA", "/home/roger02/ScienceCodes/pysm_data")

from multiprocessing import Process, Queue
import time, psutil

parser = argparse.ArgumentParser()
parser.add_argument('--SEED', type=int, required=True)
args = parser.parse_args()
seed = args.SEED


_start_time = time.time()

def print_mem(label):
    current_RSS_gb = psutil.Process().memory_info().rss / 1e9
    usage_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    time_passed = time.time() - _start_time
    hours, minutes, seconds = int(time_passed // 3600), int((time_passed % 3600) // 60), time_passed % 60
    print(f'[MEM] [{hours:02d}h{minutes:02d}m{seconds:05.2f}s] {label}: current RSS = {current_RSS_gb:.2f}; peak RSS so far = {usage_gb:.2f} GB', flush=True)



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


nside = 512

nu_table = np.loadtxt(f'./nuTable.txt')
redshift = (1420.0 / nu_table) - 1
nu_for_psm = []
for i in nu_table:
    nu_for_psm.append(round(np.average(i), 4))
nu_for_psm = np.array(nu_for_psm)

T_FG = Get_PySM_FG_sky(nu_for_psm, nside, coord = 'G')
for i in range(len(T_FG)):
    hiim = hp.ud_grade(hp.read_map(f'./colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out = nside)
    T_FG[i] += hiim
T_obs_cleaned_full_sky = PCAclean(T_FG, 4)
np.save('./full_sky_clean.npy', T_obs_cleaned_full_sky)
del T_FG
del T_obs_cleaned_full_sky
gc.collect()
print_mem('after del sim FG')

T_FG_beam = Get_PySM_FG_sky(nu_for_psm, nside, coord = 'G')
for i in range(len(T_FG_beam)):
    hiim2 = hp.ud_grade(hp.read_map(f'./beamed_colore_seed_{seed}_imap_s1_nu{i:03}.fits'), nside_out=nside)
    T_FG_beam[i] += hiim2
T_obs_cleaned_full_sky_beam = PCAclean(T_FG_beam, 4)
np.save('./beamed_full_sky_clean.npy', T_obs_cleaned_full_sky_beam)
del T_FG_beam
del T_obs_cleaned_full_sky_beam
gc.collect()
print_mem('after del beam FG')
