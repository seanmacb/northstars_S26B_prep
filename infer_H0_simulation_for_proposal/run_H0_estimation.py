import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck18
import astropy.constants as const
import astropy.units as u
from ligo.skymap.io import read_sky_map
import bilby
import pickle
from tqdm import tqdm
import h5py, os
from run_fisher_analysis import fisher_analysis_GWfish
from distancetool.find_horizon_range_network_de import calculate_range_from_distancetool
import time
import datetime
from joblib import Parallel, delayed
import sys
import traceback

###constants###
Mo = const.M_sun.value #solar mass [kg]
G = const.G.value #Newton constant [m^3 kg^-1 s^2]
c = const.c.value #light speed [m s^-1]
pc = const.pc.value #1pc [m]
###############

"""functions for cosmology calculations"""

"""functions for logging"""
class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command, which is called by print().
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        # Delegate other attributes to the original stdout
        return getattr(self.terminal, attr)

"""compute network SNR and sky localization area using GWFish Fisher analysis"""
def compute_network_SNR_and_sky_area(dL, ra, dec):
    # random_seed = 42
    # bilby.core.utils.random.seed(random_seed)
    # print(f"random seed for param sampling: {random_seed}")
    theta_jn_sample = bilby.core.prior.Sine(name='theta_jn', boundary='reflective').sample(len(dL))
    psi_sample = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi).sample(len(dL))
    phase_sample = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi).sample(len(dL))
    geocent_time = 1187008882.4  # time of GW170817
    
    param_dict = {
            'mass_1': 30.0 * np.ones_like(dL),
            'mass_2': 30.0 * np.ones_like(dL),
            'chi_1': 0.0 * np.ones_like(dL), 'chi_2': 0.0 * np.ones_like(dL),
            'luminosity_distance': dL,
            'geocent_time': geocent_time * np.ones_like(dL),
            'ra': ra,
            'dec': dec,
            'theta_jn': theta_jn_sample,
            'psi': psi_sample,
            'phase': phase_sample,
        }
    
    ### GWFish source code is modified by myself, but simpler calculation is working with the original code. ###
    ### For running the original code, see the documentation: https://gwfish.readthedocs.io/en/latest/tutorials/tutorial_170817.html ###
    params_df, network_snr, sky_area_deg2_90_list, ra_1sigma_error_list, dec_1sigma_error_list, corr_ra_dec_list, dL_1sigma_error_list = fisher_analysis_GWfish(param_dict)
    return params_df, network_snr, sky_area_deg2_90_list, ra_1sigma_error_list, dec_1sigma_error_list, corr_ra_dec_list, dL_1sigma_error_list

"""main function to compute errors and pick up some events based on the galaxy catalog and cosmological parameters"""
def compute_errors_and_pick_some_events(gal_cat_df, true_H0, omega_m, mag_cut_r, criteria, threshold, file_name, gw_sample_size=100, number_of_events_to_pick=None, random_seeds=None):
    """filter galaxy catalog by apparent magnitude"""
    galcat_df = gal_cat_df.copy()

    mag_cut_r = mag_cut_r
    mask = galcat_df['mag_true_r_lsst_no_host_extinction'] <= mag_cut_r
    filtered_df = galcat_df.loc[mask]
    true_redshifts_masked = galcat_df.loc[mask, 'redshift_true']

    """sample galaxies from the catalog according to weights"""
    sample_size = gw_sample_size
    weights = np.ones_like(true_redshifts_masked) / len(true_redshifts_masked) # uniform weighting
    # weights = 3.828e26 * 10**(-0.4 * galcat_df.loc[mask, 'Mag_true_r_lsst_z0_no_host_extinction']) / np.sum(3.828e26 * 10**(-0.4 * galcat_df.loc[mask, 'mag_true_r_lsst_no_host_extinction'])) # weighting by luminosity
    gal_random_seed = random_seeds[1] if random_seeds is not None else np.random.randint(0, 1000000)
    np.random.seed(gal_random_seed)
    print(f'random seed for galaxy sampling: {gal_random_seed}')
    sample_indices = np.random.choice(filtered_df.index, size=sample_size, replace=False, p=weights)
    sample_ra = np.array(filtered_df.loc[sample_indices, 'ra_true'])
    sample_dec = np.array(filtered_df.loc[sample_indices, 'dec_true'])
    sampled_true_z = np.array(filtered_df.loc[sample_indices, 'redshift_true'])
    sample_true_dL = luminosity_distance(sampled_true_z, true_H0, Om=omega_m)

    """compute network SNRs and sky localization areas"""
    params_df, network_snr, sky_area_deg2_90_list, ra_1sigma_error_list, dec_1sigma_error_list, corr_ra_dec_list, dL_1sigma_error_list = compute_network_SNR_and_sky_area(sample_true_dL, sample_ra, sample_dec)

    dL_random_seed = random_seeds[2] if random_seeds is not None else np.random.randint(0, 1000000)
    np.random.seed(dL_random_seed)
    print(f'random seed for distance measurement error: {dL_random_seed}')
    measured_dL = np.random.normal(loc=sample_true_dL, scale=dL_1sigma_error_list) # with error case
    # measured_dL = sample_true_dL # no error case
    # dL_1sigma_error_list = np.ones_like(measured_dL) * 1e-5 # no error case
    negative_mask = measured_dL <= 0
    while np.any(negative_mask):
        dL_random_seed += 1
        np.random.seed(dL_random_seed)
        print(f"Resampling {np.sum(negative_mask)} negative distance values..., random seed: {dL_random_seed}")
        measured_dL[negative_mask] = np.random.normal(loc=sample_true_dL[negative_mask], scale=dL_1sigma_error_list[negative_mask])
        negative_mask = measured_dL <= 0

    """pick up good events"""
    criteria = criteria
    number_of_events_to_pick = number_of_events_to_pick
    threshold = threshold

    dL_error_mask = dL_1sigma_error_list / sample_true_dL < 0.8 # only consider events with less than 80% distance error
    valid_indices_original = np.where(dL_error_mask)[0]
    ### impose threshold on sky area ###
    if criteria == 'impose_threthold_on_sky_area':
        sky_area_threshold = threshold  # deg^2
        valid_sky_areas = sky_area_deg2_90_list[valid_indices_original]
        relative_indices = np.where(valid_sky_areas <= sky_area_threshold)[0]
        best_indices = valid_indices_original[relative_indices]
        print(f"\n*** Choose events with sky area smaller than {sky_area_threshold} deg^2***")

    ### best snr###
    if criteria == 'network_snr':
        valid_snrs = network_snr[valid_indices_original]
        relative_indices = np.argsort(valid_snrs)[-number_of_events_to_pick:]
        best_indices = valid_indices_original[relative_indices]
        # print(f"\n*** Choose larger SNR events ***")

    ### best localization ###
    if criteria == 'sky_area_deg2_90':
        valid_sky_areas = sky_area_deg2_90_list[valid_indices_original]
        relative_indices = np.argsort(valid_sky_areas)[:number_of_events_to_pick]
        best_indices = valid_indices_original[relative_indices]
        print(f"\n*** Choose smaller sky area events ***")

    ### fiducial localization ###
    if criteria == 'fiducial_localization':
        fiducial_localization = 10.0  # deg^2
        best_indices = np.argsort(np.abs(sky_area_deg2_90_list-fiducial_localization))[:number_of_events_to_pick]
        print(f"\n*** Choose events with sky area closest to {fiducial_localization} deg^2 ***")

    print(f'\nfile name for this simulation: {file_name}')
    print(f'\nnumber of GW events: {len(best_indices)}')
    skymap_filename_list = []
    for i, best_index in enumerate(best_indices):
        print(f"\n------Event {i+1}--------")
        best_localization = sky_area_deg2_90_list[best_index]
        print(f"sky area (90% C.I.): {best_localization} deg^2")
        print(f"  Corresponding SNR: {network_snr[best_index]}")
        print(f"  Corresponding true and measured dL: {sample_true_dL[best_index]} [Mpc], {measured_dL[best_index]} [Mpc]")
        print(f"  Corresponding dL error: {dL_1sigma_error_list[best_index]} [Mpc], {dL_1sigma_error_list[best_index]/sample_true_dL[best_index]*100:.2f} %")
        print(f"Galaxy parameters of the event chosen here:")
        print(f"  Corresponding gal index in the catalog: {galcat_df.index[sample_indices[best_index]]}")
        print(f"  Corresponding true redshift: {sampled_true_z[best_index]}")
        print(f"  Corresponding RA and Dec: {sample_ra[best_index]}, {sample_dec[best_index]}")

        best_injection = params_df.iloc[best_index]
        print(" Injection parameters of the event chosen here:")
        for key, value in best_injection.items():
            print(f"  {key}: {value}")

        skymap_filename = f"./data/skymap_files/mock_skymap_{file_name}_event_{i+1}.fits"
        target_ra = best_injection['ra']
        target_dec = best_injection['dec']
        target_ra_rad = np.deg2rad(target_ra)
        target_dec_rad = np.deg2rad(target_dec)
        target_ra_err = ra_1sigma_error_list[best_index]
        target_dec_err = dec_1sigma_error_list[best_index]
        target_corr = corr_ra_dec_list[best_index]
        target_dL = measured_dL[best_index]
        target_dL_err = dL_1sigma_error_list[best_index]
        create_fits_skymap_assuming_gaussian(target_ra_rad, target_dec_rad, target_ra_err, target_dec_err, target_corr, target_dL, target_dL_err, filename=skymap_filename, nside=1024, be_nested=False)
        skymap_filename_list.append(skymap_filename)

    return filtered_df, skymap_filename_list

"""create a FITS skymap file assuming Gaussian distribution for ra, dec, and dL."""
def create_fits_skymap_assuming_gaussian(ra_val, dec_val, ra_err, dec_err, corr, dL_val, dL_err, filename="./data/skymap.fits", nside=1024, be_nested=False):
    npix = hp.nside2npix(nside)
    
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix = phi
    dec_pix = 0.5 * np.pi - theta

    d_ra = (ra_pix - ra_val)
    d_ra = (d_ra + np.pi) % (2 * np.pi) - np.pi # to keep within -pi to pi

    d_ra_scaled = d_ra * np.cos(dec_val)
    d_dec = dec_pix - dec_val

    sigma_x = ra_err * np.cos(dec_val)
    sigma_y = dec_err
    cov = np.array([
        [sigma_x**2, corr * sigma_x * sigma_y],
        [corr * sigma_x * sigma_y, sigma_y**2]
    ])
    
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    
    a = inv_cov[0, 0]
    b = inv_cov[0, 1]
    d = inv_cov[1, 1]
    
    exponent = -0.5 * (a * d_ra_scaled**2 + 2 * b * d_ra_scaled * d_dec + d * d_dec**2)
    prob = np.exp(exponent)
    
    prob_sum = np.sum(prob)
    if prob_sum > 0:
        prob /= prob_sum
    else:
        dist_ang = d_ra_scaled**2 + d_dec**2
        prob[np.argmin(dist_ang)] = 1.0

    distmu = np.ones(npix) * dL_val
    distsigma = np.ones(npix) * dL_err
    distnorm = np.ones(npix) / (distmu**2 + distsigma**2)

    col_prob = fits.Column(name='PROB', format='D', unit='pix-1', array=prob)
    col_distmu = fits.Column(name='DISTMU', format='D', unit='Mpc', array=distmu)
    col_distsigma = fits.Column(name='DISTSIGMA', format='D', unit='Mpc', array=distsigma)
    col_distnorm = fits.Column(name='DISTNORM', format='D', unit='Mpc-2', array=distnorm)

    cols = fits.ColDefs([col_prob, col_distmu, col_distsigma, col_distnorm])
    hdu = fits.BinTableHDU.from_columns(cols)

    hdu.header['PIXTYPE'] = 'HEALPIX'
    hdu.header['COORDSYS'] = 'C' # Celestial (Equatorial)
    hdu.header['NSIDE'] = nside
    hdu.header['INDXSCHM'] = 'IMPLICIT'
    hdu.header['OBJECT'] = 'MOCK_EVENT'
    
    if be_nested:
        # transform from RING to NESTED
        prob_nested = hp.reorder(prob, r2n=True)
        distmu_nested = hp.reorder(distmu, r2n=True)
        distsigma_nested = hp.reorder(distsigma, r2n=True)
        distnorm_nested = hp.reorder(distnorm, r2n=True)

        col_prob = fits.Column(name='PROB', format='D', unit='pix-1', array=prob_nested)
        col_distmu = fits.Column(name='DISTMU', format='D', unit='Mpc', array=distmu_nested)
        col_distsigma = fits.Column(name='DISTSIGMA', format='D', unit='Mpc', array=distsigma_nested)
        col_distnorm = fits.Column(name='DISTNORM', format='D', unit='Mpc-2', array=distnorm_nested)

        cols = fits.ColDefs([col_prob, col_distmu, col_distsigma, col_distnorm])
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header['PIXTYPE'] = 'HEALPIX'
        hdu.header['ORDERING'] = 'NESTED'
        hdu.header['COORDSYS'] = 'C'
        hdu.header['NSIDE'] = nside
        hdu.header['INDXSCHM'] = 'IMPLICIT'
        hdu.header['OBJECT'] = 'MOCK_EVENT'

        print(f"Generated FITS file: {filename} (NESTED)")        
    else:
        hdu.header['ORDERING'] = 'RING'
        print(f"Generated FITS file: {filename} (RING)")

    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(filename, overwrite=True)

"""fuctions for likelihood calculations"""
def z_from_dL(dL, H0, Om):
    H0 = H0
    Om0 = Om
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    if np.isscalar(dL):
         z = z_at_value(cosmo.luminosity_distance, dL * u.Mpc, zmin=0.0)
    else:
         z = np.array([z_at_value(cosmo.luminosity_distance, d_i * u.Mpc, zmin=0.0) for d_i in dL])
    return z

def E(z, Om):
    cosmo = FlatLambdaCDM(H0=70, Om0=Om)
    return 1 / cosmo.efunc(z)

def comoving_distance(z_array, H0, Om):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    return cosmo.comoving_distance(z_array).value

def luminosity_distance(z_array, H0, Om):
    H0 = H0
    Om = Om
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    return cosmo.luminosity_distance(z_array).value

def create_cosmo_interpolator(H0):
    z_table = np.linspace(1e-10, 10.0, 10000)
    dl_table = luminosity_distance(z_table, H0, 10000)
    return CubicSpline(z_table, dl_table)

def gauss(x, m, s, n=1):
    return n*(s*np.sqrt(2*np.pi))**(-1)*np.exp(-0.5*((x-m)/s)**2)

def log_gauss(x, m, s, n=1):
    return np.log(n) -0.5 * ((x - m) / s)**2 - np.log(s) - 0.5 * np.log(2 * np.pi)

def madau(z, gamma=4.59, k=2.86, zp=2.47):
    return (1+((1+zp)**(-gamma-k)))*((1+z)**(gamma-1))/(1+(((1+z)/(1+zp))**(gamma+k)))

def create_V_dL_max_GW_interpolated(H0_array):
    """
    Using distancetool to calculate the maximum volume.
    unit: Mpc^3
    """
    m1 = 30.0  # Solar masses
    m2 = 30.0  # Solar masses
    network = ['H','L','V']
    asdfile_list = [
                    # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/aligo_O3actual_H1.txt"),
                    # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/aligo_O3actual_L1.txt"),
                    # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/avirgo_O3actual.txt")
                    os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/aLIGO_O4_high_asd.txt"),
                    os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/aLIGO_O4_high_asd.txt"),
                    os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/AdV_asd.txt")
                   ]
    pwfile = 'o3_120_60_hlv_bbh_30_30_imrD'
    omega_m = 0.3
    omega_de = 0.7
    omega_k = 0

    volume_array = []
    for H0 in tqdm(H0_array):
        H0 = H0
        result = calculate_range_from_distancetool(m1, m2, network, asdfile_list, pwfile, omega_m, omega_de, omega_k, H0=H0)
        max_range = result[0]
        volume_array.append((4/3) * np.pi * (max_range**3))
    return CubicSpline(H0_array, volume_array)
    
def V_dL_GW_max(H0, approx=False):
    """
    Using distancetool to calculate the maximum volume.
    unit: Mpc^3
    """
    if not approx:
        m1 = 30.0  # Solar masses
        m2 = 30.0  # Solar masses
        network = ['H','L','V']
        asdfile_list = [
                        # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/aligo_O3actual_H1.txt"),
                        # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/aligo_O3actual_L1.txt"),
                        # os.path.join(os.path.expanduser("~"),"Development/distancetool/data/avirgo_O3actual.txt")
                        os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/aLIGO_O4_high_asd.txt"),
                        os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/aLIGO_O4_high_asd.txt"),
                        os.path.join(os.path.expanduser("~"),"Development/distancetool/data/from_bilby/AdV_asd.txt")
                       ]
        pwfile = 'o3_120_60_hlv_bbh_30_30_imrD'
        omega_m = 0.3
        omega_de = 0.7
        omega_k = 0

        H0 = H0
        result = calculate_range_from_distancetool(m1, m2, network, asdfile_list, pwfile, omega_m, omega_de, omega_k, H0=H0)
        max_range = result[0]

        volume = (4/3) * np.pi * (max_range**3)
    else:
        volume = H0**(3)
    return volume

def log_likelihood_rapid(z_array, H0, Om, dl_interp, gal_z, gal_zsigma, gal_m, gal_p, gal_distmu, gal_distsig, gal_distnorm, V_dL_GW_max_interpolated, alpha_selection_function=None, chunk_size=1000):
    
    
    # log_p_rate = np.log(madau(z_array))
    log_p_rate = np.log(np.ones(len(z_array))) # Uniform in comoving volume and source-frame time
    
    # lum_mask = gal_absmag<M_max+5*np.log10(H0/100)
    
    z_matrix = z_array[np.newaxis, :] # shape (1, N_z)
    log_z_step = np.log(z_array[1]-z_array[0])

    dl_array = dl_interp(z_array) # shape (N_z,)
    dl_array = np.maximum(dl_array, 1e-10) # to avoid zero or negative distances
    dL_matrix = dl_array[np.newaxis, :] # shape (1, N_z)
    
    log_integrand_array = np.log((dl_array/(1+z_array))**2 * c*1e-3 / (H0*E(z_array, Om))) # shape (N_z,)
    log_integrand_matrix = log_integrand_array[np.newaxis, :] # shape (1, N_z)

    log_likelihood_list = []
    n_gal = len(gal_z)
    chunk_size = chunk_size
    for i in range(0, n_gal, chunk_size):
        end_i = min(i + chunk_size, n_gal)

        g_z = gal_z[i:end_i, np.newaxis]       # shape (N_chunk, 1)
        g_sig = gal_zsigma[i:end_i, np.newaxis] # shape (N_chunk, 1)
        g_mu = gal_distmu[i:end_i, np.newaxis] # shape (N_chunk, 1)
        g_dsig = gal_distsig[i:end_i, np.newaxis] # shape (N_chunk, 1)
        g_dnorm = gal_distnorm[i:end_i, np.newaxis] # shape (N_chunk, 1)
        g_p = gal_p[i:end_i, np.newaxis] # shape (N_chunk, 1)
        g_m = gal_m[i:end_i, np.newaxis] # shape (N_chunk, 1)
        
        g_absmag = g_m - 5*np.log10(dL_matrix*1e6) + 5 # shape (N_chunk, N_z), For high z, k corrections are needed here
        # log_gal_lum = np.log(3.828e26*10**(-0.4*(gal_absmag))) # Luminosity weighting
        log_gal_lum = np.log(np.ones_like(g_absmag)) # shape (N_chunk,), Uniform weighting

        log_p_em = log_gauss(z_matrix, g_z, g_sig) + log_p_rate + log_gal_lum # shape (N_chunk, N_z)
        log_p_gw = log_gauss(dL_matrix, g_mu, g_dsig, n=g_dnorm * g_p) # shape (N_chunk, N_z)

        log_evidence_integrand = log_p_em + log_integrand_matrix # (N_chunk, N_z)
        log_evidence = logsumexp(log_evidence_integrand, axis=1) + log_z_step
        log_numerator_integrand = log_p_em + log_p_gw + log_integrand_matrix # (N_chunk, N_z)
        log_numerator = logsumexp(log_numerator_integrand, axis=1) + log_z_step

        valid = log_evidence != -np.inf
        log_likelihood_list.append(logsumexp(log_numerator[valid] - log_evidence[valid]))
    if alpha_selection_function is not None:
        log_beta = alpha_selection_function * np.log(H0)
    else:
        log_beta = np.log(H0**(3)) + np.log(V_dL_GW_max_interpolated(H0))
        # log_beta = 0. # no selection effect
    total_log_likelihood = logsumexp(np.array(log_likelihood_list)) - log_beta
    return total_log_likelihood

def worker_h0_likelihood(idx, h0_val,  omega_m, interpolator, z_array, masked_gal_measured_z, masked_gal_z_sigma, masked_gal_mr, gal_p, gal_distmu, gal_distsig, gal_distnorm, V_dL_GW_max_interpolated, alpha_selection_function=None):
    val = log_likelihood_rapid(
        z_array,
        h0_val,
        omega_m,
        interpolator,
        masked_gal_measured_z,
        masked_gal_z_sigma,
        masked_gal_mr,
        gal_p,
        gal_distmu,
        gal_distsig,
        gal_distnorm,
        V_dL_GW_max_interpolated,
        alpha_selection_function=alpha_selection_function,
        chunk_size=100,
    )
    return val

def run_single_simulation(gal_df, H0_array, H0_interpolators, true_H0, omega_m, mag_cut_r, criteria, threshold, file_name, gw_sample_size, number_of_events_to_pick, V_dL_GW_max_interpolated, alpha_selection_function=None, random_seeds=None):

    """generate GW data and skymap files"""
    mag_cut_r = mag_cut_r
    criteria = criteria
    threshold = threshold
    gw_sample_size = gw_sample_size
    number_of_events_to_pick = number_of_events_to_pick
    print('Generating GW data and skymap files...')
    print(f' gw sample size: {gw_sample_size}')
    print(f' criteria for picking events: {criteria}')
    print(f' threshold for picking events: {threshold}')
    print(f' number of events to pick: {number_of_events_to_pick}')
    filtered_df, skymap_filename_list = compute_errors_and_pick_some_events(
                                                                gal_cat_df=gal_df,
                                                                true_H0=true_H0,
                                                                omega_m=omega_m,
                                                                mag_cut_r=mag_cut_r,
                                                                criteria=criteria,
                                                                threshold=threshold,
                                                                file_name=file_name,
                                                                gw_sample_size=gw_sample_size,
                                                                number_of_events_to_pick=number_of_events_to_pick,
                                                                random_seeds=random_seeds
                                                                )
    
    """load galaxy catalog filtered by apparent magnitude in r band"""
    galaxy_df = filtered_df.copy()
    gal_ra = np.array(galaxy_df['ra_true'][:])
    gal_dec = np.array(galaxy_df['dec_true'][:])
    gal_mr = np.array(galaxy_df['mag_true_r_lsst_no_host_extinction'][:])
    gal_true_z = np.array(galaxy_df['redshift_true'][:])
    gal_measured_z = np.array(galaxy_df['redshift_measured'][:])
    gal_z_sigma = np.array(galaxy_df['redshift_sigma'][:])

    log_like_list_all_events = []
    for i in tqdm(range(len(skymap_filename_list))):
        print(f"\nProcessing event {i+1}...")
        GW_sky_map_file_path = skymap_filename_list[i]

        """GW data"""
        p, distmu, distsig, distnorm = hp.read_map(GW_sky_map_file_path, field=[0, 1, 2, 3]) # should be RING, not NEST, order of fields: PROB, DISTMU, DISTSIGMA, DISTNORM
        nside = hp.npix2nside(len(p))
    
        """make galaxy selections"""
        ### distance ###
        print('calculating distance cut...')
        # percentile_distance = 1.0 # 1 sigma
        # percentile_distance = 1.28155 # 80%
        percentile_distance = 1.64485 # 90%
        # percentile_distance = 1.95996 # 95%
        # percentile_distance = 5.0 # 5sigma
        gal_dL_for_largest_H0 = luminosity_distance(gal_measured_z-percentile_distance*gal_z_sigma, H0=np.max(H0_array), Om=omega_m) # for conservative higher cutoff, gal z is evaluated at lower end of the z distribution, and H0 is evaluated at the largest value
        gal_dL_for_smallest_H0 = luminosity_distance(gal_measured_z+percentile_distance*gal_z_sigma, H0=np.min(H0_array), Om=omega_m) # for conservative lower cutoff, gal z is evaluated at upper end of the z distribution for lower, and H0 is evaluated at the smallest value
        dist_max_thresh = distmu[np.argmax(p)] + percentile_distance * distsig[np.argmax(p)] # for now, all pixels has same dL distribution
        dist_min_thresh = distmu[np.argmax(p)] - percentile_distance * distsig[np.argmax(p)] # for now, all pixels has same dL distribution
        distance_mask = (gal_dL_for_smallest_H0>dist_min_thresh) & (gal_dL_for_largest_H0<dist_max_thresh) # in 90% credible interval

        ### sky position ###
        print('calculating sky position cut...')
        sorted_p = np.sort(p)[::-1]
        cumsum_p = np.cumsum(sorted_p)
        # percentile_sky = 0.64
        percentile_sky = 0.9
        idx_sky_thr = np.searchsorted(cumsum_p, percentile_sky)
        prob_threshold_90 = sorted_p[idx_sky_thr]
        gal_hpx_idx_all = hp.ang2pix(nside, gal_ra, gal_dec, lonlat=True, nest=False)
        gal_p_all = p[gal_hpx_idx_all]
        is_in_90Sky_mask = gal_p_all >= prob_threshold_90

        full_mask = distance_mask & is_in_90Sky_mask

        print(f"Number of galaxies after applying distance and sky position cuts: {np.sum(full_mask)} out of {len(galaxy_df)}")

        masked_gal_ra = gal_ra[full_mask]
        masked_gal_dec = gal_dec[full_mask]
        masked_gal_true_z = gal_true_z[full_mask]
        masked_gal_measured_z = gal_measured_z[full_mask]
        masked_gal_z_sigma = gal_z_sigma[full_mask]
        masked_gal_mr = gal_mr[full_mask]

        gal_hpx_idx = hp.ang2pix(nside, masked_gal_ra, masked_gal_dec, lonlat=True, nest=False)
        gal_p = p[gal_hpx_idx]
        gal_distmu = distmu[gal_hpx_idx]
        gal_distsig = distsig[gal_hpx_idx]
        gal_distnorm = distnorm[gal_hpx_idx]

        """set up redshift array"""
        ### determine z_array maximum ###
        z_array_max_gal = np.max(masked_gal_measured_z + 5*masked_gal_z_sigma) * 1.1
        z_array_max_GW = z_from_dL(distmu[np.argmax(p)]+5*distsig[np.argmax(p)], H0=np.max(H0_array), Om=omega_m) * 1.1
        z_array_max = np.max([z_array_max_gal, z_array_max_GW])

        ### determine z_array grid size ###
        min_sigma_GW = z_from_dL(distsig[np.argmax(p)], H0=np.min(H0_array), Om=omega_m)
        min_sigma_z = np.min(masked_gal_z_sigma)
        dz_required = np.min([min_sigma_GW, min_sigma_z]) / 5
        N_grid = np.max([int(z_array_max / dz_required), 1000])
        z_array = np.linspace(1e-5, z_array_max, N_grid)
        print(f'redshift array: {z_array[0]} to {z_array[-1]}')
        print(f'number of redshift grid points: {len(z_array)}')

        """Evaluate H0 likelihood"""
        log_likelihood_list = []
        # total_cores = os.cpu_count()
        # core_margin = 2
        # n_jobs = max(1, total_cores - core_margin)
        # n_jobs = 5 # total physical cores is 10
        n_jobs = 25 # for cluster
        print("Starting parallel likelihood calculation...")
        log_likelihood_list = Parallel(n_jobs=n_jobs)(
            delayed(worker_h0_likelihood)(
                j, H0_array[j], omega_m, H0_interpolators[j], z_array,
                masked_gal_measured_z, masked_gal_z_sigma, masked_gal_mr,
                gal_p, gal_distmu, gal_distsig, gal_distnorm,
                V_dL_GW_max_interpolated, alpha_selection_function=alpha_selection_function
            ) for j in tqdm(range(len(H0_array)))
        )
        
        log_like = np.array(log_likelihood_list)
        log_like_list_all_events.append(log_like)
    return log_like_list_all_events

if __name__ == "__main__":
    start_time = time.time()

    """parameters for GW data generation and event selection"""
    mag_cut_r = 21
    gw_sample_size = 100
    number_of_events_to_pick = None

    criteria = 'impose_threthold_on_sky_area' # criteria for selection function estimation
    threshold = 5.0 # deg^2
    # criteria = 'sky_area_deg2_90' # criteria for selection function estimation
    # threshold = 80.0 # deg^2
    # criteria = 'network_snr' # criteria for selection function estimation
    # threshold = 8.0 # SNR

    deltaz = 0.04 # photometric redshift error
    # deltaz = 0.0004 # spectroscopic redshift error

    random_seeds = [0, 1, 2] # for gal_z, true GW z, measured dL
    # random_seeds = [np.random.randint(0, 100) for _ in range(3)]
    # random_seeds = None

    if number_of_events_to_pick is None:
        file_name = f'deltaz_{deltaz}_mag_cut_r_{mag_cut_r}_criteria_{criteria}_threshold_{threshold}_random_seed_{random_seeds[0]}-{random_seeds[1]}-{random_seeds[2]}'
    else:
        file_name = f'deltaz_{deltaz}_mag_cut_r_{mag_cut_r}_criteria_{criteria}_num_events_{number_of_events_to_pick}_random_seed_{random_seeds[0]}-{random_seeds[1]}-{random_seeds[2]}'
    
    """setting log file"""
    outdir = f"./outdirs/outdir_{file_name}"
    os.makedirs(outdir, exist_ok=True)
    log_file = outdir + f"/{file_name}.log"
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file)    
    tqdm.pandas(file=original_stdout)

    """cosmological parameters used in the simulation"""
    h = 0.71
    omega_cdm = 0.1109
    omega_b = 0.02258
    n_s = 0.963
    sigma_8 = 0.8
    w = -1.0

    omega_m = (omega_cdm + omega_b) / (h ** 2)
    omega_lambda = 1.0 - omega_m
    true_H0 = h * 100.0  # in unit of km/s/Mpc
    print("Cosmological parameters used in the simulation:")
    print(f"H0: {true_H0} km/s/Mpc")
    print(f"Omega_m: {omega_m}")
    print(f"Omega_lambda: {omega_lambda}")

    """load galaxy data"""
    # gal_catalog_path = "./data/small_fiducial.parquet"
    gal_catalog_path = "./data/large_sim_gal_cat.parquet"
    gal_catalog_df = pd.read_parquet(gal_catalog_path)
    print('-------columns--------')
    print(gal_catalog_df.columns)
    print('catalog num: ' + str(len(gal_catalog_df)))

    """set delta z"""
    deltaz = deltaz
    gal_z_sigma = deltaz * (1 + gal_catalog_df['redshift_true'])
    gal_measured_z_ramdom_seed = random_seeds[0] if not random_seeds is None else np.random.randint(0, 10000)
    np.random.seed(gal_measured_z_ramdom_seed)
    print(f'random seed for measured redshift: {gal_measured_z_ramdom_seed}')
    gal_catalog_df['redshift_measured'] = np.random.normal(gal_catalog_df['redshift_true'], gal_z_sigma)
    gal_catalog_df['redshift_sigma'] = gal_z_sigma

    while np.any(gal_catalog_df['redshift_measured'] <= 0):
        gal_measured_z_ramdom_seed = gal_measured_z_ramdom_seed + 1
        np.random.seed(gal_measured_z_ramdom_seed)
        print(f'random seed for measured redshift resampling: {gal_measured_z_ramdom_seed}')
        mask = gal_catalog_df['redshift_measured'] <= 0
        gal_catalog_df.loc[mask, 'redshift_measured'] = np.random.normal(gal_catalog_df.loc[mask, 'redshift_true'], gal_catalog_df.loc[mask, 'redshift_sigma'])

    """set up H0 array"""
    h0_array_params = (20, 140, 100) # H0 grid for likelihood eval
    h0_array = np.linspace(*h0_array_params)
    
    """Create array of splines for luminosity distance"""
    interp_file = './data/h0_interpolators_{}_{}_{}.pickle'.format(*h0_array_params)
    if os.path.exists(interp_file):
        with open(interp_file, 'rb') as f:
            h0_interpolators = pickle.load(f)
    else:
        h0_interpolators = []
        for i in tqdm(range(len(h0_array))):
            h0_interpolators.append(create_cosmo_interpolator(h0_array[i]))
        with open(interp_file, 'wb') as f:
            pickle.dump(h0_interpolators, f)
    
    """Create array of splines for V_dL_GW_max if not exist"""
    interp_file = './data/V_dL_max_GW_interpolated_{}_{}_{}.pickle'.format(*h0_array_params)
    if os.path.exists(interp_file):
        with open(interp_file, 'rb') as f:
            V_dL_GW_max_interpolated = pickle.load(f)
    else:
        print('Creating V_dL_GW_max_interpolated...')
        V_dL_GW_max_interpolated = create_V_dL_max_GW_interpolated(h0_array)
        with open(interp_file, 'wb') as f:
            pickle.dump(V_dL_GW_max_interpolated, f)

    """Calculate selection function from injections"""
    h0_array_for_selection_function = np.arange(20, 140, 1)
    p_det_array = np.zeros(len(h0_array_for_selection_function))
    print('\nCalculating selection function from injections...')
    for i in tqdm(range(len(h0_array_for_selection_function))):
        injection_file_path = f'./data/injections/injection_10000_magcut_{mag_cut_r}_H0_{h0_array_for_selection_function[i]:.3f}_for_selection_function.h5'
        with h5py.File(injection_file_path, 'r') as f:
            if criteria == 'sky_area_deg2_90' or criteria == 'impose_threthold_on_sky_area':
                val_array = f['sky_area_deg2_90'][:]
                selected_mask = val_array <= threshold
            if criteria == 'network_snr':
                val_array = f['network_snr'][:]
                selected_mask = val_array >= threshold
        p_det = np.sum(selected_mask) / len(val_array)
        p_det_array[i] = p_det
    x = h0_array_for_selection_function
    y = p_det_array
    mask = (x > 0) & (y > 0)
    logx = np.log(x[mask])
    logy = np.log(y[mask])
    alpha, logA = np.polyfit(logx, logy, 1)
    A = np.exp(logA)
    p_det_fit = A * x**alpha

    """run single simulation"""
    log_like_list_all_events = run_single_simulation(
                                gal_df=gal_catalog_df, 
                                H0_array=h0_array,
                                H0_interpolators=h0_interpolators,
                                true_H0=true_H0,
                                omega_m=omega_m, 
                                mag_cut_r=mag_cut_r,
                                criteria=criteria,
                                threshold=threshold,
                                file_name=file_name,
                                gw_sample_size=gw_sample_size,
                                number_of_events_to_pick=number_of_events_to_pick,
                                V_dL_GW_max_interpolated=V_dL_GW_max_interpolated,
                                alpha_selection_function=alpha,
                                random_seeds=random_seeds
                                )
    
    """Combine likelihoods from all events"""
    if len(log_like_list_all_events) > 1:
        total_log_like = np.sum(np.array(log_like_list_all_events), axis=0)
    else:
        total_log_like = log_like_list_all_events[0]
    like = np.exp(total_log_like - logsumexp(total_log_like))

    """Plot H0 likelihood"""
    plt.style.use('~/research/my_plot_style.style')
    plt.figure()
    plt.plot(h0_array, like, color='blue', lw=2, marker='o')
    plt.axvline(x=true_H0, color='red', ls='--', label=r'True $H_0$')
    plt.xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    plt.ylabel('Posterior Density')
    plt.grid(True)
    # plt.show()
    # save_plot_path = f'./figs/H0_likelihood_mag_cut_r_{mag_cut_r}_criteria_{criteria}_{number_of_events_to_pick}events_dL_error<0.8_alpha_{alpha:.2f}_v5.pdf'
    save_plot_path = f'{outdir}/H0_likelihood_{file_name}.pdf'
    plt.savefig(save_plot_path, dpi=200, bbox_inches='tight')
    print(f'Saved H0 likelihood figure to {save_plot_path}')

    """Save likelihood data to file"""
    output_file = f"{outdir}/likelihood_{file_name}.h5"
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('H0', data=h0_array, dtype='f4', compression='gzip')
        f.create_dataset('likelihood', data=like, dtype='f4', compression='gzip')
        f.create_dataset('log_likelihood_all_events', data=log_like_list_all_events, dtype='f4', compression='gzip')
    print(f'Saved likelihood data to {output_file}')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total runtime: {str(datetime.timedelta(seconds=elapsed))}")