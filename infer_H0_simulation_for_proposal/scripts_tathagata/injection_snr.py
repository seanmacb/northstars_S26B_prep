#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import bilby
import pandas as pd
import sys
sys.path.append ('/home/tathagata.ghosh/cosmo_well_localized_events/models')
from mass_model import *
from astropy.cosmology import FlatLambdaCDM
import GWFish.modules as gw
from GWFish.modules.fishermatrix import compute_network_errors
from GWFish.modules.detection import Network

galaxy_data = pd.read_csv ('/home/tathagata.ghosh/cosmo_well_localized_events/proposal_study/galaxy_data/data_medium.csv', sep=',')

z_all = galaxy_data['redshift_true'].to_numpy()
ra_all = galaxy_data['ra_true'].to_numpy()
dec_all = galaxy_data['dec_true'].to_numpy()
appmag_r_all = galaxy_data['mag_true_r_lsst_no_host_extinction'].to_numpy()

mask_obs = appmag_r_all<=21

z_true = z_all[mask_obs]
ra_true = ra_all[mask_obs]*np.pi/180
dec_true = dec_all[mask_obs]*np.pi/180


nsamples = 10000

np.random.seed (99)

gal_host_id = np.random.randint (0, z_true.size, nsamples)

bbh_priors = bilby.gw.prior.BBHPriorDict(filename='precessing_spins_bbh.prior')

bbh_samples = bbh_priors.sample (nsamples)


theta_jn = bbh_samples['theta_jn']
psi = bbh_samples['psi']
phase = bbh_samples['phase']
phi_12 = bbh_samples['phi_12']
phi_jl = bbh_samples['phi_jl']
a_1 = bbh_samples['a_1']
a_2 = bbh_samples['a_2']



alpha = 2.90
beta = 1.04
mminbh = 5
mmaxbh = 80
lambda_g = 0.38
lambda_g_low = 0.84
mu_g_low = 9.67
sigma_g_low = 0.74
mu_g_high = 30.65
sigma_g_high = 6.30
delta_m = 4.82

ra = ra_true[gal_host_id]
dec = dec_true[gal_host_id]

mass_prior_param = BBH_multi_peak_gaussian(alpha,beta,mminbh,mmaxbh,lambda_g,lambda_g_low,mu_g_low,sigma_g_low,mu_g_high,sigma_g_high,delta_m)
m1, m2 = mass_prior_param.sample(nsamples)

z = z_true[gal_host_id]

m1z = m1*(1+z)
m2z = m2*(1+z)

mcz = bilby.gw.conversion.component_masses_to_chirp_mass (m1z, m2z)
q = m2/m1


H0_array = np.arange (10, 201, 1)

H0val = H0_array[int(sys.argv[1])]

print ('H0=', H0val)

cosmo = FlatLambdaCDM (H0=H0val, Om0=0.3, Tcmb0=2.725)
dl = cosmo.luminosity_distance(z).value

parameters = {
    'chirp_mass': mcz,
    'mass_ratio': q,
    'luminosity_distance': dl,
    'theta_jn': theta_jn,
    'ra': ra,
    'dec': dec,
    'psi': psi,
    'phase': phase,
    'geocent_time': np.zeros (q.size),
    'a_1':a_1,
    'a_2':a_2}

parameters = pd.DataFrame(parameters)

fisher_params = ['luminosity_distance', 'ra', 'dec', 'theta_jn', 'chirp_mass', 'mass_ratio', 'psi', 'phase', 'a_1', 'a_2']

network = Network(['LHO', 'LLO', 'VIR'])

_, snr, _, _ = compute_network_errors(network, parameters, fisher_parameters=fisher_params, waveform_model='IMRPhenomD')

result_path = '/home/tathagata.ghosh/cosmo_well_localized_events/proposal_study/H0inference_trial/injections_out'

snr_df = pd.DataFrame(np.column_stack([snr]), columns=['snr'])

snr_df.to_csv (f'{result_path}/inj_{H0val}.txt', index=False, sep=' ')
