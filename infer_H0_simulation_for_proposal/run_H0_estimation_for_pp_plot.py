from io import RawIOBase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import os
import pickle
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
from run_H0_estimation import create_cosmo_interpolator, create_V_dL_max_GW_interpolated, run_single_simulation, Logger
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest, beta
import h5py

def calculate_p_val_from_likelihood(likelihood_array, h0_array, true_h0):
    cdf = cumulative_trapezoid(likelihood_array, h0_array, initial=0)
    if cdf[-1] == 0:
        return None
        
    cdf /= cdf[-1]
    f_cdf = interp1d(h0_array, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
    p_val = f_cdf(true_h0)
    return p_val

start_time = time.time()

"""parameters for selection function estimation and H0 estimation simulation"""
mag_cut_r = 21
gw_sample_size = 100
number_of_events_to_pick = None

criteria = 'sky_area_deg2_90' # criteria for selection function estimation
threshold = 10.0 # deg^2
# criteria = 'network_snr' # criteria for selection function estimation
# threshold = 8.0 # SNR

deltaz = 0.04 # photometric redshift error
# deltaz = 0.0004 # spectroscopic redshift error

random_seeds = [0, 1, 2] # for gal_z, true GW z, measured dL

if number_of_events_to_pick is None:
    file_name = f'deltaz_{deltaz}_mag_cut_r_{mag_cut_r}_criteria_{criteria}_threshold_{threshold}_random_seed_{random_seeds[0]}-{random_seeds[1]}-{random_seeds[2]}'
else:
    file_name = f'deltaz_{deltaz}_mag_cut_r_{mag_cut_r}_criteria_{criteria}_num_events_{number_of_events_to_pick}_random_seed_{random_seeds[0]}-{random_seeds[1]}-{random_seeds[2]}'

"""set up H0 samples for simulation"""
H0_sample_size = 100
H0_prior_min = 20
H0_prior_max = 140
H0_samples = np.random.uniform(H0_prior_min, H0_prior_max, H0_sample_size)
# H0_samples = 71.0 * np.ones(H0_sample_size)
omega_m = 0.2647887323943662

"""setting log file"""
outdir = f"./outdirs/outdir_pp_plot_H0_{H0_prior_min}-{H0_prior_max}_{H0_sample_size}iterations_{file_name}"
os.makedirs(outdir, exist_ok=True)
log_file = outdir + f"/pp_plot_H0_{H0_prior_min}-{H0_prior_max}_{H0_sample_size}iterations_{file_name}.log"
original_stdout = sys.stdout
sys.stdout = Logger(log_file)    
tqdm.pandas(file=original_stdout)

"""load galaxy data"""
gal_catalog_path = "./data/small_fiducial.parquet"
gal_cat_df = pd.read_parquet(gal_catalog_path)
print('-------columns--------')
print(gal_cat_df.columns)
print('catalog num: ' + str(len(gal_cat_df)))

"""set delta z"""
deltaz = deltaz
gal_z_sigma = deltaz * (1 + gal_cat_df['redshift_true'])
gal_measured_z_ramdom_seed = random_seeds[0] if not random_seeds is None else np.random.randint(0, 10000)
np.random.seed(gal_measured_z_ramdom_seed)
print(f'random seed for measured redshift: {gal_measured_z_ramdom_seed}')
gal_cat_df['redshift_measured'] = np.random.normal(gal_cat_df['redshift_true'], gal_z_sigma)
gal_cat_df['redshift_sigma'] = gal_z_sigma

while np.any(gal_cat_df['redshift_measured'] <= 0):
    gal_measured_z_ramdom_seed = gal_measured_z_ramdom_seed + 1
    np.random.seed(gal_measured_z_ramdom_seed)
    print(f'random seed for measured redshift resampling: {gal_measured_z_ramdom_seed}')
    mask = gal_cat_df['redshift_measured'] <= 0
    gal_cat_df.loc[mask, 'redshift_measured'] = np.random.normal(gal_cat_df.loc[mask, 'redshift_true'], gal_cat_df.loc[mask, 'redshift_sigma'])

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

"""create selection function from injections"""
h0_array_for_selection_function = np.arange(20, 140, 1)
p_det_array = np.zeros(len(h0_array_for_selection_function))
print('\nCalculating selection function from injections...')
for i in tqdm(range(len(h0_array_for_selection_function))):
    injection_file_path = f'./data/injections/injection_10000_magcut_{mag_cut_r}_H0_{h0_array_for_selection_function[i]:.3f}_for_selection_function.h5'
    with h5py.File(injection_file_path, 'r') as f:
        if criteria == 'sky_area_deg2_90':
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

"""Checkpoint Setup"""
checkpoint_file = f'{outdir}/pp_plot_H0_{H0_prior_min}-{H0_prior_max}_{H0_sample_size}iterations_{file_name}.pickle'
if os.path.exists(checkpoint_file):
    print(f"\n[Checkpoint Found] Loading previous state from {checkpoint_file}...")
    with open(checkpoint_file, 'rb') as f:
        cp_data = pickle.load(f)
    H0_samples = cp_data['H0_samples']
    p_values = cp_data['p_values']
    likelihoods = cp_data['likelihoods']
    start_index = len(p_values)
    print(f"Resuming from iteration {start_index + 1} / {H0_sample_size}...")
else:
    print("\n[No Checkpoint] Starting a fresh simulation...")
    H0_samples = H0_samples.tolist()  # Convert to list for easier appending
    p_values = []
    likelihoods = []
    start_index = 0

for i in range(start_index, len(H0_samples)):
    """cosmological parameters used in the simulation"""
    omega_m = omega_m
    omega_lambda = 1.0 - omega_m
    true_H0 = H0_samples[i]  # in unit of km/s/Mpc

    time_now = time.time()
    elapsed = time_now - start_time
    print(f"\n------{i+1}/{len(H0_samples)}: runtime={str(datetime.timedelta(seconds=elapsed))}------")
    print(f"Cosmological parameters used in the simulation:")
    print(f" H0: {true_H0} km/s/Mpc")
    print(f" Omega_m: {omega_m}")
    print(f" Omega_lambda: {omega_lambda}")

    """run single simulation"""
    like = run_single_simulation(gal_df=gal_cat_df,
                                 H0_array=h0_array,
                                 H0_interpolators=h0_interpolators,
                                 true_H0=true_H0,
                                 omega_m=omega_m, 
                                 mag_cut_r=mag_cut_r,
                                 criteria=criteria,
                                 threshold=threshold,
                                 gw_sample_size=gw_sample_size,
                                 number_of_events_to_pick=number_of_events_to_pick,
                                 V_dL_GW_max_interpolated=V_dL_GW_max_interpolated,
                                 alpha_selection_function=alpha,
                                 random_seeds=random_seeds
                                 )
    likelihoods.append(like)

    """calculate p-value for true H0"""
    p_val = calculate_p_val_from_likelihood(like, h0_array, true_H0)
    p_values.append(p_val)

    """ --- Save Checkpoint (Atomic Save) --- """
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, 'wb') as f:
        pickle.dump({
            'H0_samples': H0_samples,
            'p_values': p_values,
            'likelihoods': likelihoods
        }, f)
    os.replace(temp_file, checkpoint_file) 
    print(f"\n--> Saved checkpoint for iteration {i+1}")

p_values = np.array(p_values)
sorted_p = np.sort(p_values)
cdf_theoretical = np.linspace(0, 1, len(sorted_p))
ks_stat, ks_p_val = kstest(p_values, 'uniform')
N = len(sorted_p)
k = np.arange(1, N + 1)
sigma_levels = [1, 2, 3]
alphas = [0.68, 0.95, 0.997]

plt.style.use('~/research/my_plot_style.style')
plt.figure()
plt.plot(np.linspace(0,1, 1000), np.linspace(0,1, 1000), 'k--', label='ideal', lw=2)
plt.plot(sorted_p, cdf_theoretical, label=fr'$p$ value$={ks_p_val:.3f}$', lw=2)

for sigma, alpha in zip(sigma_levels, alphas):
    lower = beta.ppf((1 - alpha) / 2, k, N - k + 1)
    upper = beta.ppf(1 - (1 - alpha) / 2, k, N - k + 1)
    plt.fill_between(cdf_theoretical, lower, upper, color='gray', alpha=0.2, label=fr'${sigma}\sigma$ C.I.')

plt.xlabel('Credible Interval')
plt.ylabel('Fractional counts in C.I.')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
# plt.show()

save_fig_path = f'{outdir}/pp_plot_H0_{H0_prior_min}-{H0_prior_max}_{H0_sample_size}iterations_{file_name}.pdf'
plt.savefig(save_fig_path, dpi=200, bbox_inches='tight', pad_inches=0.05)

end_time = time.time()
elapsed = end_time - start_time
print(f"Total runtime: {str(datetime.timedelta(seconds=elapsed))}")