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
from run_H0_estimation import create_cosmo_interpolator, create_V_dL_max_GW_interpolated, run_single_simulation
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest, beta

def calculate_p_val_from_likelihood(likelihood_array, h0_array, true_h0):
    cdf = cumulative_trapezoid(likelihood_array, h0_array, initial=0)
    if cdf[-1] == 0:
        return None
        
    cdf /= cdf[-1]
    f_cdf = interp1d(h0_array, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
    p_val = f_cdf(true_h0)
    return p_val

start_time = time.time()

"""set up H0 samples for simulation"""
H0_sample_size = 100
H0_prior_min = 20
H0_prior_max = 140
H0_samples = np.random.uniform(H0_prior_min, H0_prior_max, H0_sample_size)
# H0_samples = 71.0 * np.ones(H0_sample_size)
omega_m = 0.2647887323943662

"""load galaxy data"""
gal_catalog_path = "./data/small_fiducial.parquet"
gal_small_fiducial_df = pd.read_parquet(gal_catalog_path)
print('-------columns--------')
print(gal_small_fiducial_df.columns)
print('catalog num: ' + str(len(gal_small_fiducial_df)))

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

p_values = []
for i, H0 in enumerate(H0_samples):
    """cosmological parameters used in the simulation"""
    omega_m = omega_m
    omega_lambda = 1.0 - omega_m
    true_H0 = H0  # in unit of km/s/Mpc

    time_now = time.time()
    elapsed = time_now - start_time
    print(f"\n------{i}/{len(H0_samples)}: runtime={str(datetime.timedelta(seconds=elapsed))}------")
    print(f"Cosmological parameters used in the simulation:")
    print(f" H0: {true_H0} km/s/Mpc")
    print(f" Omega_m: {omega_m}")
    print(f" Omega_lambda: {omega_lambda}")

    """run single simulation"""
    mag_cut_r = 21
    gw_sample_size = 100
    number_of_events_to_pick = 10
    like = run_single_simulation(gal_df=gal_small_fiducial_df, H0_array=h0_array, H0_interpolators=h0_interpolators, true_H0=true_H0, omega_m=omega_m, mag_cut_r=mag_cut_r, gw_sample_size=gw_sample_size, number_of_events_to_pick=number_of_events_to_pick, V_dL_GW_max_interpolated=V_dL_GW_max_interpolated)

    """calculate p-value for true H0"""
    p_val = calculate_p_val_from_likelihood(like, h0_array, true_H0)
    p_values.append(p_val)

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

save_fig_path = f'./figs/pp_plot_H0_{H0_prior_min}-{H0_prior_max}_mag_cut_r_{mag_cut_r}_better_skyloc_{number_of_events_to_pick}events_{H0_sample_size}iterations.pdf' 
# save_fig_path = f'./figs/pp_plot_H0_{H0_samples[0]}_mag_cut_r_{mag_cut_r}_better_skyloc_{number_of_events_to_pick}events_{H0_sample_size}iterations.pdf'
# save_fig_path = f'./figs/pp_plot_H0_{H0_prior_min}-{H0_prior_max}_mag_cut_r_{mag_cut_r}_near_10deg2_{number_of_events_to_pick}events_{H0_sample_size}iterations.pdf'
# save_fig_path = f'./figs/pp_plot_H0_{H0_samples[0]}_mag_cut_r_{mag_cut_r}_near_10deg2_{number_of_events_to_pick}events_{H0_sample_size}iterations.pdf'
plt.savefig(save_fig_path, dpi=200, bbox_inches='tight', pad_inches=0.05)

end_time = time.time()
elapsed = end_time - start_time
print(f"Total runtime: {str(datetime.timedelta(seconds=elapsed))}")