from run_fisher_analysis import fisher_analysis_GWfish
import numpy as np
import bilby
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import h5py
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

"""load gal catalog"""
gal_catalog_path = "./data/small_fiducial.parquet"
gal_small_fiducial_df = pd.read_parquet(gal_catalog_path)

z_true_all = gal_small_fiducial_df['redshift_true'].values
ra_true_all = gal_small_fiducial_df['ra_true'].values
dec_true_all = gal_small_fiducial_df['dec_true'].values
mag_r_all = gal_small_fiducial_df['mag_true_r_lsst_no_host_extinction'].values

"""mag cut"""
# mag_r_cutoff = 21
mag_r_cutoff = 22
mask_obs = mag_r_all <= mag_r_cutoff
z_true = z_true_all[mask_obs]
ra_true = ra_true_all[mask_obs]*np.pi/180 # convert to radians
dec_true = dec_true_all[mask_obs]*np.pi/180 # convert to radians

"""randomly select host galaxies for the simulated GW events"""
sample_size = 10000
random_seed_z = 1
np.random.seed(random_seed_z)
gal_host_id = np.random.randint(0, len(z_true), sample_size)

"""calculate the luminosity distance for each host galaxy and perform the Fisher analysis for each H0 value"""
random_seed_bilby = 2
bilby.core.utils.random.seed(random_seed_bilby)

mass_1_sample = 30 * np.ones(sample_size)  # in source frame
mass_2_sample = 30 * np.ones(sample_size)  # in source frame
theta_jn_sample = bilby.core.prior.Sine(name='theta_jn', boundary='reflective').sample(sample_size)
psi_sample = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi).sample(sample_size)
phase_sample = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi).sample(sample_size)
geocent_time = 1187008882.4  # time of GW170817

z_sample = z_true[gal_host_id]
ra_sample = ra_true[gal_host_id]
dec_sample = dec_true[gal_host_id]

def process_h0(H0):
    print(f"Starting H0 = {H0:.3f}")
    omega_m = 0.2647887323943662
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)
    dL_sample = cosmo.luminosity_distance(z_sample).value

    param_dict = {
        'mass_1': mass_1_sample * (1 + z_sample),
        'mass_2': mass_2_sample * (1 + z_sample),
        'luminosity_distance': dL_sample,
        'chi_1': np.zeros_like(mass_1_sample),
        'chi_2': np.zeros_like(mass_2_sample),
        'geocent_time': geocent_time * np.ones_like(dL_sample),
        'ra': ra_sample,
        'dec': dec_sample,
        'theta_jn': theta_jn_sample,
        'psi': psi_sample,
        'phase': phase_sample,
    }

    params_df, network_snr, sky_area_deg2_90_list, ra_1sigma_error_list, dec_1sigma_error_list, corr_ra_dec_list, dL_1sigma_error_list = fisher_analysis_GWfish(param_dict)

    output_dir = "./data/injections"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/injection_{sample_size}_magcut_{mag_r_cutoff}_H0_{H0:.3f}_for_selection_function.h5"
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('H0', data=H0)
        f.create_dataset('network_snr', data=network_snr, dtype='f4', compression='gzip')
        f.create_dataset('dL_1sigma_error', data=dL_1sigma_error_list, dtype='f4', compression='gzip')
        f.create_dataset('sky_area_deg2_90', data=sky_area_deg2_90_list, dtype='f4', compression='gzip')
        
    return f"SUCCESS: H0={H0:.3f}"

if __name__ == "__main__":
    # H0_array = np.linspace(20, 140, 1000)
    H0_array = np.arange(20, 201, 1)
    
    max_workers = 64 
    print(f"Starting parallel processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_h0, H0_array))

    for res in results:
        print(res)
