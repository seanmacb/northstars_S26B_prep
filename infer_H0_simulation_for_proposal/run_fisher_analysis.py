from __future__ import division, print_function
import pandas as pd
import numpy as np
from astropy import constants as const

from GWFish.modules.detection import Network
from GWFish.modules.fishermatrix import compute_network_errors
import GWFish.modules.waveforms as wf

import bilby
from gwpy.timeseries import TimeSeries
from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_symmetric_mass_ratio, luminosity_distance_to_redshift, chirp_mass_and_mass_ratio_to_component_masses
from bilby.core import utils
from gwosc import datasets
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import logging
from GWFish.modules.fishermatrix import sky_localization_percentile_factor

###fisher analysis using GWFish###
def fisher_analysis_GWfish(param_dict):
   parameters = pd.DataFrame.from_dict({k:v*np.array([1.]) for k, v in param_dict.items()})
   fisher_parameters = list(param_dict.keys())
   fisher_parameters.remove('chi_1') # to avoid singular matrix error for symmetric mass and spin systems
   fisher_parameters.remove('chi_2') # to avoid singular matrix error for symmetric mass and spin systems
   waveform_model = 'IMRPhenomD'
   polarization_list = None
   save_file_name = None
   save_file_path = None

   detectors_GWFish = ['LHO', 'LLO', 'VIR']
   network = Network(detectors_GWFish)
   network.detection_SNR = [0.0, 0.0] # for single and network SNR

   ###save file name and path###
   if save_file_name is not None:
       iFIM_file_name = save_file_name
   else:
      iFIM_file_name = ''.join([detector for detector in detectors_GWFish])
      # iFIM_file_name = 'HLV'
      # iFIM_file_name = 'D'
   if save_file_path is not None:
      iFIM_file_path = save_file_path
   else:
      iFIM_file_path = '../data/'
   # print(f'save file path : {iFIM_file_path}inv_fisher_matrices_{iFIM_file_name}')
   ###############################################

   detected, snr, errors, sky_localization, iFIM, FIM = compute_network_errors(
       network = network,
       parameter_values = parameters,
       fisher_parameters = fisher_parameters,
       waveform_model = waveform_model,
       waveform_class = wf.non_tensor_waveform,
       polarization_list = polarization_list,
       save_matrices = True,
       save_matrices_path = iFIM_file_path,
       matrix_naming_postfix = iFIM_file_name,
       eps = 1e-5,
       eps_mass = 1e-8
       )

   ra_1sigma_error_list = errors[:, fisher_parameters.index('ra')]
   dec_1sigma_error_list = errors[:, fisher_parameters.index('dec')]
   corr_ra_dec_list = iFIM[:, fisher_parameters.index('ra'), fisher_parameters.index('dec')] / (ra_1sigma_error_list * dec_1sigma_error_list)
   sky_area_deg2_90_list = sky_localization * sky_localization_percentile_factor() #convert to 90% C.I. area in deg^2
   dL_1sigma_error_list = errors[:, fisher_parameters.index('luminosity_distance')]

   return parameters, snr, sky_area_deg2_90_list, ra_1sigma_error_list, dec_1sigma_error_list, corr_ra_dec_list, dL_1sigma_error_list