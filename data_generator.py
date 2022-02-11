#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import random
import glob

# A CLI progress bar
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange

import numpy as np

import decimate_and_crop

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# GPS constants: chip rate and number of chips per PRN code period
Fc = 1.023e6; Tc = 1/Fc; Nc = 1023; T = Nc/Fc

# Coherent integration time
Ti = 20*T

# Power of the signal at the output of the correlator
C = 8/Ti**2

# Observation range for tau_prime, around the estimated propagation delay
min_tau_prime = -3*Tc/2
max_tau_prime = 3*Tc

# Observation range for f_prime, around the estimated Doppler shift
max_f_prime = min(5/Ti, 800 + 2.5/Ti)
min_f_prime = -max_f_prime
# Number of points in [min_f_prime, max_f_prime]: N_f
N_f = 80
# The solution is constrained to obtain a point in f_prime = 0
N_f = N_f + (1 if (N_f%2 == 0) else 0)
i_f_prime_0 = (N_f - 1)/2 + 1
f_prime = np.linspace(min_f_prime,max_f_prime,N_f)

# Sampling frequency and period
Fs = 20e6; Ts = 1/Fs

# Number of points to keep before and after the Prompt point
n_min_tau_prime = round(min_tau_prime/Ts)
n_max_tau_prime = round(max_tau_prime/Ts)
# tau_prime vector in units of µs
tau_prime = np.arange(n_min_tau_prime,n_max_tau_prime + 1)/Fs/1e-6
N_t = len(tau_prime)
# Index value for tau_prime = 0
i_tau_prime_0 = -n_min_tau_prime

def _load_noise(noise_dataset_path, nb_samples=None):
  paths_i = glob.glob(noise_dataset_path + '/*I*.csv')
  # glob does not guarantee any specific order in the returned list.
  # So the list is sorted for the following shuffle to always have the same
  # initial state.
  paths_i.sort()
  random.shuffle(paths_i)

  if nb_samples != None:
    if nb_samples > len(paths_i):
      logger.warning("Less samples ({:d}) than requested ({:d})"\
        .format(len(paths_i),nb_samples))
    else:
      # First samples after shuffling
      paths_i = paths_i[:nb_samples]

  # I and Q channels must be paired.
  paths_q = [path_i.replace("_I","_Q") for path_i in paths_i]

  list_I = []; list_Q = []
  # Initialisation of the CLI progress bar
  with logging_redirect_tqdm():
    for i in trange(len(paths_i),desc='Loading noise files'):
      noise_I = np.loadtxt(paths_i[i], delimiter=',')
      noise_Q = np.loadtxt(paths_q[i], delimiter=',')
      list_I.append(noise_I)
      list_Q.append(noise_Q)
    
  return list_I, list_Q

def _create_path(alpha, delta_tau, delta_f, delta_theta):
  delta_f_prime = f_prime - delta_f
  # Generation of the sinc function in frequency, multiplied by cos for the I
  # channel and sin for the Q channel.
  # pi is already included in the NumPy sinc function.
  sinc_vector = alpha*np.sinc(delta_f_prime*Ti)
  I_sinc_vector = sinc_vector*np.cos(np.pi*delta_f_prime*Ti - delta_theta)
  Q_sinc_vector = sinc_vector*np.sin(np.pi*delta_f_prime*Ti - delta_theta)
  I_sinc_matrix = np.tile(I_sinc_vector,(N_t,1)).transpose()
  Q_sinc_matrix = np.tile(Q_sinc_vector,(N_t,1)).transpose()
  # Generation of the triangle function in delay
  triangle_vector = np.zeros(N_t)
  for i in range(N_t):
    x = i - i_tau_prime_0; y = 1 - (abs(x*Ts - delta_tau))/Tc
    triangle_vector[i] = max(0,y)
  triangle_matrix = np.tile(triangle_vector,(N_f,1))

  # Return the matrices for the I and Q channels
  return I_sinc_matrix*triangle_matrix, Q_sinc_matrix*triangle_matrix

def data_generator(direct_param, with_mp, random_state, nb_samples=None,
  mp_param=None, image_size=(80,80)):

  rng = np.random.default_rng(random_state)

  (i_d, j_d, i_Doppler, j_delay) = decimate_and_crop.get_indices(image_size)

  # LOS
  matrix_I, matrix_Q = _create_path(1,0,0,0)

  noise_dataset_path = direct_param['noise_dataset_path']
  list_I_n, list_Q_n = _load_noise(noise_dataset_path, nb_samples)
  l_I_n = len(list_I_n)
  nb_s = (nb_samples if (nb_samples != None) else l_I_n)

  C_N0_dB_lh = direct_param['C_N0_dB_range']
  C_N0_dB = rng.uniform(low=C_N0_dB_lh[0], high=C_N0_dB_lh[1], size=nb_s)

  # Navigation bit: 0 -> +1, 1 -> -1
  nav_bit = (rng.integers(0,2,size=nb_s) - 0.5)*2

  if (with_mp == True):
    alpha_lh = mp_param['mp_alpha_range']
    mp_tau_lh = mp_param['mp_tau_range']
    mp_f_sigma = mp_param['mp_f_sigma']
    mp_theta_lh = mp_param['mp_theta_range'] 
    alpha = rng.uniform(low=alpha_lh[0], high=alpha_lh[1], size=nb_s)
    mp_tau = rng.uniform(low=mp_tau_lh[0], high=mp_tau_lh[1], size=nb_s)*Tc
    mp_f = mp_f_sigma*rng.standard_normal(size=nb_s)
    mp_theta = rng.uniform(low=mp_theta_lh[0], high=mp_theta_lh[1], size=nb_s)

  # Add multipath, if requested, and noise
  data_samples = []
  # Initialisation of the CLI progress bar
  with logging_redirect_tqdm():
    for i in trange(nb_s,desc='Creating correlator images'):
      corr_matrix_I = nav_bit[i]*matrix_I
      corr_matrix_Q = nav_bit[i]*matrix_Q

      if (with_mp == True):
        mp_matrix_I, mp_matrix_Q = _create_path(alpha[i],mp_tau[i],mp_f[i],
          mp_theta[i])
        corr_matrix_I += nav_bit[i]*mp_matrix_I
        corr_matrix_Q += nav_bit[i]*mp_matrix_Q
        label_classification = 1
      else:
        label_classification = 0

      C_N0 = 10**(C_N0_dB[i]/10)
      # Power of the noise on each I and Q channel
      N0 = C/C_N0
      Pbruit = N0*Ti/16

      # Images for the I and Q channels: signal + noise
      # If l_I_n < nb_samples some noise images have to be reused
      j = i%l_I_n
      I_n = list_I_n[j]
      Q_n = list_Q_n[j]
      P_I_n = np.var(I_n)
      P_Q_n = np.var(Q_n)
      corr_matrix_I += np.sqrt(Pbruit/P_I_n)*I_n
      corr_matrix_Q += np.sqrt(Pbruit/P_Q_n)*Q_n

      # Apply decimation and cropping mask
      corr_matrix_I_resize = corr_matrix_I[i_d,:][:,j_d]
      corr_matrix_Q_resize = corr_matrix_Q[i_d,:][:,j_d]

      if (with_mp == False):
        np.savetxt(image_no_mp_file_format.format(i+1,C_N0_dB[i],'I'),
          corr_matrix_I_resize,delimiter=',')
        np.savetxt(image_no_mp_file_format.format(i+1,C_N0_dB[i],'Q'),
          corr_matrix_Q_resize,delimiter=',')
      else:
        np.savetxt(image_mp_file_format.format(i+1,C_N0_dB[i],'I'),
          corr_matrix_I_resize,delimiter=',')
        np.savetxt(image_mp_file_format.format(i+1,C_N0_dB[i],'Q'),
          corr_matrix_Q_resize,delimiter=',')

      # An empty axis is added in each image
      corr_matrix_I_resize = corr_matrix_I_resize[...,np.newaxis]
      corr_matrix_Q_resize = corr_matrix_Q_resize[...,np.newaxis]
      # The two I and Q images are concatenated along this new axis
      corr_matrix = np.concatenate((corr_matrix_I_resize,
        corr_matrix_Q_resize), axis=2)
          
      # This pair of images (I,Q) is added to the list data_samples
      data_samples.append({'img': corr_matrix, 'label':label_classification})

    return data_samples

if __name__ == "__main__":

  image_size = (80, 80)
  nb_images = 1000

  # Put the images in this directory
  dataset_path = './Dataset'
  
  # File name format
  image_no_mp_file_format = dataset_path + "/no_mp" 
  image_no_mp_file_format += '/snap_DS_{:d}x{:d}'.\
    format(image_size[0], image_size[1])
  image_no_mp_file_format += "_{:d}_{:2.1f}_dBHz_{:s}.csv" 

  image_mp_file_format = dataset_path + "/mp" 
  image_mp_file_format += '/snap_MP_DS_{:d}x{:d}'.\
    format(image_size[0], image_size[1])
  image_mp_file_format += "_{:d}_{:2.1f}_dBHz_{:s}.csv" 
  
  label_file_format = dataset_path + "/mp" 
  label_file_format += '/label_MP_DS_{:d}x{:d}.csv'.\
    format(image_size[0], image_size[1])

  noise_dataset_path = './Noise_Samples/SX3_noise_n1_5_sat_10'

  C_N0_dB_range = (45, 45)
  direct_param = {'C_N0_dB_range': C_N0_dB_range,
    'noise_dataset_path': noise_dataset_path}

  mp_alpha_range = (0.1, 0.9)
  mp_tau_range = (0, 3/2)
  mp_f_sigma = min(800,2.5/Ti)
  mp_theta_range = (0, 2*np.pi)
  mp_param = {'mp_alpha_range': mp_alpha_range, 'mp_tau_range': mp_tau_range,
    'mp_f_sigma': mp_f_sigma, 'mp_theta_range': mp_theta_range}

  data_generator(direct_param, False, None, nb_samples=nb_images,
                 image_size=image_size)
  data_generator(direct_param, True, None, nb_samples=nb_images,
                 mp_param=mp_param, image_size=image_size)

