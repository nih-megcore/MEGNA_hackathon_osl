#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:55:26 2023

@author: jstout
"""

import os, os.path as op
import glob
import numpy as np
from osl_dynamics.models.hmm import Config
import mne
# import mne_bids
# from mne_bids import BIDSPath
from osl_dynamics.data import Data
import pickle
import nilearn
import nibabel as nb
from osl_dynamics.models.hmm import Model
from osl_dynamics.analysis import spectral
from osl_dynamics.analysis import power
from osl_dynamics.analysis import connectivity

def get_data(fmin, fmax, topdir, outdir_suffix=None):
    '''
    Helper function to load data

    Parameters
    ----------
    fmin : int
        Low freq.
    fmax : int
        High freq.

    Returns
    -------
    data : osl_dyanmics.data.Data
        data object from osl_dynamics
    data_dir : str path
        input data directory defined by freq
    out_dir : str path
        output directory for saving data

    '''
    data_dir = op.join(topdir, f'f_{fmin}_{fmax}_npy')
    print(data_dir)
    data = Data(data_dir,time_axis_first=False )
    out_dir = data_dir[:-3]+'oslresults'
    if outdir_suffix !=None:
        out_dir+='_'+outdir_suffix
    if not op.exists(out_dir): os.mkdir(out_dir)
    return data, data_dir, out_dir
    
#fmin=13;  fmax=35; topdir='/fast/Movie/results_f'    

# =============================================================================
# Commandline Options
# =============================================================================

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-fmin', type=int, help='Low frequency')
    parser.add_argument('-fmax', type=int, help='High frequency')
    parser.add_argument('-topdir', 
                        help='''Path to top level directory.  Frequency reults
                        will be one level down from this''')
    parser.add_argument('-outdir_suffix', help='''Suffix for output directory.  This 
                        will be appended to the end.  f_<<fmin>>_<<fmax>>_oslresults_<<suffix>>''',
                        default=None)
    parser.add_argument('-n_jobs', default=10, type=int)
    args=parser.parse_args()
    topdir=args.topdir
    fmin=args.fmin
    fmax=args.fmax
    n_jobs = args.n_jobs
    outdir_suffix = args.outdir_suffix



# =============================================================================
# Processing
# =============================================================================

data, data_dir, output_dir = get_data(fmin, fmax, topdir, outdir_suffix=outdir_suffix)    
num_pca = 10
#Prepare the envelope data
methods = {
    "pca": {"n_pca_components": num_pca},
    #"amplitude_envelope": {},
    "standardize": {},
    "moving_average":{"n_window":5}
}

data.prepare(methods)


# Create a config object
config = Config(
    n_states=4,
    n_channels=num_pca,
    sequence_length=200,
    learn_means=True,
    learn_covariances=True,
    batch_size=200,
    learning_rate=1e-3,
    n_epochs=15,  
)

# Initiate a Model class and print a summary
model = Model(config)

init_history = model.random_state_time_course_initialization(data, n_epochs=10, n_init=3)
history = model.fit(data, workers=n_jobs)

alpha = model.get_alpha(data)
np.save(op.join(output_dir, 'alpha.npy'), alpha)

raw_data = model.get_training_time_series(data, prepared = False)

# Calculate multitaper spectra for each state and subject 
f, psd, coh = spectral.multitaper_spectra(
    data=raw_data,
    alpha=alpha,
    sampling_frequency=150,
    time_half_bandwidth=4,
    n_tapers=7,
    n_jobs=n_jobs
)

np.save(op.join(output_dir, 'f.npy'), f)
np.save(op.join(output_dir, 'psd.npy'), psd)
np.save(op.join(output_dir, 'coh.npy'), coh)


# Integrate the power spectra over the alpha band (8-12 Hz)
p = power.variance_from_spectra(f, psd)#, frequency_range=[fmin, fmax])
np.save(op.join(output_dir, 'p.npy'), p)


c = connectivity.mean_coherence_from_spectra(f, coh)
np.save(op.join(output_dir, 'c.npy'), c)
