import sys
import numpy as np
import re
import glob
import pandas as pd
from scipy.interpolate import interp1d
from my_funcs import isolation
from pathlib import Path
import json
import sys
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os  

#This code gives a hardcoded version to merge the by alphas in our search

# Origin directory where the files are stored
origin = "/Collider/scripts_2208/data/clean/"

# Get a list of all files in the directory
all_files = os.listdir(origin)

# Filter only the pickle files
pickle_files = [f for f in all_files if f.endswith('.pickle')]


# Separate the pickle files by type

photon_files_4 = [f for f in pickle_files if 'Alpha4_13_photons' in f]
photon_files_5 = [f for f in pickle_files if 'Alpha5_13_photons' in f]
photon_files_6 = [f for f in pickle_files if 'Alpha6_13_photons' in f]

lepton_files_4 = [f for f in pickle_files if 'Alpha4_13_leptons' in f]
lepton_files_5 = [f for f in pickle_files if 'Alpha5_13_leptons' in f]
lepton_files_6 = [f for f in pickle_files if 'Alpha6_13_leptons' in f]

jet_files_4 = [f for f in pickle_files if 'Alpha4_13_jets' in f]
jet_files_5 = [f for f in pickle_files if 'Alpha5_13_jets' in f]
jet_files_6 = [f for f in pickle_files if 'Alpha6_13_jets' in f]

efphoton_files_4 = [f for f in pickle_files if 'Alpha4_13_efphotons' in f]
efphoton_files_5 = [f for f in pickle_files if 'Alpha5_13_efphotons' in f]
efphoton_files_6 = [f for f in pickle_files if 'Alpha6_13_efphotons' in f]

eftrack_files_4 = [f for f in pickle_files if 'Alpha4_13_eftracks' in f]
eftrack_files_5 = [f for f in pickle_files if 'Alpha5_13_eftracks' in f]
eftrack_files_6 = [f for f in pickle_files if 'Alpha6_13_eftracks' in f]

ecal_files_4 = [f for f in pickle_files if 'Alpha4_13_ecals' in f]
ecal_files_5 = [f for f in pickle_files if 'Alpha5_13_ecals' in f]
ecal_files_6 = [f for f in pickle_files if 'Alpha6_13_ecals' in f]


def merge_and_save(files, output_name):
    """Merges the DataFrames from the given pickle files and saves the result."""
    dataframes = []
    for file in files:
        df = pd.read_pickle(os.path.join(origin, file))
        dataframes.append(df)
    merged_df = pd.concat(dataframes)
    merged_df.to_pickle(os.path.join(origin, output_name))
    print(f"{output_name} saved successfully.")

# Merge and save the pickle files for each type
merge_and_save(photon_files_4, "megaphoton_4.pickle")
merge_and_save(photon_files_5, "megaphoton_5.pickle")
merge_and_save(photon_files_6, "megaphoton_6.pickle")

merge_and_save(lepton_files_4, "megaleptons_4.pickle")
merge_and_save(lepton_files_5, "megaleptons_5.pickle")
merge_and_save(lepton_files_6, "megaleptons_6.pickle")

merge_and_save(jet_files_4, "megajets_4.pickle")
merge_and_save(jet_files_5, "megajets_5.pickle")
merge_and_save(jet_files_6, "megajets_6.pickle")

merge_and_save(eftrack_files_4, "megatracks_4.pickle")
merge_and_save(eftrack_files_5, "megatracks_5.pickle")
merge_and_save(eftrack_files_6, "megatracks_6.pickle")

merge_and_save(efphoton_files_4, "megaefphoton_4.pickle")
merge_and_save(efphoton_files_5, "megaefphoton_5.pickle")
merge_and_save(efphoton_files_6, "megaefphoton_6.pickle")

merge_and_save(ecal_files_4, "megaecals_4.pickle")
merge_and_save(ecal_files_5, "megaecals_5.pickle")
merge_and_save(ecal_files_6, "megaecals_6.pickle")

