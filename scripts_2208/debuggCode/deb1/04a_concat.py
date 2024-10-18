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

photon_4 = [f for f in pickle_files if 'Alpha4_13_photons' in f]
photon_5 = [f for f in pickle_files if 'Alpha5_13_photons' in f]
photon_6 = [f for f in pickle_files if 'Alpha6_13_photons' in f]

lepton_4 = [f for f in pickle_files if 'Alpha4_13_leptons' in f]
lepton_5 = [f for f in pickle_files if 'Alpha5_13_leptons' in f]
lepton_6 = [f for f in pickle_files if 'Alpha6_13_leptons' in f]

jet_4 = [f for f in pickle_files if 'Alpha4_13_jets' in f]
jet_5 = [f for f in pickle_files if 'Alpha5_13_jets' in f]
jet_6 = [f for f in pickle_files if 'Alpha6_13_jets' in f]

efphoton_4 = [f for f in pickle_files if 'Alpha4_13_efphotons' in f]
efphoton_5 = [f for f in pickle_files if 'Alpha5_13_efphotons' in f]
efphoton_6 = [f for f in pickle_files if 'Alpha6_13_efphotons' in f]

eftrack_4 = [f for f in pickle_files if 'Alpha4_13_eftracks' in f]
eftrack_5 = [f for f in pickle_files if 'Alpha5_13_eftracks' in f]
eftrack_6 = [f for f in pickle_files if 'Alpha6_13_eftracks' in f]

ecal_4 = [f for f in pickle_files if 'Alpha4_13_ecals' in f]
ecal_5 = [f for f in pickle_files if 'Alpha5_13_ecals' in f]
ecal_6 = [f for f in pickle_files if 'Alpha6_13_ecals' in f]


def combine_pickle(file_list, output_filename):
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Loop through each file in the list and process them
    for file in file_list:
        # Read the pickle file into a DataFrame
        df = pd.read_pickle(os.path.join(origin, file))
        
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(dataframes)
    
    # Reset the 'id' values to be sequential within each 'N'
    combined_df = combined_df.reset_index()
    
    # Sort the DataFrame by 'N' first and then by the original 'id'
    combined_df = combined_df.sort_values(by=['N', 'id'])
    
    # Create new sequential 'id' values within each 'N' group
    combined_df['id'] = combined_df.groupby('N').cumcount()
    
    # Set the index back to ['N', 'id']
    combined_df = combined_df.set_index(['N', 'id'])
    
    # Sort the resulting DataFrame by 'N' and 'id'
    combined_df = combined_df.sort_index()
    
    # Save the combined DataFrame to a new pickle file
    combined_df.to_pickle(os.path.join(origin, output_filename))


# Merge and save the pickle files for each type
combine_pickle(photon_4, "megaphoton_4.pickle")
combine_pickle(photon_5, "megaphoton_5.pickle")
combine_pickle(photon_6, "megaphoton_6.pickle")

combine_pickle(lepton_4, "megaleptons_4.pickle")
combine_pickle(lepton_5, "megaleptons_5.pickle")
combine_pickle(lepton_6, "megaleptons_6.pickle")

combine_pickle(jet_4, "megajets_4.pickle")
combine_pickle(jet_5, "megajets_5.pickle")
combine_pickle(jet_6, "megajets_6.pickle")

combine_pickle(eftrack_4, "megatracks_4.pickle")
combine_pickle(eftrack_5, "megatracks_5.pickle")
combine_pickle(eftrack_6, "megatracks_6.pickle")

combine_pickle(efphoton_4, "megaefphoton_4.pickle")
combine_pickle(efphoton_5, "megaefphoton_5.pickle")
combine_pickle(efphoton_6, "megaefphoton_6.pickle")

combine_pickle(ecal_4, "megaecals_4.pickle")
combine_pickle(ecal_5, "megaecals_5.pickle")
combine_pickle(ecal_6, "megaecals_6.pickle")

