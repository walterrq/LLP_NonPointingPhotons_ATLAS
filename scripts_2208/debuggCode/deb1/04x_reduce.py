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


def reduceN1000(input_pickle, output_pickle):
    # Load the pickle file into a DataFrame
    df = pd.read_pickle(os.path.join(origin, input_pickle))
    
    # Filter the DataFrame to only include rows where N is between 0 and 999
    df_filtered = df.loc[df.index.get_level_values('N') < 1000]
    
    # Save the filtered DataFrame to a new pickle file
    df_filtered.to_pickle(os.path.join(origin, output_pickle))


# Reduce files of interest

reduceN1000('megatracks_4.pickle', "redtracks_4.pickle")
reduceN1000('megatracks_5.pickle', "redtracks_5.pickle")
reduceN1000('megatracks_6.pickle', "redtracks_6.pickle")

reduceN1000('megaefphoton_4.pickle', "redefphoton_4.pickle")
reduceN1000('megaefphoton_5.pickle', "redefphoton_5.pickle")
reduceN1000('megaefphoton_6.pickle', "redefphoton_6.pickle")

reduceN1000('megaecals_4.pickle', "redecals_4.pickle")
reduceN1000('megaecals_5.pickle', "redecals_5.pickle")
reduceN1000('megaecals_6.pickle', "redecals_6.pickle")


