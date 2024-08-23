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

def calculate_delta_r(df_photons, df_leptons):
    """
    Calculates the minimum ΔR between each photon and all electrons in each event,
    and stores the minimum ΔR for each photon.

    Parameters:
    df_photons (DataFrame): The DataFrame containing photon data.
    df_leptons (DataFrame): The DataFrame containing lepton data.

    Returns:
    min_delta_r_values (numpy array): An array of minimum ΔR values for each photon in each event.
    """
    min_delta_r_values = np.array([])

    # Get unique event indices
    events = df_photons.index.get_level_values('N').unique()

    for event in events:

        # Check if the event has both photons and electrons
        if event in df_photons.index.get_level_values('N') and event in df_leptons.index.get_level_values('N'):
            # Extract photons and electrons in the event
            photons = df_photons.loc[event]
            electrons = df_leptons.loc[event]
            
            # Extract phi and eta values as numpy arrays
            photon_phi = photons['phi'].values
            photon_eta = photons['eta'].values
            lepton_phi = electrons['phi'].values
            lepton_eta = electrons['eta'].values
            
            # Calculate Δphi and Δη using numpy broadcasting (outer subtraction)
            delta_phi = np.subtract.outer(photon_phi, lepton_phi)
            delta_eta = np.subtract.outer(photon_eta, lepton_eta)
        
            
            # Calculate ΔR for all photon-electron pairs
            delta_r = np.sqrt(delta_phi**2 + delta_eta**2)

            # Ignore ΔR values that are too small
            #mandamos a infinito los que tienen cero para que asi nunca puedan ser seleccionados
            delta_r = np.where(delta_r > 1e-15, delta_r, np.inf)
            
            # Find the minimum ΔR for each photon
            min_delta_r_per_photon = np.min(delta_r, axis=1)
            
            # Append the minimum ΔR for each photon in this event to the result array
            min_delta_r_values = np.append(min_delta_r_values, min_delta_r_per_photon)

    return min_delta_r_values

def plot_delta_r_histogram(delta_r_values, alpha, destiny):
    """
    Plots and saves a histogram of ΔR values.

    Parameters:
    delta_r_values (list): A list of ΔR values to plot.
    destiny (str): The directory where the histogram image will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(delta_r_values, bins=30, color='blue', edgecolor='black')
    plt.title(f'Histogram of ΔR between Most Energetic Photon and Electron {alpha.capitalize()}')
    plt.xlabel('ΔR')
    plt.ylabel('Frequency')
    
    # Save the histogram as a PNG file
    plt.savefig(f"{destiny}/delta_r_histogram_{alpha}.png")
    
    # Optionally, display the plot
    plt.show()

def reset_id_by_pt(electrons):
    """
    Sorts the DataFrame by 'pt' within each 'N', then assigns a new 'id' starting from 0 for each group.

    Parameters:
    electrons (DataFrame): The input DataFrame with a multi-index ('N', 'id').

    Returns:
    electrons (DataFrame): The DataFrame with a new multi-index ('N', 'id') sorted by 'pt'.
    """
    # Reset index to treat 'N' and 'id' as columns
    electrons = electrons.reset_index()

    electrons = electrons.drop(columns=['id'])

    # Sort the DataFrame by 'N' and 'pt'
    electrons = electrons.sort_values(by=['N', 'pt'], ascending=[True, False])

    g = electrons.groupby('N', as_index=False).cumcount()

    electrons['id'] = g

    electrons = electrons.set_index(['N', 'id'])

    return electrons

origin = "/Collider/scripts_2208/data/clean/"
destiny = f"./data/deltaR_tracks/"
Path(destiny).mkdir(exist_ok=True, parents=True)


for alpha in [4, 5, 6]:

    print("Alpha: ", alpha)

    input_file = origin + f"megaefphoton_{alpha}.pickle"

    efphotons = pd.read_pickle(input_file)
    eftrack = pd.read_pickle(input_file.replace('efphoton', 'tracks'))
    ecals = pd.read_pickle(input_file.replace('efphoton', 'ecals'))

    # Create sub DataFrame for electrons (id = 11)

    alpha_s = str(alpha)

    # Calculate ΔR values
    deltaR_ph_track = calculate_delta_r(efphotons, eftrack)
    deltaR_ph_ecals = calculate_delta_r(efphotons, ecals)

    # Plot ΔR histogram
    plot_delta_r_histogram(deltaR_ph_track, alpha_s, destiny)
    plot_delta_r_histogram(deltaR_ph_ecals, alpha_s, destiny)
