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

# Function to load and print initial and final lines of a pickle file
def print_initial_and_final_lines(pickle_file):
    # Load the pickle file into a DataFrame
    df = pd.read_pickle(os.path.join(origin, pickle_file))
    
    # Print the first 5 rows
    print(f"Initial lines of {pickle_file}:")
    print(df.head())
    
    # Print the last 5 rows
    print(f"\nFinal lines of {pickle_file}:")
    print(df.tail())
    print("\n" + "="*80 + "\n")


# Function to print the content based on its type
def print_contents(label, data):
    print(f"\n{label}:")
    if isinstance(data, pd.DataFrame):
        print(data.head(10))  # Adjust the number of rows as needed
    elif isinstance(data, list):
        print(data[:100])  # Adjust the number of elements as needed
    elif isinstance(data, dict):
        for key, value in list(data.items())[:100]:  # Adjust the number of items as needed
            print(f"{key}: {value}")
    else:
        print(data)

def plot_most_energetic_histogram(df, particle_type, destiny):
    """
    Plots and saves a histogram of the most energetic particle (id = 0) for the given particle type.
    
    Parameters:
    df (DataFrame): The DataFrame containing the particle data.
    particle_type (str): The name of the particle type (e.g., 'electrons', 'muons', 'photons').
    destiny (str): The directory where the histogram image will be saved.
    """
    # Ensure the directory exists
    os.makedirs(destiny, exist_ok=True)

    # Filter to get the most energetic particle (id = 0) in each event
    most_energetic_particles = df.xs(0, level='id')  # Extract rows where id = 0

    # Create the histogram of 'pt' (transverse momentum)
    plt.figure(figsize=(10, 6))
    plt.hist(most_energetic_particles['pt'], bins=30, color='blue', edgecolor='black')
    plt.title(f'Histogram of Most Energetic {particle_type.capitalize()} Transverse Momentum (pt)')
    plt.xlabel('Transverse Momentum (pt)')
    plt.ylabel('Frequency')

    # Save the histogram as a PNG file
    plt.savefig(f"{destiny}most_energetic_{particle_type}_pt_histogram.png")

    # Optionally, display the plot
    plt.show()

def plot_met_histogram(df, destiny):
    """
    Plots and saves a histogram of the MET for the most energetic photon (id = 0) in each event.

    Parameters:
    df (DataFrame): The DataFrame containing the photon data.
    destiny (str): The directory where the histogram image will be saved.
    """
    # Ensure the directory exists
    os.makedirs(destiny, exist_ok=True)

    # Filter to get the most energetic photon (id = 0) in each event
    most_energetic_photons = df.xs(0, level='id')  # Extract rows where id = 0

    # Create the histogram of 'MET' (Missing Transverse Energy)
    plt.figure(figsize=(10, 6))
    plt.hist(most_energetic_photons['MET'], bins=30, color='green', edgecolor='black')
    plt.title('Histogram of MET for Most Energetic Photons')
    plt.xlabel('MET (Missing Transverse Energy)')
    plt.ylabel('Frequency')

    # Save the histogram as a PNG file
    plt.savefig(f"{destiny}most_energetic_photon_met_histogram.png")

    # Optionally, display the plot
    plt.show()

# Origin directory where the mega archives are stored
origin = "/Collider/scripts_2208/data/clean/"
destiny = f"./data/basics_graphs_merge/"
Path(destiny).mkdir(exist_ok=True, parents=True)

# List of mega pickle files
mega_files = ["megaphoton.pickle", "megaleptons.pickle", "megajets.pickle", "megatracks.pickle", "megaecals.pickle"]

# Loop through each mega file and print its initial and final lines
#for file in mega_files:
#    print_initial_and_final_lines(file)

input_file = origin + f"megaphoton.pickle"
photons = pd.read_pickle(input_file)
leptons = pd.read_pickle(input_file.replace('photon', 'leptons'))
jets = pd.read_pickle(input_file.replace('photon', 'jets'))
tracks = pd.read_pickle(input_file.replace('photon', 'tracks'))
ecals = pd.read_pickle(input_file.replace('photon', 'ecals'))


# Create sub DataFrame for electrons (id = 11)
electrons = leptons[leptons['pdg'] == 11].copy()

# Generate the 'new_id' column by resetting the id within each group
electrons['new_id'] = electrons.groupby(level=0).cumcount()

# Reset the index to turn the current 'id' level of the multi-index into a column
electrons = electrons.reset_index(level='id')

# Replace the 'id' in the index with 'new_id'
electrons = electrons.set_index('new_id', append=True)

# Rename the 'new_id' index level to 'id' to maintain the original naming
electrons.index = electrons.index.rename('id', level='new_id')

# Drop the 'id' column (not the multi-index)
electrons = electrons.drop(columns=['id'])



# Create sub DataFrame for muons (id = 13)
muons = leptons[leptons['pdg'] == 13].copy()
muons['new_id'] = muons.groupby(level=0).cumcount()  # Reset id within each group

# Reset the index to turn the current 'id' level of the multi-index into a column
muons = muons.reset_index(level='id')

# Replace the 'id' in the index with 'new_id'
muons = muons.set_index('new_id', append=True)

# Rename the 'new_id' index level to 'id' to maintain the original naming
muons.index = muons.index.rename('id', level='new_id')

# Drop the 'id' column (not the multi-index)
muons = muons.drop(columns=['id'])

#print_contents("Photons", photons)
#print_contents("Leptons", leptons)
#print_contents("Jets", jets)
#print_contents("Electrons", electrons)
#print_contents("Muons", muons)

# Plot for electrons
plot_most_energetic_histogram(electrons, 'electrons', destiny)

# Plot for muons
plot_most_energetic_histogram(muons, 'muons', destiny)

# Plot for photons
plot_most_energetic_histogram(photons, 'photons', destiny)

# Plot the MET histogram for the most energetic photons
plot_met_histogram(photons, destiny)