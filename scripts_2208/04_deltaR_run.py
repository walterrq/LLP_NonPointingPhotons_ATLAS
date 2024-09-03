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

import numpy as nps
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def isolate_photons(df_photons, df_leptons, delta_r_max=0.2, pt_min=0.1, pt_ratio_max=0.065):

    """
    Isolates photons using the Mini-cone algorithm and returns a DataFrame of isolated photons.

    Parameters:
    df_photons (DataFrame): The DataFrame containing photon data.
    df_leptons (DataFrame): The DataFrame containing lepton (e.g., electron) data.
    delta_r_max (float): The maximum ΔR to consider for isolation (cone size).
    pt_min (float): The minimum pT threshold for leptons to be considered in the isolation.
    pt_ratio_max (float): The maximum allowable ratio of the sum of lepton pT to photon pT.

    Returns:
    isolated_photons (DataFrame): A DataFrame containing only the isolated photons.
    """
    df_isolated_photons = pd.DataFrame(columns=['N', 'id', 'E', 'pt', 'eta', 'phi', 'z_origin', 'rel_tof', 'MET'])

    # Get unique event indices
    events = df_photons.index.get_level_values('N').unique()

    for event in events:
        # Check if the event has both photons and leptons
        if event in df_photons.index.get_level_values('N') and event in df_leptons.index.get_level_values('N'):
            # Extract photons and leptons in the event
            photons = df_photons.loc[event]
            leptons = df_leptons.loc[event]
            

            # Extract phi, eta, and pt values as numpy arrays
            photon_phi = photons['phi'].values
            photon_eta = photons['eta'].values
            photon_pt = photons['pt'].values
            lepton_phi = leptons['phi'].values
            lepton_eta = leptons['eta'].values
            lepton_pt = leptons['pt'].values

            # Calculate Δphi and Δη using numpy broadcasting (outer subtraction)
            delta_phi = np.subtract.outer(photon_phi, lepton_phi)
            delta_eta = np.subtract.outer(photon_eta, lepton_eta)

            # Calculate ΔR for all photon-lepton pairs
            delta_r = np.sqrt(delta_phi**2 + delta_eta**2)

            # Determine if there are no leptons within the ΔR max condition
            no_leptons_within_cone = np.all(delta_r > delta_r_max, axis=1)

            # Apply the ΔR max condition
            within_cone = (delta_r < delta_r_max)

            # Apply the pT min condition to the leptons
            lepton_pt_filtered = np.where(lepton_pt > pt_min, lepton_pt, 0)

            # Calculate the sum of pT of leptons within the cone for each photon
            sum_pt_within_cone = np.sum(lepton_pt_filtered * within_cone, axis=1)

            # Calculate the isolation ratio for each photon
            isolation_ratio = sum_pt_within_cone / photon_pt

                                    # Use filter to find values less than 0.065
            values_below_threshold = list(filter(lambda x: x < 0.065 and x > 0, isolation_ratio))

            if values_below_threshold:
                print("Values less than 0.065:", values_below_threshold)

            # Determine if each photon is isolated based on the isolation ratio or if there are no leptons nearby
            isolated_photon_mask = (isolation_ratio < pt_ratio_max) | no_leptons_within_cone

            # Print statements for all variables
            not_isolated_photon_mask = ~isolated_photon_mask
            
            """
            print("photonsdataframe")
            print(photons)
            print("electronsdataframe")
            print(leptons)
            print("delta_phi:")
            print(delta_phi)
            print("delta_eta:")
            print(delta_eta)
            print("delta_r:")
            print(delta_r)
            print("no_leptons_within_cone:")
            print(no_leptons_within_cone)
            print("within_cone:")
            print(within_cone)
            print("lepton_pt_filtered:")
            print(lepton_pt_filtered)
            print("sum_pt_within_cone:")
            print(sum_pt_within_cone)
            print("isolation_ratio:")
            print(isolation_ratio)
            print("isolated_photon_mask:")
            print(isolated_photon_mask)
            """
            

            # Filter and store the isolated photons with the event number (N) and photon id
            if any(not_isolated_photon_mask):
                # Filter isolated photons
                not_isolated_photons = photons[not_isolated_photon_mask].copy()
                # Add the event number (N) as a column
                #print("N: ", event)
                #isolated_photons['N'] = event
                # Add the photon id as a column
                #isolated_photons['id'] = isolated_photons.index
                index_list = not_isolated_photons.index.tolist()
                #print("id: ", index_list)

                for index_event in index_list:
                    df_photons = df_photons.drop((event, index_event))
                # Elimina los que no estan aislados
                #df_photons = df_photons
                #df_isolated_photons = pd.concat([df_isolated_photons, isolated_photons[['N', 'id', 'E', 'pt', 'eta', 'phi', 'z_origin', 'rel_tof', 'MET']]])
                #print("isolated_photon:")
                #print(isolated_photons)

    return df_photons

def isolate_photons_no_opt(df_photons, df_leptons, delta_r_max=0.2, pt_min=0.1, pt_ratio_max=0.065):
    """
    Isolates photons using the Mini-cone algorithm and returns a DataFrame of isolated photons.

    Parameters:
    df_photons (DataFrame): The DataFrame containing photon data.
    df_leptons (DataFrame): The DataFrame containing lepton (e.g., electron) data.
    delta_r_max (float): The maximum ΔR to consider for isolation (cone size).
    pt_min (float): The minimum pT threshold for leptons to be considered in the isolation.
    pt_ratio_max (float): The maximum allowable ratio of the sum of lepton pT to photon pT.

    Returns:
    isolated_photons (DataFrame): A DataFrame containing only the isolated photons.
    """
    df_isolated_photons = pd.DataFrame(columns=['N', 'id', 'E', 'pt', 'eta', 'phi', 'z_origin', 'rel_tof', 'MET'])

    # Get unique event indices
    events = df_photons.index.get_level_values('N').unique()

    for event in events:
        # Check if the event has both photons and leptons
        if event in df_photons.index.get_level_values('N') and event in df_leptons.index.get_level_values('N'):
            # Extract photons and leptons in the event
            photons = df_photons.loc[event]
            leptons = df_leptons.loc[event]

            
            print("photonsdataframe")
            print(photons)
            print("electronsdataframe")
            print(leptons)
            

            # Extract phi, eta, and pt values as numpy arrays
            photon_phi = photons['phi'].values
            photon_eta = photons['eta'].values
            photon_pt = photons['pt'].values
            lepton_phi = leptons['phi'].values
            lepton_eta = leptons['eta'].values
            lepton_pt = leptons['pt'].values

            # Calculate Δphi and Δη using numpy broadcasting (outer subtraction)
            delta_phi = np.subtract.outer(photon_phi, lepton_phi)
            delta_eta = np.subtract.outer(photon_eta, lepton_eta)

            # Calculate ΔR for all photon-lepton pairs
            delta_r = np.sqrt(delta_phi**2 + delta_eta**2)

            # Determine if there are no leptons within the ΔR max condition
            no_leptons_within_cone = np.all(delta_r > delta_r_max, axis=1)

            # Apply the ΔR max condition
            within_cone = (delta_r < delta_r_max)

            # Apply the pT min condition to the leptons
            lepton_pt_filtered = np.where(lepton_pt > pt_min, lepton_pt, 0)

            # Calculate the sum of pT of leptons within the cone for each photon
            sum_pt_within_cone = np.sum(lepton_pt_filtered * within_cone, axis=1)

            # Calculate the isolation ratio for each photon
            isolation_ratio = sum_pt_within_cone / photon_pt

            # Use filter to find values less than 0.065
            values_below_threshold = list(filter(lambda x: x < 0.065 and x > 0, isolation_ratio))

            if values_below_threshold:
                print("Values less than 0.065:", values_below_threshold)

            # Determine if each photon is isolated based on the isolation ratio or if there are no leptons nearby
            isolated_photon_mask = (isolation_ratio < pt_ratio_max) | no_leptons_within_cone

            # Print statements for all variables
            
            
            print("delta_phi:")
            print(delta_phi)
            print("delta_eta:")
            print(delta_eta)
            print("delta_r:")
            print(delta_r)
            print("no_leptons_within_cone:")
            print(no_leptons_within_cone)
            print("within_cone:")
            print(within_cone)
            print("lepton_pt_filtered:")
            print(lepton_pt_filtered)
            print("sum_pt_within_cone:")
            print(sum_pt_within_cone)
            print("isolation_ratio:")
            print(isolation_ratio)
            print("isolated_photon_mask:")
            print(isolated_photon_mask)
            
            

            # Filter and store the isolated photons with the event number (N) and photon id
            if any(isolated_photon_mask):
                # Filter isolated photons
                isolated_photons = photons[isolated_photon_mask].copy()
                # Add the event number (N) as a column
                isolated_photons['N'] = event
                # Add the photon id as a column
                isolated_photons['id'] = isolated_photons.index
                # Append to the result DataFrame
                df_isolated_photons = pd.concat([df_isolated_photons, isolated_photons[['N', 'id', 'E', 'pt', 'eta', 'phi', 'z_origin', 'rel_tof', 'MET']]])
                #print("isolated_photon:")
                #print(isolated_photons)

        else:
            # If no leptons are present, consider all photons in this event as isolated
            isolated_photons = df_photons.loc[event].copy()
            # Add the event number (N) as a column
            isolated_photons['N'] = event
            # Add the photon id as a column
            isolated_photons['id'] = isolated_photons.index
            # Append to the result DataFrame
            df_isolated_photons = pd.concat([df_isolated_photons, isolated_photons[['N', 'id', 'E', 'pt', 'eta', 'phi', 'z_origin', 'rel_tof', 'MET']]])

    
    df_isolated_photons = df_isolated_photons.sort_values(by=['N', 'id'])
    # Set 'N' and 'id' as a multi-index
    df_isolated_photons_multi = df_isolated_photons.set_index(['N', 'id'])
    #print_first_and_last_10(df_isolated_photons_multi)

    return df_isolated_photons_multi

def print_first_and_last_10(df):
    """
    Prints the first 10 rows and the last 10 rows of the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to be printed.
    """
    print("First 10 rows of the DataFrame:")
    print(df.head(10))
    
    print("\nLast 10 rows of the DataFrame:")
    print(df.tail(10))

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

        #print("Evento: ", event)
        # Check if the event has both photons and electrons
        if event in df_photons.index.get_level_values('N') and event in df_leptons.index.get_level_values('N'):
            # Extract photons and electrons in the event
            photons = df_photons.loc[event]
            electrons = df_leptons.loc[event]

            #print("photonsdataframe")
            #print(photons)
            #print("electronsdataframe")
            #print(electrons)
            
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
            
            #print("deltar antes de corte:")
            #print(delta_r)


            # Ignore ΔR values that are too small
            #mandamos a infinito los que tienen cero para que asi nunca puedan ser seleccionados
            delta_r = np.where(delta_r > 1e-15, delta_r, np.inf)
            
            #print("deltar despues de corte: ")
            #print(delta_r)
            
            # Find the minimum ΔR for each photon
            min_delta_r_per_photon = np.min(delta_r, axis=1)

            #print("min_delta_r_per_photon")
            #print(min_delta_r_per_photon)

            #print("min_delta_r_per_photon: ", min_delta_r_per_photon)
            
            # Append the minimum ΔR for each photon in this event to the result array
            min_delta_r_values = np.append(min_delta_r_values, min_delta_r_per_photon)

    return min_delta_r_values

def plot_delta_r_histogram(delta_r_values, alpha, destiny, output_name):
    """
    Plots and saves a histogram of ΔR values.

    Parameters:
    delta_r_values (list): A list of ΔR values to plot.
    destiny (str): The directory where the histogram image will be saved.
    """
    plt.figure(figsize=(10, 6))

    bins = np.arange(0, 0.1, 0.005)  # Bins from 0 to 1000 with steps of 100

    plt.hist(delta_r_values, bins=bins, color='blue', edgecolor='black')
    plt.title(f'{output_name}, {alpha.capitalize()}')
    plt.xlabel('ΔR')
    plt.ylabel('Frequency')
    
    # Save the histogram as a PNG file
    plt.savefig(f"{destiny}/deltaR_{output_name}_{alpha}.png")
    
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

    #print_initial_and_final_lines(electrons)

    #sys.exit("Salimos")

    # Sort the DataFrame by 'N' and 'pt'
    electrons = electrons.sort_values(by=['N', 'pt'], ascending=[True, False])

    g = electrons.groupby('N', as_index=False).cumcount()

    electrons['id'] = g

    electrons = electrons.set_index(['N', 'id'])

    return electrons


#destiny = f"./data/deltaR_nomerge/"
#Path(destiny).mkdir(exist_ok=True, parents=True)


# Define the strings
iso = "iso"
no_iso = "no_iso"

# Create a list containing both strings
modes = [no_iso]

# Define steps in deltaR_min


# Create a list containing both strings
deltaR_mins = [3,4,5]

for deltaR_min in deltaR_mins:
    # Loop through each deltaRmin
    print(f"deltaR_min: {deltaR_min}")
    
    for mode in modes:
        # Perform actions with each mode
        print(f"Processing mode: {mode}")

        for alpha in [4, 5, 6]:

            print("Alpha: ", alpha)
            
            for type in ['ZH', 'WH', 'TTH']:
                
                print("Type: ", type)
                
                origin = f"/Collider/scripts_2208/data/clean/deltaR_change/Alpha{alpha}_dR{deltaR_min}/"

                destiny = f"./data/deltaR_{deltaR_min}/Channels/{mode}/{type}_{alpha}/"
                Path(destiny).mkdir(exist_ok=True, parents=True)

                destiniy_txt = f"/Collider/scripts_2208/data/clean/{deltaR_min}/"
                Path(destiniy_txt).mkdir(exist_ok=True, parents=True)
                
                input_file = origin + f"full_op_{type}_M9_Alpha{alpha}_13_{deltaR_min}_photons.pickle"

                #print("input_photons: ", input_file)

                #print("leptons: ", input_file.replace('photons', 'leptons'))
                
                photons = pd.read_pickle(input_file)

                leptons = pd.read_pickle(input_file.replace('photons', 'leptons'))

                # Create sub DataFrame for electrons (id = 11)
                electrons = leptons[leptons['pdg'] == 11].copy()
                
                #reseteamos el id del electron
                electrons = reset_id_by_pt(electrons)

                # Apply isolation algorithm
                if mode == "iso":
                    photons = isolate_photons(photons, electrons)
                    photons = reset_id_by_pt(photons)

                    electrons = isolate_photons(electrons, photons)
                    electrons = reset_id_by_pt(electrons)


                #print(electrons)
                #sys.exit("Salimos")
                alpha_s = str(alpha)
                # Example usage:

                
                # Calculate ΔR values
                deltaR_ph_elec = calculate_delta_r(photons, electrons)

                np.savetxt(f"{destiniy_txt}/deltaR_Alpha{alpha}_{type}.txt", deltaR_ph_elec)

                print("Start phvselectrons")
                # Plot ΔR histogram
                plot_delta_r_histogram(deltaR_ph_elec, alpha_s, destiny, 'Photons_vs_Electrons')
        
    
