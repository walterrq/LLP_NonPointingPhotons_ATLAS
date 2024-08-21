import pandas as pd
import os

# Origin directory where the files are stored
origin = "/Collider/scripts_2208/data/clean/"

# Get a list of all files in the directory
all_files = os.listdir(origin)

# Filter only the pickle files
pickle_files = [f for f in all_files if f.endswith('.pickle')]


# Separate the pickle files by type
photon_files = [f for f in pickle_files if 'photons' in f]
lepton_files = [f for f in pickle_files if 'leptons' in f]
jet_files = [f for f in pickle_files if 'jets' in f]

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
merge_and_save(photon_files, "megaphoton.pickle")
merge_and_save(lepton_files, "megaleptons.pickle")
merge_and_save(jet_files, "megajets.pickle")
