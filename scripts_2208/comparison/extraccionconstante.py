import json
import sys
import glob
import re
import numpy as np

for number_alpha56 in [4, 5, 6]:

    # Use the name of your text file
    file_name = f'etayz_new{number_alpha56}.txt'
    f = open(f"eta_histoconstant{number_alpha56}.txt", "w")

    with open(file_name, 'r') as file:
        lines = file.readlines()

    for sentence in lines:
        line = sentence.split()
        R1 = float(line[0])
        R2 = float(line[1])
        zsimpl_value = float(line[4])
        zcrisitan = float(line[5])
        z_atlas_value = np.abs(float(line[6]))
        deltaz = (z_atlas_value - zsimpl_value)
        deltazabs = np.abs((z_atlas_value - zsimpl_value)/z_atlas_value)
        line3 = f"{z_atlas_value} {zsimpl_value} {deltaz} {deltazabs}\n"
        f.write(line3)
        #if( R1 < 1500 or R1 > 1590 or R2 < 1590):
        #    print("R1, R2, anomalo: ", R1,R2)
        #else:
        #    print("No hay R1 R2 anomalo")