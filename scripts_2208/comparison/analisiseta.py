import json
import sys
import glob
import re
import numpy as np

for number_alpha56 in [4, 5, 6]:

    file_name = f"etaforptmax{number_alpha56}.txt"
    f = open(f"etayz_new{number_alpha56}.txt", "a")
    # Open the file and read the lines
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Count the number of lines with more than a 5% error
    for sentence in lines:
        line = sentence.split()
        eta1 = np.abs(float(line[2]))
        eta2 = np.abs(float(line[3]))
        if(eta1 < 1.4 and eta2 < 1.4):
            f.write(sentence)
    # Use the name of your text file

    

