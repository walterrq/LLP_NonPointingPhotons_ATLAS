from pathlib import Path
import sys
import os
import glob
from multiprocessing import Pool

#This file collects direct inputs from masterZH.sh

#The first input variable is root: where we want the outputs to be saved. $destiny_folder
#The second variable is the Delphes directory. $delphes_folder = '/Collider/MG5_aMC_v2_9_11/Delphes'
def main(in_file_arg):
    #Separate elements in tuple    
    in_file, order = in_file_arg
    #outfile. Same name as the full_op hepmc but ending in root instead of hepmc.
    out_file = in_file.replace('.hepmc', f'_{order}.root').replace(origin, destiny)
    #We run Delphes with the following command
    #First, we go to the Delphes folder, and then we run DelphesHepMC2.
    #Note that the Delphes file path is very specific to my PC.
    print('Order: ', order)
    os.system(f'cd {sys.argv[2]} && ./DelphesHepMC2 '
                #f'/Collider/llpatlas/Delphes_cards/delphes_card_LLHNscanV5.tcl {out_file} {in_file} > /dev/null 2>&1')     
                # We change the delphes card to obtain the tracks in branches.           
                f'/Collider/llpatlas/Delphes_cards/deltaR_change/delphes_card_LLHNscanVbasic_{order}.tcl {out_file} {in_file} > /dev/null 2>&1')
    return 
destiny_base = './data/clean'
types = ['ZH', "WH", "TTH"]
tevs = [13]

root = sys.argv[1]
origin = root + f"/scripts_2208/data/raw/"
destiny = root + f"/scripts_2208/data/clean/deltaR_change/"

#The following command deletes all the root files in the destination.
#This is done because root files cannot be created with the same name, so the previous one with the same name is deleted.
Path(destiny).mkdir(exist_ok=True, parents=True)
os.system(f'cd {destiny} && find . -name \*.root -type f -delete')

#We define the order of the deltaR files: 1e-3, 1e-4 and 1e-5
orders = ['3', '4', '5']

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(origin + f"full_op_{typex}*{tevx}.hepmc"))[:]:
            allcases.append(file_inx)

if __name__ == '__main__':
    allcases_orders = [(file_inx, order) for file_inx in allcases for order in orders]
    #print(allcases_orders)
    with Pool(1) as pool:    
        pool.map(main, allcases_orders)
