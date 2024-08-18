from pathlib import Path
import sys
import os
import glob
from multiprocessing import Pool

#Este archivo recoge los inputs directos desde masterZH.sh

#La primera variable de entrada es root: donde queremos que se guarden los output. $destiny_folder
#La segunda variable es el directorio de delphes. $delphes_folder = '/Collider/MG5_aMC_v2_9_11/Delphes'
def main(in_file):
    #outfile. Mismo nombre del complete hepmc pero en lugar de hepmc termina en root.
    out_file = in_file.replace('.hepmc', '.root').replace(origin, destiny)
    #Corremos delphes con el siguiente comando
    #Primero vamos a la carpeta de delphes y corremos luego corremos DelphesHepMC2
    #Ojo que la direccion del file de delphes es muy especifica de mi pc
    os.system(f'cd {sys.argv[2]} && ./DelphesHepMC2 '
                f'/Collider/limon/Delphes_cards/delphes_card_LLHNscanV5.tcl {out_file} {in_file} > /dev/null 2>&1')
    return
destiny_base = './data/clean'
types = ['ZH', "WH", "TTH"]
tevs = [13]

root = sys.argv[1]
origin = root + f"/scripts_2208/data/raw/"
destiny = root + f"/scripts_2208/data/clean/"

#El comando siguiente elimina todos los roots que haya en el destino.
#Esto se hace porque no se pueden crear roots con el mismo nombre, por eso se borra el anterior con el mismo nombre.
Path(destiny).mkdir(exist_ok=True, parents=True)
os.system(f'cd {destiny} && find . -name \*.root -type f -delete')

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(origin + f"complete_{typex}*{tevx}.hepmc"))[:]:
            allcases.append(file_inx)

if __name__ == '__main__':
    with Pool(1) as pool:
        pool.map(main, allcases)


#Cambiamos de imagen, asi que tendremos que nuevamene hacer cambios en direcciones
#Para esta sesion comenzamos cambiando la locacion donde se guardan los hepmc. Queremos que se guarden en collider/scripts2208/data/raw
#Tambien tenemos que colocar la direccion correcta en el masterZH.sh