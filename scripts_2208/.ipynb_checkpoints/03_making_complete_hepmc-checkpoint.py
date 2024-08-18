from pathlib import Path
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool

import time

# Record the start time
start_time = time.time()

def main(parameters):

    file_in, type = parameters

    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'complete_{base_out}.hepmc'

    #print(f'\nRUNNING: {base_out}') #Commented by JJP

    new_observs = pd.read_pickle(f'/Collider/scripts_2208/data/clean/photon_df-{base_out}.pickle')
    #print(new_observs)
    #keys = new_observs[['Event','pid']].values.tolist()
    #print(keys)
    #print("\n")
    #estamos desapareciendo el indice por default 0,1,2,3,4,5 y los nuevos indices seran "Event" y "pid"
    new_observs = new_observs.set_index(['Event','pid'])
    #print(new_observs)
    
    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')

    event = -1
    sentences = ''
    #print(hepmc)
    
    for sentence in hepmc:
        #print(sentence)
    
        zorigin = 0.0
        relt = 0.0
        #line es una lista
        line = sentence.split()
        if len(line) > 0:
            if line[0] == 'E':
                event += 1
                #if (event % 100) == 0:
                #    print(f'{base_out}: Event {event}')
                if (event % 1000) == 0:
                    new_hepmc.write(sentences)
                    sentences = ''
                #print(event)
            elif line[0] == 'P':
                pid = int(line[1])
                #si es un foton y es un particula final (nos limitamos a ese caso)
                if (abs(int(line[2])) == 22) and (int(line[11]) == 0):
                    try:
                        #hacer un calculo rapido para el t y zorigin sin necesidad
                        #de hacer la busqueda en el pickle
                        #aqui se usa el multi-indice para acceder a la informacion de la serie
                        this = new_observs.loc[(event,pid)]

                        #print(this)
                        #sys.exit("salimos del codigo")
                        #print(event)
                        #print("\n")
                        #print(pid)
                        #print(this)
                        #extrae el valor de z_origin
                        zorigin = this.z_origin
                        #extrae el valor de rel_tof
                        relt = this.rel_tof
                        new_observs = new_observs.drop(index = (event,pid))
                    except KeyError:
                        zorigin = 0.0
                        relt = 0.0
                #con esta linea empujamos el valor que se encuentra en 13 a la derecha
                #relt termina en la posicion 15 o indice 14
                line.insert(13, str(relt))
                #zorigin termina en la posicion 14 o indice 13
                line.insert(13, str(zorigin))
                
                #line es una lista (1,2,3,....15)
                #junta los elementos de una lista en base al string espacio devolviendo un string
                sentence = ' '.join(line) + '\n'
        #acumulamos el string en sentences y luego lo liberamos en 1000
        sentences += sentence
    new_hepmc.write(sentences)
    hepmc.close()
    new_hepmc.close()
    
    return


destiny = "/Collider/scripts_2208/data/raw/"
types = ["TTH"] #por ahora solo queremos un punto
#types = ["ZH","WH","TTH"]
tevs = [13]

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(f"/Collider/scripts_2208/data/raw/run_{typex}*{tevx}.hepmc"))[:]:
            allcases.append([file_inx, typex])

if __name__ == '__main__':
    with Pool(1) as pool:
        #print(allcases[-1:])
        pool.map(main, allcases)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
