import json
import numpy as np
from pathlib import Path
import gc
import glob
import re
from multiprocessing import Pool
import sys


def main(parameters):
    # Programming Parameters
    file_in, type = parameters 
    #extraemos los valores teniendo en cuenta que parametros es una lista de 2 elementos
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    #debuggeo
    print(base_out)
    #sys.exit()
    
    file_out = destiny + f'recollection_photons-{base_out}.json'
    
    it = 0
    i = 0
    limit = 2

    it_start = 0
    batch = 5000
    corte_inf = it_start * batch
    corte_sup = corte_inf + batch * 99999
    final_part_active = True

    # Action
    df = open(file_in, "r") #df open abre el archivo. Se abre el archivo on "read"(no se cambiara nada)
    
    """
    while it < 2:
        #df.readline()
        x = df.readline()
        print(x)
        it += 1
    """
    
    while it < 2:
        df.readline() 
        #el readline avanzo en el archivo
        it += 1
      

    # Initializing values
    data = dict() #Estamos inicializando un diccionario
    num = 0
    p_scaler = None
    d_scaler = None
    #Comentamos el for
    #for sentence in df:
    while i<(limit+10):
        sentence = df.readline() #readline se encarga de los saltos de linea
        # print(sentence)
        # split sentence using white spaces
        line = sentence.split()
        print(sentence)
        print(line)
        i += 1
        
        if num <= corte_inf:
            holder = {'v': dict(), 'a': [], 'n5': dict()}
            tpx = 0
            tpy = 0
            if line[0] == 'E':
                if (num % 500) == 0:
                    print(f'RUNNING: {base_out} ' + f'Event {num}')
                    print(0)
                num += 1
            nfile = it_start + 1
            continue
        elif line[0] == 'E':
            # num = int(line[1])
            if num > 0:  # Selection of relevant particles/vertices in the last event
                # print(mpx,mpy)
                selection = set()
                data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']}
                for n5_k, n5_v in holder['n5'].items():
                    # print(n5_k , n5_i)
                    selection.add(n5_k)
                    selection.add(n5_v[-1])
                for photon in holder['a']:
                    # select only the photons that come from a n5 vertex
                    outg_a = photon[-1]
                    data[num - 1]['a'].append(photon)
                    '''
                    Esto no se necesita ahora que trabajamos con delphes
                    if outg_a in selection:
                        x, y, z = [d_scaler * ix for ix in holder['v'][outg_a][0:3]]
                        # print(x,y,z)
                        r = np.sqrt(x ** 2 + y ** 2)'''
                    selection.add(outg_a)
                for vertex in selection:
                    # select only the vertices that have a neutralino as incoming
                    data[num - 1]['v'][vertex] = holder['v'][vertex]
            # print(data)
            holder = {'v': dict(), 'a': [], 'n5': dict()}
            i += 1
            if (num % 500) == 0:
                print(f'RUNNING: {base_out} ' + f'Event {num}')
                print(len(data))
            if num == nfile * batch:
                with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                    json.dump(data, file)
                print(f'Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}')
                del data
                gc.collect()

                data = dict()
                holder = {'v': dict(), 'a': [], 'n5': dict()}
                nfile += 1
            if num == corte_sup:
                final_part_active = False
                break
            num += 1
        elif line[0] == 'U':
            params = line[1:]
            if params[0] == 'GEV':
                p_scaler = 1
            else:
                p_scaler = 1 / 1000
            if params[1] == 'MM':
                d_scaler = 1
            else:
                d_scaler = 10
            # print(p_scaler)
        elif line[0] == 'V':
            outg = int(line[1])
            info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing
            holder['v'][outg] = list(info)
            # print(outg)
        elif line[0] == 'P':
            pid = line[1]
            pdg = line[2]
            # Extracting the MET of the event
            in_vertex = int(line[11])

            if (abs(int(pdg)) == 22) and (in_vertex == 0):
                # id = int(line[1])
                # px, py, pz, E, m = [float(x) for x in line[3:8]]
                info = int(pid), *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,vertex from where it comes
                holder['a'].append(list(info))
            elif abs(int(pdg)) in neutralinos:
                info = *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,out_vertex
                holder['n5'][in_vertex] = list(info)
    df.close()

    if final_part_active:
        # Event selection for the last event
        selection = set()
        data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']}
        for n5_k, n5_v in holder['n5'].items():
            # print(n5_k , n5_i)
            selection.add(n5_k)
            selection.add(n5_v[-1])
        for photon in holder['a']:
            # select only the photons that come from a n5 vertex
            outg_a = photon[-1]
            data[num - 1]['a'].append(photon)
            if outg_a in selection:
                x, y, z = [d_scaler * ix for ix in holder['v'][outg_a][0:3]]
                # print(x,y,z)
                r = np.sqrt(x ** 2 + y ** 2)
            selection.add(outg_a)
        for vertex in selection:
            # select only the vertices that have a neutralino as incoming
            data[num - 1]['v'][vertex] = holder['v'][vertex]

        # print(data[num])
        # print(data.keys())

        with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
            json.dump(data, file)

        print(f'FINAL {base_out}: Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}\n')
    return

# Particle Parameters
neutralinos = [9900016, 9900014, 9900012, 1000023]
#PDG heavy neutrinos (llamados neutralinos pero son neutrinos pesados)
#Queremos saber en que vertice se encuentran los neutrinos
neutrinos = [12, 14, 16, 1000022]

destiny = "/Collider/scripts_2208/data/clean/"
#destiny es la direccion donde se guardara. Se creara la carpeta clean destiny si es que no existe
#destiny = "/Collider/2023_LLHN_CONCYTEC/"

types = ["ZH","WH","TTH"]
#3 tipos de proceso 

tevs = [13]
#por ahora solo usamos la energia en 13

Path(destiny).mkdir(exist_ok=True, parents=True)
#Path es un paquete importado (pathlib)

allcases = []
#definimos allcases como una lista vacia

#Ahora hacemos un for sobre los tipos de procesos
for typex in types[:]:
    for tevx in tevs[:]:
        #el glob busca nombres de archivos(file names) en las direcciones indicadas con cierta regular expression(regex). El output es un object iterable
        #Este glob pide: saca el archivo run que contenga el tipo de proceso typex(ZH, WH o TTH) que contenga la energia tevx (13)
        #glob.glob te develve una lista de nombres de archivos
        for file_inx in sorted(glob.glob(f"/Collider/scripts_2208/data/raw/run_{typex}*{tevx}.hepmc"))[:]:
            allcases.append([file_inx, typex])
            #con append almacena el nombre del archivo y el tipo de proceso dentro de la variable allcases, ya definida como arreglo.

if __name__ == '__main__':
    with Pool(1) as pool:
        #Pool indica el numero de cores a usar de la PC. Indica la cantidad de veces que este script se corre a la vez.
        pool.map(main, allcases)
        #un map corre una funcion, en este caso el main, sobre todos los elementos. Main es la funcion principal. allcases es la lista que estamos llenando con los filenames. Los filenames son los hepmc. Al lado le estamos poniendo el tipo de proceso(ZH,WH,TTH). Son listas de listas. Cada elemento es una lista de 2 elementos.
        
        #Lo bueno de map es lograr correr varios procesos a la vez con el pool.
        
        
        #podemos imprimir allcases como debuggeo para practicar y podemos correr este python de forma independiente


