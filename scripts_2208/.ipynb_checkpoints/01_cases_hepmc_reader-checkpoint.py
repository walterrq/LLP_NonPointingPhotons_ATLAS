import json
import numpy as np
from pathlib import Path
import gc
import glob
import re
from multiprocessing import Pool
import sys

#Este script extrae los datos del HepMC

#La funcion main no acepta como input listas de listas; si no, listas de elementos. La funcion del main es extraer la informacion importante del HEPMC

import time
# Record the start time
start_time = time.time()

def main(parameters):
    # Programming Parameters
    file_in, type = parameters
    #extraemos los valores teniendo en cuenta que parameters es una lista de 2 elementos

    # Extract label
    #Usamos regular expressions para encontrar los archivos de interes
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    # Name of output file
    
    #f is string in python
    #'.': This matches any character except for a newline.
    #'+': This quantifier means "one or more occurrences" of the preceding element (which is . in this case).
    #'\.': This matches a literal dot (.). The dot needs to be escaped with a backslash
    #The regular expression is looking for a substring in file_in that starts with the value of type ("ZH, WH, TTH") and is followed by one or more of any characters (except a dot), up to the first literal dot encountered.
    
    #los group hacen referencias al label. Cada parentesis indica un grupo de cosas que "atrapa" dentro del patron. 
    #Si quitara el group(1), el patron incluiria el punto, pues solo tomaria la informacion o patrones del primer parentesis. Al poner el group(1) incluyendo a todo, el punto se excluye del patron.
    
    '''
    #debuggeo(ya no)
    Lo que me sale en la consola es:
    print(type) = ZH
    print(base_out) = ZH_M3_Alpha1_13
    print(file_in) = /Collider/scripts_2208/data/raw/run_ZH_M3_Alpha1_13.hepmc
    '''
    
    #construimos el nombre del file en donde pondremos lo que obtengamos del procesos. Lo llamamos usando el label. 
    #Directorio: destiny(ya definido en masterzh),
    #Nombre del archivo donde se guarda: recollection_photons-{base_out}.json
    #Lo util de json es que sirve para guardar diccionarios y no ocupa mucho espacio. Json mantiene la estructura del diccionario, algo que un simple archivo .txt no haria. 
    #Tambien, lo util es que si vuelvo a importar el json, lo importa como diccionario
    file_out = destiny + f'recollection_photons-{base_out}.json'

    it = 0 # Iterator for unnecessary lines
    i = 0
    limit = 2
    
    #El HEPMC tiene muchos eventos y es muy pesado; ante esto, dividimos el hepmc en banches(chunks) de 5000 eventos. Cada 5000 eventos se guardan los datos, se borra la variable y se vuelve a empezar. De este modo, no se satura la RAM.
    
    it_start = 0
    batch = 5000
    corte_inf = it_start * batch
    corte_sup = corte_inf + batch * 99999 #Le damos un numero muy alto para que trabaje con todos los que hay.
    final_part_active = True

    # Open the hepmc
    df = open(file_in, "r")
    #df open abre el archivo. Se abre el archivo on "read"(no se cambiara nada)

    # Reading unnecesary lines SO THAT WE CAN SKIP THEM (we read in order)
    while it < 2:
        df.readline()
        it += 1

    # Initializing values
    data = dict() #Variable donde guardaremos todas nuestras cosas: es un diccionario. Asi se definen los diccionarios vacios y luego se le pueden añadir elementos.
    num = 0 #Evento en el que estamos (evento 1, evento 10000, etc.)
    p_scaler = None
    d_scaler = None
    for sentence in df:
        #Para el debugging comentamos el for y descomentamos el while y sentence.
        #while i<(limit+20):
        #sentence = df.readline()
        # print(sentence)
        
        
        #CADA SENTENCE ES UNA LINEA. Con split, estamos creando una lista line que divida sus elementos por white spaces
        line = sentence.split() # Divide the string in whitespaces
        if num <= corte_inf: #Por como esta seteado el codigo, solo entrara una vez a este if
            
            #En data los values seran un diccionario y estos diccionarios seran el holder. La informacion guardada por evento es la siguiente:
            #v: vertice
            #a: fotones
            #n5: todos los neutrinos pesados (codigos ya definidos abajo, guardados como neutralinos)
            holder = {'v': dict(), 'a': [], 'n5': dict()}
            if line[0] == 'E': 
                #Si estamos empezando un evento
                if (num % 500) == 0: 
                    #Se printea un loading bar para describir en que parte del proceso vamos
                    print(f'RUNNING: {base_out} ' + f'Event {num}')
                    print(0)
                num += 1 #Aumentamos en 1 el numero del evento
            nfile = it_start + 1
            continue 
            #El continue indica que interrumpamos el for o while y volvemos a empezar(pasa a la siguiente iteracion)
        elif line[0] == 'E':
            # num = int(line[1])
            if num > 0:  # Selection of relevant particles/vertices in the last event
                selection = set() # Vertices that interact with a heavy neutrino
                #Lo bueno del set es que solo guarda unique values. Es decir, los elementos seran unicos.
                
                #Muchos de los vertices vienen del neutrino pesado. Estos son los mismos que el origen de ciertos fotones.
                #Por esto, no quiero guardar varias veces un mismo vertice.
                
                data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']} 
                
                #si printeamos data tenemos, por ejemplo: 
                #[778, 0.08743982596008013, 0.22215550068336953, 0.6316336979790049, 0.6752480418856767, 0.0, -440], [779, 0.005544681963453748, -0.012817290247479293, 0.0052780426227581195, 0.014929305452418974, 0.0, -440]
                
                #params: si es gev, mm (ya definido antes)
                #DEPURAMOS los Vertices y fotones pero los neutrinos nos interesan todos. n5 = todos los neutrinos pesados que tengamos
                for n5_k, n5_v in holder['n5'].items(): # Extracting the initial and decaying vertex of the heacvy neutirno
                    #Recordamos que holder['n5'] es un diccionario que tiene como key el vertice que entra. Como ultimo dato, tiene el vertice del que sale
                    #Si solo pusieramos for (sin .items) solo devolveria los values.
                    
                    #Cuando haces un for sobre una lista te da elemento por elemento. CUANDO HACES ITEMS SOBRE UN DICCIONARIO TE DA 2 ELEMENTOS: KEY Y VALUES
                    #n5_v es una lista.
                    
                    #n5 es un diccionario. holder tambien es un diccionario. Tenemos un diccionario dentro de un diccionario.
                    #n5_k es un integer (que sale del key)
                    #n5_v es la lista (valor de la lista)
                    #El vertice en el que entra el neutrino pesado es el key del diccionario. Outg es el ultimo elemento de su value.
                    
                    #outg = vertice donde se origina; in_vertex = vertice al que esta entrando(donde decae la particula)
                    
                    #Quremos solo quedarnos con los vertices de interes, pues el hepmc tiene mucha informacion y no queremos sobrecargarnos
                    
                    # print(n5_k , n5_i)
                    selection.add(n5_k) # decaying vertex (vertice de donde viene: Key)
                    selection.add(n5_v[-1]) # initial vertex (vertice en donde esta decayendo)
                for photon in holder['a']:
                    # add photons
                    
                    #Guardamos todos los fotones porque, en principio, no sabemos de donde vienen los fotones. Luego haremos la correspondencia con un analisis para decidir con cuales nos quedamos
                    
                    outg_a = photon[-1] #photon[-1] = vertex from where it comes
                    data[num - 1]['a'].append(photon)
                    selection.add(outg_a) #Añadimos los vertices relacionados a los fotones
                for vertex in selection:
                    # select only the vertices that have a heavy neutrino or a photon interacting
                    # dentro de selections, vertex es un numero negativo
                    
                    data[num - 1]['v'][vertex] = holder['v'][vertex]
                    
                    #En data solo guardo la informacion de los vertices que me interesa, ubicado en selection.
                    
                #print(data)
            
            #Inicializo nuevamente holder a vacio
            holder = {'v': dict(), 'a': [], 'n5': dict()}
            i += 1
            if (num % 500) == 0: #Loading bar
                print(f'RUNNING: {base_out} ' + f'Event {num}')
                print(len(data))
            if num == nfile * batch: #nfile inicia como 1 y batch=5000
                
                
                #Guardamos en varios json. Necesito poder borrar mi variable.
                #En el segundo script juntamos los json
                with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file: #Esta linea abre un json en write mode. Le da el nombre nfile (batch)
                    json.dump(data, file)
                    #La linea de arriba escribe la variable "data" en el JSON abierto usando json.dump()
                    
                #Tenemos un mensaje de que se esta guardando la info exitosamente
                print(f'Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}')
                del data #elimino la data anterior para liberar la memoria
                gc.collect()
                
                
                #Repetimos el codigo para el ultimo batch
                data = dict()
                holder = {'v': dict(), 'a': [], 'n5': dict()}
                nfile += 1
            if num == corte_sup:
                final_part_active = False
                break
            num += 1
        elif line[0] == 'U': #U: unidades
            params = line[1:] #Esto te dice considera todos los elementos menos el primero. Dame una sublista desde el elemento 1 hacia adelante. (recordar que python indexa desde 0)
            if params[0] == 'GEV':
                p_scaler = 1
            else:
                p_scaler = 1 / 1000 #Si tenemos MEV
            if params[1] == 'MM':
                d_scaler = 1
            else:
                d_scaler = 10 #si tenemos cm
            # print(p_scaler)
        elif line[0] == 'V': #si la linea es un vertice
            outg = int(line[1]) #Tomamos el segundo elemento (Barcode) y lo convertimos en integer. Queremos guardar la informacion de qué vertice se origina la particula
            info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing
            #Arriba sacamos la informacion del vertice, correspondiente a los elementos del 3 al 6 y 8.
            #El asterisco permite que la lista sea una secuencia. Ejm: {1,*{2,3},4} -> {1,2,3,4}
            
            #Abajo ya guardamos informacion en el diccionario holder. Key=vertice. Recordamos que el valor de v se inicio como un diccionario. Al label barcode int outg del vertice le estamos guardando la info sustraida del vertice (info). El vertice tiene mas info, pero solo guardamos la antes hallada.
            #Ojo que debemos diferenciar 'V': etiqueta del Hepmc, con 'v': etiqueta en el holder.
            #info es una secuencia. Con list lo volvemos lista
            holder['v'][outg] = list(info)
            # print(outg)
        elif line[0] == 'P':
            pid = line[1] #definimos el barcode  
            pdg = line[2] #definimos el pdgid
            #Uso como key el in_vertex para el diccionario de n5
            in_vertex = int(line[11]) #A que vertice entra(si es part final, da cero)
            
            #Recordemos que solo queremos analizar los fotones que llegan al detector, los finales; no los intermedios.
            #Ademas, solo nos interesan los fotones finales que vengan de neutrinos pesados.
            
            #Si la particula es final, y es foton, extraemos cantidades y guardamos en info. Momentum, masa, vertice de donde viente.
            if (abs(int(pdg)) == 22) and (in_vertex == 0):
                # id = int(line[1])
                # px, py, pz, E, m = [float(x) for x in line[3:8]]
                info = int(pid), *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,vertex from where it comes
                #Guardo outg del neutrino porque cositas
                holder['a'].append(list(info))
                
                #Si no es un foton final pero esta dentro de nuestra lista de neutrinos pesados, guardamos informacion. Ahora, guardamos la info en el diccionario pero en la parte del n5. 
            elif abs(int(pdg)) in neutralinos:
                info = *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,out_vertex
                holder['n5'][in_vertex] = list(info)
                #El vertice a donde esta entrando el neutrino pesado es el mismo del que esta saliendo el foton de nuestro interes.
                #Usamos un key = vertice a donde entra
    df.close()
    
    #Ojo que solo nos interesan los fotones finales que vengan de neutrinos pesados. De estas particulas queremos los observables. Quiero asociar cada foton con su vertice. Hay vertices que no tendran fotones. Esos no nos interesan.

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
            selection.add(outg_a)
        for vertex in selection:
            # select only the vertices that have a neutralino as incoming
            data[num - 1]['v'][vertex] = holder['v'][vertex]

        with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
            json.dump(data, file)

        print(f'FINAL {base_out}: Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}\n')
    return

# Particle Parameters
neutralinos = [9900016, 9900014, 9900012, 1000023]
neutrinos = [12, 14, 16, 1000022]

destiny = "/Collider/scripts_2208/data/clean/"
#destiny = "/home/cristian/"
types = ["TTH"] #por ahora solo queremos un punto
#types = ["ZH","WH","TTH"]
tevs = [13]

Path(destiny).mkdir(exist_ok=True, parents=True)

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(f"/Collider/scripts_2208/data/raw/run_{typex}*{tevx}.hepmc"))[:]:
            allcases.append([file_inx, typex])

if __name__ == '__main__':
    with Pool(1) as pool:
        pool.map(main, allcases)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

