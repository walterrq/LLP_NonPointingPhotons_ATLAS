from pathlib import Path
import json
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool
import numpy as np
from my_funcs import my_arctan

import time

# Record the start time
start_time = time.time()

def main(parameters):
    
    global t_n
    #De los dataframes, en base al evento y al id, seleccionamos un registro y sacamos datos.
    
    #parameters, nuevamente, son el nombre del archivo y el tipo


    file_in, type = parameters
    
    #Extraemos el baseout y ponemos el nombre del archivo de salida (complete...hepmc)
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'full_ctb_{base_out}.hepmc'

    #print(f'\nRUNNING: {base_out}') #Commented by JJP
    '''    
    #Antes del comando set_index tenemos un index generico (0,1,2,...)
    #Luego de hacer el set index seteamos las variables de Event y pid como un multiindex
    #El nombre de cada registro estara basado en el numero de su evento y el id del foton
    #Notamos que la etiqueta 0,1,2,3,etc de la primera columna original desaparece porque la estamos sobreescribiendo con el index de las columnas especificas evento e id
    #NO PUEDO USAR ESTO
    #new_observs = new_observs.set_index(['Event','pid'])
    #print(new_observs)
    '''

    #importante para emular codigo de Cristian
    it = 0 # Iterator for unnecessary lines
    i = 0
    limit = 2

    it_start = 0
    #queremos que el codigo se corra de 5000 eventos en 5000 eventos, eso es el batch
    batch = 5000
    #este codigo ya no se usa debido a que se a automatizado este procedimiento
    corte_inf = it_start * batch
    corte_sup = corte_inf + batch * 99999
    final_part_active = True

    ######
    
    #Abrimos los hepmc. Uno para leer y otro para escribir
    df = open(file_in, 'r')
    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')

    event = -1
    sentences = ''
    #Leemos linea por linea del hepmc
    #Cuando se abre un archivo, el for lo interpreta como readline
    while it < 2:
        df.readline()
        it += 1

    # Initializing values
    data = dict()
    #selectiond = dict()
    num = 0
    p_scaler = None
    d_scaler = None
    holder = {'v': dict(), 'a': [], 'n5': dict()}
    selectiond = dict()
    selection = set()
    

    for sentence in df:
        #while i<(limit+20):
        #sentence = df.readline()
        # print(sentence)
        line = sentence.split() # Divide the wtring in whitespaces
        #como en nuestro caso corte_inf es cero, solo correra una vez
        
        
        if line[0] == 'E':
            # num = int(line[1])
            if(num > 0):
                
                #aqui se esta creando un label en el diccionario con valor num -1 que contiene todo lo que esta a la
                #derecha de la igualdad
                data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']}
                for n5_k, n5_v in holder['n5'].items(): # Extracting the initial and decaying vertex of the heavy neutrino
                    #print(n5_k , n5_v)
                    selection.add(n5_k) # decaying vertex
                    #debido a que hicimos list(info) entonces se tiene que dentro de n5 tenemos:
                    # label del vertice decay: lista con px,py,etc... verticesaliente
                    #como vemos el ultimo elemento es el verticesaliente
                    selection.add(n5_v[-1]) # initial vertex
                #print(selection)
                #sacamos info del foton venga del neutrino o no
                for photon in holder['a']:
                    # select only the photons that come from a n5 vertex
                    # add photons
                    outg_a = photon[-1]
                    data[num - 1]['a'].append(photon)
                    #en selection esta el vertice entrante del neutrino pesado y el saliente
                    selection.add(outg_a)
                #set se queda con valores unicos y por ello se borran los duplicados
                #selection2.update(selection)
                #print("num: ", num)
                #print("selection: ", list(selection))
                selectiond[num - 1] = list(selection)
                
                for vertex in selection:
                    # select only the vertices that have a heavy neutrino or a photon interacting
                    #aqui el vertex tiene la misma estructura in_vertex : .... outvertex
                    data[num - 1]['v'][vertex] = holder['v'][vertex]
                #print("the data is:")        
                #print(data)
                #sys.exit("Exiting the script...")
                selection = set()

                holder = {'v': dict(), 'a': [], 'n5': dict()}
            num += 1

            """
            if (num % 500) == 0:
                print(f'RUNNING: {base_out} ' + f'Event {num}')
                print(len(data))
            if num == nfile * batch:
                #with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                #    json.dump(data, file)
                    #estamos guardando en json diferentes por cada batch
                #print(f'Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}')
                del data
                gc.collect()

                data = dict()
                holder = {'v': dict(), 'a': [], 'n5': dict()}
                nfile += 1
            if num == corte_sup:
                final_part_active = False
                break
            num += 1
            """

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
            #estamos guardando el varcode en una variable
            outg = int(line[1])
            #el 8 es el numero de particulas que sale (revisar manual hepmc)
            #el asterisco al inicio hace que sea una sequence
            info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing
            #estamos guardando la informacion relevante del vertice
            #la linea holder['v'] esta accediendo a un diccionario
            #como sabemos, en un diccionario si realizamos dic['camote'] = int(7.5)
            #estamos creando un label llamado camote con el valor int(7.5) dentro
            #de esta forma estamos agregando un label al diccionario v que esta dentro de holder con el valor de list(info)
            holder['v'][outg] = list(info)
            # print(outg)
        elif line[0] == 'P':
            pid = line[1]
            pdg = line[2]
            in_vertex = int(line[11])
            #cual es la diferencia entre outg y in_vertex
	        #outg es el vertice donde aparece o se origina (primer item del v) y el in_vertex es el item 11 del p
	        # in_vertex == 0 implica que la particula no decae
            if (abs(int(pdg)) == 22) and (in_vertex == 0):
                # id = int(line[1])
                # px, py, pz, E, m = [float(x) for x in line[3:8]]
                info = int(pid), *[float(x) for x in line[3:8]], outg  # id px,py,pz,E,m,vertex from where it comes
                holder['a'].append(list(info))
            elif abs(int(pdg)) in neutralinos:
                info = *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,out_vertex
                 # en princip en el diccionario n5 se veria algo asi:
                # numero del vertice : info asociada al px,py,pz,E,m,out_vertex
                holder['n5'][in_vertex] = list(info)
    df.close()
    
    #mismo analisis para el evento final
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

        #selection2.update(selection)
        selectiond[num - 1] = list(selection)
        for vertex in selection:
            # select only the vertices that have a neutralino as incoming
            data[num - 1]['v'][vertex] = holder['v'][vertex]

        #with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
        #    json.dump(data, file)
                        
    
    df.close()
    
    #data[num - 1] = {'params': params, 'v': holder['v'], 'n5': holder['n5']}
    #selectiond[num - 1] = list(selection)
    #selection = set()
    #print(data)
    #sys.exit("salimos")
    #print(data[0])
    #print(list(data.keys()))
    
    #print(selectiond)
    #sys.exit("salimos")
    #print(selectiond)
    
    t_n = None
    for sentence in hepmc:
        #Seteeamos estos valores en 0 a menos que usemos un valor diferente para estos valores.
        #Con el delphes modificado, agregamos a las particulas dos mandatory doubles.
        #Todas las particulas tendran los doubles aunque no nos importe calcular su zorigin y tiempo relativo.
        #Lo agregado es z_origin y tiempo relativo.
        #Delphes, cuando se pase el hepmc, modifica para ya no llamar por ubicacion en la lista, si no por nombre y atributo
        #zorigin = 0.0
        #relt = 0.0
        line = sentence.split()
        #line2= sentence2.split()
        
        if len(line) > 0:
            if line[0] == 'E':

                event += 1
                holder = data[event]
                params = holder['params']

                # Defining scaler according to parameters units
                if params[0] == 'GEV':
                    p_scaler = 1  # GeV to GeV
                elif params[0] == 'MEV':
                    p_scaler = 1 / 1000  # MeV to GeV
                else:
                    #print(params[0])
                    continue

                if params[1] == 'MM':
                    d_scaler = 1  # mm to mm
                elif params[1] == 'CM':
                    d_scaler = 10  # cm to mm
                else:
                    #print(params[1])
                    continue                 
                
                if (event % 1000) == 0: #loading bar
                    print(f'{base_out}: Event {event}')
                    new_hepmc.write(sentences)
                    sentences = ''
                #print(event)
            
            elif line[0] == 'V': #si la linea es un vertice
                outg = int(line[1]) #Tomamos el segundo elemento (Barcode) y lo convertimos en integer. Queremos guardar la informacion de qué vertice se origina la particula
                info = *[float(x) for x in line[3:7]], int(line[8])  # x,y,z,ctau,number of outgoing particles
                #Abajo ya guardamos informacion en el diccionario holder. Key=vertice. Recordamos que el valor de v se inicio como un diccionario. Al label barcode int outg del vertice le estamos guardando la info sustraida del vertice (info). El vertice tiene mas info, pero solo guardamos la antes hallada.
                #info es una secuencia. Con list lo volvemos lista
                info=list(info)

            
            elif line[0] == 'P':
                
                
                vertex = outg
                viene_del_neutrino = outg in selectiond[event]
                if(viene_del_neutrino):
                    pid = int(line[1]) #el id lo hacemos integer porque esta como string
                    pdg = line[2]
                    in_vertex = line[11]
                    #vertex = outg
                    
                
                    x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                    px, py, pz = float(line[3])* p_scaler, float(line[4])* p_scaler, float(line[5])* p_scaler
                    mass_ph = float(line[7]) * p_scaler
                    
                    
                    r = np.sqrt(x ** 2 + y ** 2) #Radius of trajectory
                    
                    pt = np.sqrt(px ** 2 + py ** 2)
                    Et = np.sqrt(mass_ph ** 2 + pt ** 2)
                    E = np.sqrt(mass_ph ** 2 + pt ** 2 + pz ** 2)

                    corte_inical =  pt >= 10.0 and not (r >= (r_detec) or abs(z) >= (z_detec))

                    #print(selectiond[event])
                    #viene_del_neutrino = outg in selectiond[event]

                    #if viene_del_neutrino:
                    #    print(px)

                    #sys.exit("salimos")
                    
                    es_foton_final = (abs(int(line[2])) == 22) and (int(line[11]) == 0)

                    realizar_analisis_cumple = corte_inical and es_foton_final

                    if (realizar_analisis_cumple): #Si es que la particula es foton y es particula final (si no es negativo)

                        #print(vertex)    
                        v_z = np.array([0, 0, 1])  # point in the z axis
                        d_z = np.array([0, 0, 1])  # z axis vector

                        v_ph = np.array([x, y, z])
                        d_ph = np.array([px, py, pz])

                        n = np.cross(d_z, d_ph)

                        n_ph = np.cross(d_ph, n)
                
                        c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z

                        #calculamos tn
                        #topcional = 0.0
                        #global t_n
                        try:
                            
                            #hay problemas con vertex, esta saliendo distinto al vertex
                            #de Cristian,podria ser por la definicion de selection.
                            #vertex_n si esta saliendo igual al de Cristian
                            vertex_n = int(holder['n5'][vertex][-1])
                            #print(vertex_n)
                            #print("outg es : ", outg)
                            #print("vertex_n es : ", vertex_n)
                            #sys.exit("salimos")
                            mass_n = holder['n5'][vertex][-2] * p_scaler

                            
                            # print(mass_n)
                            px_n, py_n, pz_n = [p_scaler*ix for ix in holder['n5'][vertex][0:3]]
                            
                            x_n, y_n, z_n = [d_scaler*ix for ix in holder['v'][vertex_n][0:3]]
                            
                            
                            
                            # print(vertex_n)
                            #Hallamos la distancia entre su vertice de origen y su vertice de decaimiento 
                            #(en el cual se origina el foton delayed)
                            dist_n = np.sqrt((x - x_n) ** 2 + (y - y_n) ** 2 + (z - z_n) ** 2)
                            
                            p_n = np.sqrt(px_n ** 2 + py_n ** 2 + pz_n ** 2)
                            
                            #beta_n = p_n/E_n

                            #print("beta is: ", beta_n)
                            conversionmanual = p_conversion/mass_conversion
                            prev_n2= p_n / mass_n
                            prev_n = prev_n2*conversionmanual
                            

                            #usar formula relativista vector (p) = gamma.m.vector(v)
                            v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                            
                            # Dividimos la distancia entre la rapidez del NP
                            t_n = dist_n / v_n  # s
                            
                            t_n = t_n * (10 ** 9)  # ns

                            ic = 0

                            #print(t_n)
                            #tns.append(t_n)
                            
                            #sys.exit("salimos")
                            #print(vertex)

                        except KeyError:
                        
                            t_n = 0.0
                            ic = 1
                                    
                        #calculamos t_ph
                        #print(t_n)
                        vx = (c_speed * px / np.linalg.norm(d_ph)) * 1000  # mm/s
                        vy = (c_speed * py / np.linalg.norm(d_ph)) * 1000  # mm/s
                        vz = (c_speed * pz / np.linalg.norm(d_ph)) * 1000  # mm/s
                        

                        tr = (-(x * vx + y * vy) + np.sqrt(
                        (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * (r_detec ** 2 - r ** 2))) / (
                            (vx ** 2 + vy ** 2))


                        if tr < 0:
                            tr = (-(x * vx + y * vy) - np.sqrt(
                            (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * ((r_detec) ** 2 - r ** 2))) / (
                                (vx ** 2 + vy ** 2))
                
                        tz = (np.sign(vz) * z_detec - z) / vz
                        
                        if tr < tz:
                            rf = r_detec
                            zf = z + vz * tr
                            t_ph = tr * (10 ** 9)

                            x_final = x + vx * tr
                            y_final = y + vy * tr

                        elif tz < tr:
                            rf = np.sqrt((y + vy * tz) ** 2 + (x + vx * tz) ** 2)
                            zf = np.sign(vz) * z_detec
                            t_ph = tz * (10 ** 9)

                            x_final = x + vx * tz
                            y_final = y + vy * tz

                        else:
                            rf = r_detec
                            zf = np.sign(vz) * z_detec
                            t_ph = tz * (10 ** 9)

                            x_final = x + vx * tz
                            y_final = y + vy * tz
                        
                        #print("tz, rf, r_detec, zf, t_ph, x_final, y_final: ", tz, rf, r_detec, zf, t_ph, x_final, y_final)
                        #fin calculo t_ph
                        
                        #print(t_n)
                        tof = t_ph + t_n
                            
                        prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
                        rel_tof = tof - prompt_tof
                        
                        #print("prompt_tof, rel_tof: ",prompt_tof, rel_tof)
                        #El valor absoluto evita valores negativos para el zorigin
                        z_origin = abs(c_z[-1])

                        #line.insert(13, str(t_v))
                        #line.insert(13, str(t_ph))
                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))

                        sentence = ' '.join(line) + '\n'
                        
                    else:
                        
                        rel_tof = 0.0        
                        z_origin = 0.0
                        #t_v=0.0
                        #t_ph=0.0
                        
                        #line.insert(13, str(t_v))
                        #line.insert(13, str(t_ph))
                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))
                        
                        sentence = ' '.join(line) + '\n'
                else:
                        
                        rel_tof = 0.0        
                        z_origin = 0.0
                        #t_v=0.0
                        #t_ph=0.0
                        
                        #line.insert(13, str(t_v))
                        #line.insert(13, str(t_ph))
                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))
                        
                        sentence = ' '.join(line) + '\n'

        sentences += sentence 

    #sys.exit("Salimos de cod Walter")
    #print(sentences)
    new_hepmc.write(sentences)
    hepmc.close()
    new_hepmc.close()        

    return

t_n = None
ATLASdet_radius= 1.5 #radio del detector de ATLAS
ATLASdet_semilength = 3.512 #Mitad de la longitud del radio de atlas (metros) (z_atlas)

# Adjusting detector boundaries
r_detec = ATLASdet_radius * 1000  # m to mm
z_detec = ATLASdet_semilength * 1000

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s

neutralinos = [9900016, 9900014, 9900012, 1000023]

destiny = "/Collider/scripts_2208/data/raw/"
types = ["ZH","WH","TTH"]
tevs = [13]

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        #Nuevamente, abrimos los hepmc para reescribirlos
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