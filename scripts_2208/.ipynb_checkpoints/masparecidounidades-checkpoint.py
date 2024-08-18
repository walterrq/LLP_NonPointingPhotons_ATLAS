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
    #Abrimos los hepmc. Uno para leer y otro para escribir
    hepmc = open(file_in, 'r')
    hepmc2 = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')

    event = -1
    sentences = ''
    #Leemos linea por linea del hepmc
    #Cuando se abre un archivo, el for lo interpreta como readline
    
    #Leemos por primera vez el hepmc para crear los diccionarios (forma ineficiente)
    #event2 = -1
    #sentences2 = ''
    #Leemos linea por linea del hepmc
    #Cuando se abre un archivo, el for lo interpreta como readline
    holder = {'v': dict(), 'n5': dict()}
    selectiond = dict()
    num = 0
    selection = set()
    data = dict()

    for sentence2 in hepmc2:
        line2 = sentence2.split()
        if len(line2) > 0:
            if line2[0] == 'E':
                #print(f'RUNNING: {base_out} ' + f'Event {num}')
                

                if(num >= 1):
                    selectiond[num - 1] = list(selection)
                    selection = set()

                    data[num - 1] = {'params': params, 'v': holder['v'], 'n5': holder['n5']}
    
                num += 1

                holder = {'params': [], 'v': dict(), 'n5': dict()}

            elif line2[0] == 'U':
                params = line2[1:]
                if params[0] == 'GEV':
                    p_scaler = 1
                else:
                    p_scaler = 1 / 1000
                if params[1] == 'MM':
                    d_scaler = 1
                else:
                    d_scaler = 10

            elif line2[0] == 'V':
            #estamos guardando el varcode en una variable
                outg2 = int(line2[1])
                #info = *[float(x) for x in line2[3:6]], int(line2[8])  # x,y,z,number of outgoing
                
                #MEJOR ESTARIA INCLUIR AL ctau con
                info2 = *[float(x) for x in line2[3:7]], int(line2[8])  # x,y,z,ctau,number of outgoing
                holder['v'][outg2] = list(info2)

            elif line2[0] == 'P':
                    #pid = int(line[1]) #el id lo hacemos integer porque esta como string
                    pdg = line2[2]
                    in_vertex = int(line2[11])
                    if abs(int(pdg)) in neutralinos:
                        #print("outg2 es: ", outg2)
                        #print("invertex es: ", in_vertex)

                        
                        #seleccionamos los eventos que tienen neutrinos
                        info2 = *[float(x) for x in line2[3:8]], outg2  # px,py,pz,E,m,out_vertex
                        holder['n5'][in_vertex] = list(info2) #de este modo, si tenemos un foton con out_vertex = m, solo hacemos holder['n5'][m]
                        #AÑADIMOS SELECTION
                        #print(info2)
                        
                        selection.add(outg2)
                        selection.add(in_vertex)
        
                    #if (abs(int(pdg)) == 22) and (in_vertex == 0):
                    #    selection.add(outg2)
                        
                    
    hepmc2.close()
    
    data[num - 1] = {'params': params, 'v': holder['v'], 'n5': holder['n5']}
    selectiond[num - 1] = list(selection)
    #selection = set()

    #print(data[0])
    #sys.exit("salimos")
        
    #print(selectiond)
    
    
    
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
                pid = int(line[1]) #el id lo hacemos integer porque esta como string
                pdg = line[2]
                in_vertex = line[11]
                vertex = outg
                
                x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                px, py, pz = float(line[3])* p_scaler, float(line[4])* p_scaler, float(line[5])* p_scaler
                mass_ph = float(line[7]) * p_scaler
                
                r = np.sqrt(x ** 2 + y ** 2) #Radius of trajectory
                
                pt = np.sqrt(px ** 2 + py ** 2)
                Et = np.sqrt(mass_ph ** 2 + pt ** 2)
                E = np.sqrt(mass_ph ** 2 + pt ** 2 + pz ** 2)

                corte_inical =  pt >= 10.0 and not (r >= (r_detec) or abs(z) >= (z_detec))

                #print(selectiond[event])
                viene_del_neutrino = outg in selectiond[event]

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
                    tn = 0.0
                    if(viene_del_neutrino):
                        try:
                            #print(vertex)
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
                        
                            prev_n = (p_n * p_conversion) / (mass_n * mass_conversion)
                            #usar formula relativista vector (p) = gamma.m.vector(v)
                            v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                            # Dividimos la distancia entre la rapidez del NP
                            t_n = dist_n / v_n  # s
                            t_n = t_n * (10 ** 9)  # ns
                        except KeyError:
                            tn = 0.0
                                
                    #fin calculo tn

                    #calculamos t_ph
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
                    
                    #fin calculo t_ph
                    
                    tof = t_ph + t_n
                        
                    prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
                    rel_tof = tof - prompt_tof
                    
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
                
        sentences += sentence 

    sys.exit("Salimos del codigo")
    #print(sentences)
    new_hepmc.write(sentences)
    hepmc.close()
    new_hepmc.close()        

    return

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