from pathlib import Path
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool
import numpy as np

import time

# Record the start time
start_time = time.time()

def main(parameters):
    
    
    #De los dataframes, en base al evento y al id, seleccionamos un registro y sacamos datos.
    
    #parameters, nuevamente, son el nombre del archivo y el tipo
    selection = set()
    file_in, type = parameters
    
    #Extraemos el baseout y ponemos el nombre del archivo de salida (complete...hepmc)
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'fulldanilo_{base_out}.hepmc'

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
    new_hepmc = open(file_out, 'w')

    event = -1
    sentences = ''
    #Leemos linea por linea del hepmc
    #Cuando se abre un archivo, el for lo interpreta como readline
    holder = {'v': dict(), 'n5': dict()}
    for sentence in hepmc:
        line = sentence.split()
        if len(line) > 0:

            if line[0] == 'V':
            #estamos guardando el varcode en una variable
                outg = int(line[1])

            if line[0] == 'P':
                    #pid = int(line[1]) #el id lo hacemos integer porque esta como string
                    pdg = line[2]
                    in_vertex = int(line[11])
                    if abs(int(pdg)) in neutralinos:
                        #seleccionamos los eventos que tienen neutrinos
                        selection.add(in_vertex)
    
    hepmc = open(file_in, 'r')

    for sentence in hepmc:
        #Seteeamos estos valores en 0 a menos que usemos un valor diferente para estos valores.
        #Con el delphes modificado, agregamos a las particulas dos mandatory doubles.
        #Todas las particulas tendran los doubles aunque no nos importe calcular su zorigin y tiempo relativo.
        #Lo agregado es z_origin y tiempo relativo.
        #Delphes, cuando se pase el hepmc, modifica para ya no llamar por ubicacion en la lista, si no por nombre y atributo
        zorigin = 0.0
        relt = 0.0
        line = sentence.split()
        if len(line) > 0:
            if line[0] == 'E':
                event += 1
                if (event % 1000) == 0: #loading bar
                    print(f'{base_out}: Event {event}')
                    new_hepmc.write(sentences)
                    sentences = ''
                #print(event)
                
            elif line[0] == 'U': #U: unidades
                params = line[1:]
                if params[0] == 'GEV':
                    p_scaler = 1
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
            
            elif line[0] == 'V': #si la linea es un vertice
                
                outg = int(line[1]) #Tomamos el segundo elemento (Barcode) y lo convertimos en integer. Queremos guardar la informacion de qué vertice se origina la particula
                info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing particles

                ctau = float(line[6])
                #print("We are in V")  
                #print("outg is: ",outg)
            
                #El asterisco permite que la lista sea una secuencia. Ejm: {1,*{2,3},4} -> {1,2,3,4}
                #holder = data[event]
                #Abajo ya guardamos informacion en el diccionario holder. Key=vertice. Recordamos que el valor de v se inicio como un diccionario. Al label barcode int outg del vertice le estamos guardando la info sustraida del vertice (info). El vertice tiene mas info, pero solo guardamos la antes hallada.
                #Ojo que debemos diferenciar 'V': etiqueta del Hepmc, con 'v': etiqueta en el holder.
                #info es una secuencia. Con list lo volvemos lista
                info=list(info)
                
                #Por ahora no no necesitamos los diccionarios
                #holder['v'][outg] = list(info)
            
                # print(outg)
                #Extraemos la info del ctau
                
                #print("ctau")
                #print(ctau)
                #sys.exit("Exiting the script...")
            
            elif line[0] == 'P':
                pid = int(line[1]) #el id lo hacemos integer porque esta como string
                pdg = line[2]

                if (abs(int(line[2])) == 22) and (int(line[11]) == 0): #Si es que la particula es foton y es particula final (si no es negativo)
                   
                    if outg in selection:
                        x, y, z = float(info[0]), float(info[1]), float(info[2])
                        px, py, pz, mass = float(line[3]), float(line[4]), float(line[5]), float(line[7])
                        
                        
                        r = np.sqrt(x ** 2 + y ** 2) #Radius of trajectory
                        
                        pt = np.sqrt(px ** 2 + py ** 2)
                        
                        Et = np.sqrt(mass ** 2 + pt ** 2)
                        E = np.sqrt(mass ** 2 + pt ** 2 + pz ** 2)
                    
                        v_z = np.array([0, 0, 1])  # point in the z axis
                        d_z = np.array([0, 0, 1])  # z axis vector

                        v_ph = np.array([x, y, z])
                        d_ph = np.array([px, py, pz])

                        n = np.cross(d_z, d_ph)

                        n_ph = np.cross(d_ph, n)
                
                        c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z
                        
                        t_n = ctau/c_speed
                                    
                        #si no viene del neutrino pesado, asumimos que es cero tn
                        #Obtenemos los componentes de la velocidad del foton
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

                #si ni siquiera es un foton, asumimos que todo es cero
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

    #print(sentences)
    new_hepmc.write(sentences)
    hepmc.close()
    new_hepmc.close()        
                        #new_observs = new_observs.drop(index = (event,pid))
                    #except KeyError: 
                        #Si no existe la combinacion de evento y pid(porque el foton no pasa los cortes), dar el valor de 0 a las variables
                    #    z_origin = 0.0
                    #    tof = 0.0
                       
                #Agregamos los valores anteriores a la linea donde estan nuestros nuevos datos
                #El primer input de insert (13) indicara el espacio de la lista donde se agrega
                #El segundo input implica lo que se esta añadiendo(relt o zorigin) en esta posicion 13
                #El segundo input empuja el primero. zorigin termina en la posicion 13 y relt en la posicion 14
                
                #Ojo que bastantes valores seran 0, pues solo, en promedio, hay 2 fotones de nuestro interes con los valores distintos de 0
                
                #Construimos nuevamente el sentence agregando el elemento separandolo con espacios en blanco y ponindo un newline.
                #El join junta los elementos de la lista en base al string (en este caso, el espacio '')
                #si usaramos el join con una x en lugar de un espacio en blanco tendriamos lo siguiente:
                    # [1,2] -> 1x2
                #Recordemos line = sentence.split()
                #Entonces, convertimos una lista en un string unido por espacios. Importante para mantener la estructura del hepmc
        
        #En lugar de escribir linea por linea, que es pesado en memoria, juntamos linea por linea y luego, cuando la linea sea 1000 o 2000 lo escribimos todo en newhepmc y reseteamos sentences.
        #Mejor es escribir por banches cada 1000 o 2000 elementos
    

    return

ATLASdet_radius= 1.5 #radio del detector de ATLAS
ATLASdet_semilength = 3.512 #Mitad de la longitud del radio de atlas (metros) (z_atlas)

# Adjusting detector boundaries
r_detec = ATLASdet_radius * 1000  # m to mm
z_detec = ATLASdet_semilength * 1000

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s



destiny = "/Collider/scripts_2208/data/raw/"
types = ["ZH","WH","TTH"]
tevs = [13]
neutralinos = [9900016, 9900014, 9900012, 1000023]

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