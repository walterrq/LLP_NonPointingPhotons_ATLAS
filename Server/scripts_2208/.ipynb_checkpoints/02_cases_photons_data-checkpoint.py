import re
import json
import numpy as np
from pathlib import Path
import pandas as pd
from my_funcs import my_arctan
#Hacemos una funcion my_arctan propia porque numpy trabaja en el rango de -pi a pi y yo quiero que abarque de 0 a 2pi.
#La funcion se encuentra en limon/scripts2208
import sys
import glob
from multiprocessing import Pool

import time

# Record the start time
start_time = time.time()

#En este script obtenemos los observables usando las cosas que extrajimos del HepMC en el codigo 1: tiempo de vuelo, z origin, t gamma
#Estos observables son los medidos en un detector.
#Luego, se reescribira el HepMC y pasarlo por delphes

#Para calcular z origin se necesita: punto de decaimiento del neutrino pesado (x,y,z), velocidad de neutrino pesado, momento del foton (x,y,z)
#Cuando Delphes procese estos datos solo dara lo que el detector observaria(foton final) 
#Si tenemos solo el foton final, sin ninguna informacion del neutrino pesado, no podemos calcular el z origin ni el t gamma. Por eso este proceso se hace pre-Delphes

def main(paremeters):
    
    #Recordamos que solo registramos fotones que tengan >10GeV y que se originen dentro del detector
    #Los unicos fotones que cumplen estos requerimientos deberian ser los que vienen de los neutrinos pesados.
    #En principio, los fotones que vienen del proceso que nos interesa son 2.
    #Si hiciera un histograma de cuantos fotones tendremos por evento, en el dataframe saldria un pico en 2.
    #Siempre pueden haber mas fotones provinientes de otras fuentes.
    
    #El main recibe como input parameters. Parameters es un objeto que tiene una lista con los nombres de los json y la etiqueta (especificado luego del main)
    
    #No todas estas listas son necesarias. Lo fueron para la parte inicial del codigo.
    z_prompts=[]
    z_origin = []
    pts = []
    pzs = []
    tofs = []
    tofs_b = []
    tofs_e = []
    p_tofs = []
    rel_tofs = []
    nus = []
    tns = []
    tphs = []
    trs = []
    tzs = []
    counter = 0
    dicts = []

    files, base_out = paremeters
    for file_in in files[:]:

        #print(file_in) #Commented by JJP
        
        #Borramos la data anterior por memoria
        try: #Intenta si se puede, si no normal
            del data
        except UnboundLocalError: #Si la data no existe (y no hay nada que borrar), no te preocupes
            file_in
        
        #Abrimos el json del file in especifico y asociamos esta estructura de diccionario (json) a la variable data
        with open(file_in, 'r') as file:
            data = json.load(file)
        #print(len(data.keys())) #Commented by JJP
        
        #Recordemos que en el json los keys del diccionario principal son los eventos (numero del evento 0,1,2,...). Por esto, ahora hacemos un for con una variable event.
        for event in list(data.keys())[:]:
            #if (int(event) % 500) == 0:
                #print(f'RUNNING: {base_out} - ATLAS - Event {event}') #Commented by JJP
                
                #Usualmente los keys (params, event, etc) son strings, pero no siempre lo seran. Los keys de diccionarios pueden ser integers o doubles tambien.
                #Si me refiero a un string, no lo puedo llamar sin comilla
                
                #Recordemos que la estructura del diccionario es la siguiente:
                #data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']} 
                
                
            holder = data[event]
            params = holder['params']
            # Defining scaler according to parameters units
            if params[0] == 'GEV':
                p_scaler = 1  # GeV to GeV
            elif params[0] == 'MEV':
                p_scaler = 1 / 1000  # MeV to GeV
            else:
                print(params[0])
                continue

            if params[1] == 'MM':
                d_scaler = 1  # mm to mm
            elif params[1] == 'CM':
                d_scaler = 10  # cm to mm
            else:
                print(params[1])
                continue

            # Adjusting detector boundaries
            r_detec = ATLASdet_radius * 1000  # m to mm
            z_detec = ATLASdet_semilength * 1000

            # Define our holder for pairs:

            ix = 1
            
            #De los fotones nos interesa saber los observables
            for photon in holder['a']:
                info = dict() #Definimos diccionario vacio
                info['Event'] = int(event) #Hago que el key sea el numero del event en integer (pues originalmente esta en string)
                
                #El diccionario info guardara las cosas que queremos. Agregamos el key event porque queremos saber desde que evento corresponde el foton del que vamos a sacar el observable. (seria como hacer un append pero a un diccionario para anadir un key)
                
                #Recordemos que la estructura del diccionario que incluye la informacion del foton tiene la siguiente estructura:
                # px,py,pz,E,m,vertex from where it comes
                
                #Del anterior codigo(01):
                #info = int(pid), *[float(x) for x in line[3:8]], outg
                #holder['a'].append(list(info))
                
                vertex = str(photon[-1])
                pid = photon[0]
                px, py, pz = [p_scaler*ix for ix in photon[1:4]]
                #Recordemos no tenemos la informacion del x,y,z del foton, pero tenemos el vertice de donde viene. Usando la informacion del diccionario de vertices que tenemos, extraemos la informacion de este vertice (x,y,z)
                x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                mass_ph = photon[-2] * p_scaler #Ya sabemos que es 0, pero el archivo bota un numero aproximado a cero (fluctuaciones experimentales) 
                r = np.sqrt(x ** 2 + y ** 2) #Trabajamos con el radio de trayectoria
                # Calculating transverse momentum
                pt = np.sqrt(px ** 2 + py ** 2)
                Et = np.sqrt(mass_ph ** 2 + pt ** 2)
                E = np.sqrt(mass_ph ** 2 + pt ** 2 + pz ** 2)
                
                #Guardamos los datos hallados en info (incluye los datos del foton)
                
                # print(mass_ph)
                info['pid'] = pid
                #No queremos el radio como tal. Solo queremos saber el radio y z de la posicion con respecto al detector.
                #Por ello, si el valor de la fraccion > 1 decae fuera del detector. fraccion <1 decae dentro del detector
                info['r'] = r / r_detec 
                info['z'] = z / z_detec
                info['px'] = px
                info['py'] = py
                info['pt'] = pt
                info['pz'] = pz
                info['ET'] = Et
                info['E'] = E
                ix += 1
                
                #Filtramos para solo analizar los fotones de nuestro interes. Si no cumple estos requisitos, continua con el otro foton
                if pt < 10.0:
                    continue
                elif r >= (r_detec) or abs(z) >= (z_detec):
                     continue

                # Calculating the z_origin of each photon
                v_z = np.array([0, 0, 1])  # point in the z axis
                d_z = np.array([0, 0, 1])  # z axis vector

                v_ph = np.array([x, y, z])
                d_ph = np.array([px, py, pz])

                n = np.cross(d_z, d_ph)

                n_ph = np.cross(d_ph, n)
                
                #Si el z_origin es una medida virtual; puede aparecer fuera del detector. Es una medida de donde hubiera venido el foton si fuese prompt. Si esta a 10 veces la distancia del detector, sabremos que no viene del detector. 
                #Queremos encontrar fotones con z_origin largos para saber que son desplazados.
                
                #@ == producto punto
                #c_z: punto de cruce: punto de interseccion (punto mas cercano si no hubiera cruce)
                c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z

                # Calculating the time of flight
                #Usamos try except pues no todos los fotones vienen de un neutrino
                try:
                    # Time of the neutralino
                    vertex_n = str(holder['n5'][vertex][-1]) 
                    #Lo de arriba extrae out_g del n5. El vertice en donde esta entrando el neutrino pesado es el vertice del foton. 
                    #Recordemos que outg es el vertice de donde viene el neutrino pesado
                    #Nos interesa el outg del n5 (vertice donde se origina), pues queremos la posicion de origen del n5
                    
                    #Si no hay un numero asociado al out vertex del n5, significa que este foton no viene de un n5
                    #Si no encuentras este key, hacemos el tiempo del neutrino 0 y tomamos al foton como prompt.
                    #Recordemos que time of flight = t(gamma) + t(n5). 
                    #En un caso con foton prompt, t(n5)=0
                    
                    mass_n = holder['n5'][vertex][-2] * p_scaler #recordemos que el penultimo valor es la masa. Convertimos la escala correcta
                    # print(mass_n)
                    px_n, py_n, pz_n = [p_scaler*ix for ix in holder['n5'][vertex][0:3]]
                    x_n, y_n, z_n = [d_scaler*ix for ix in holder['v'][vertex_n][0:3]]
                    # print(vertex_n)
                    dist_n = np.sqrt((x - x_n) ** 2 + (y - y_n) ** 2 + (z - z_n) ** 2) #Distancia entre el origen del n5 y el origen del foton
                    p_n = np.sqrt(px_n ** 2 + py_n ** 2 + pz_n ** 2) #Magnitud del momento del neutrino
                    
                    #Hacemos una conversion puesto que quiero la velocidad el m/s o mm/s. Para esto, multiplico por los factores de conversion, pues la masa y los momentum estan en GeV
                    #Factores de conversion definidos fuera del main.
                    
                    #Recordemos v = (p/m)/ (1+((p/m)/c)^2)^(1/2)
                    
                    prev_n = (p_n * p_conversion) / (mass_n * mass_conversion) #p/m con conversion de unidades
                    v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                    
                    #Con v_n tenemos la velocidad a la que viajo el neutrino
                    #Queremos que el tiempo de llegada y el z_origin esten en segundo y milimetros por motivos de unidades en analisis
                    
                    t_n = dist_n / v_n  # s
                    t_n = t_n * (10 ** 9)  # ns
                    tns.append(t_n)
                    #print(z)
                    ic = 0 #variable de debugging
                except KeyError:
                    t_n = 0.0
                    #x= y = z = r = 0.0
                    ic = 1
                    z_prompts.append(z)
                    #print(z)
                #print(t_n)
                
                # Now, time of the photon
                #El tiempo de vuelo del foton terminara porque choca con el detector.
                #Puede chocar tanto por el lado radial(endcaps - tapas del cilindro) como por el lado longitudinal(barrel - cilindro)
                
                #Tendremos dos tiempos para cada foton: el tiempo que le demora al foton llegar al radio y el tiempo que le demora en llegar al endcap. Al final, el tiempo de vuelo sera el menor de esos
                
                
                #Obtenemos los componentes de la velocidad del foton
                vx = (c_speed * px / np.linalg.norm(d_ph)) * 1000  # mm/s
                vy = (c_speed * py / np.linalg.norm(d_ph)) * 1000  # mm/s
                vz = (c_speed * pz / np.linalg.norm(d_ph)) * 1000  # mm/s
                
                #Aplicamos la formula(larga) para hallar el tr
                #1ra forma:
                tr = (-(x * vx + y * vy) + np.sqrt(
                    (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * (r_detec ** 2 - r ** 2))) / (
                         (vx ** 2 + vy ** 2))
                
                #Si lo anterior es menor a 0, usamos la otra forma para obtener la parte positiva.
                if tr < 0:
                    tr = (-(x * vx + y * vy) - np.sqrt(
                        (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * ((r_detec) ** 2 - r ** 2))) / (
                             (vx ** 2 + vy ** 2))
                
                #recordemos: z_detec = ATLASdet_semilength * 1000, m to mm
                #El tiempo negativo puede ser negativo, pero implica que paso en el pasado
                #Tiempo de llegada al z (endcap)
                
                #Si mi velocidad tiene componente en z negativo, la unica opcion es que llegue al endcap negativo. No llegara al endcap positivo
                #Por esto, uso el endcap que tenga el mismo signo que la velocidad.
                #Si mi velocidad es negativa, considero que llegara al endcap negativo. Si la velocidad es positiva, llegara al endcap positivo.
                
                tz = (np.sign(vz) * z_detec - z) / vz

                # Now we see which is the impact time
                #Nos quedamos con el menor
                if tr < tz:
                    #En este caso, se que chocara en el radio. Tengo que calcular en que punto de x chocara.
                    # rf = r_detec
                    rf = r_detec
                    zf = z + vz * tr
                    t_ph = tr * (10 ** 9)

                    x_final = x + vx * tr
                    y_final = y + vy * tr

                elif tz < tr:
                    #En este caso, chocara en el endcap. Tengo que calcular el punto radial donde chocara
                    rf = np.sqrt((y + vy * tz) ** 2 + (x + vx * tz) ** 2)
                    zf = np.sign(vz) * z_detec
                    t_ph = tz * (10 ** 9)

                    x_final = x + vx * tz
                    y_final = y + vy * tz

                else:
                    #Caso, muy poco probable, en que ambos tiempos anteriores sean iguales. Llega en la esquina.
                    #Tiempo de vuelo puede ser cualquiera de los dos.
                    rf = r_detec
                    zf = np.sign(vz) * z_detec
                    t_ph = tz * (10 ** 9)

                    x_final = x + vx * tz
                    y_final = y + vy * tz
                
                #Suma del tiempo del neutrino mas el tiempo del foton
                tof = t_ph + t_n
                
                #Tiempo que le toma a un foton prompt, con el mismo momento, en llegar al mismo punto en el detector detector.
                #Quiero saber cuanto mas se demora mi foton en llegar en comparacion a un foton prompt
                prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
                rel_tof = tof - prompt_tof
                
                #Calculamos el theta y phi para calcular el pseudorapidity (de la posicion final)
                #Este calculo ya no es tan importante porque Delphes lo hace.
                phi = my_arctan(y_final, x_final)
                theta = np.arctan2(rf, zf)
                nu = -np.log(np.tan(theta / 2))

                counter += 1
                #Queremos el componente en z del punto de interseccion
                #Recordemos que c_z es el punto mas cercano entre las dos rectas(la trayectoria del foton y el eje z)
                z_origin.append(c_z[-1])
                pts.append(pt)
                pzs.append(pz)
                tofs.append(tof)
                if abs(nu) < abs(-np.log(np.tan(np.arctan2(r_detec, z_detec) / 2))):
                    tofs_b.append(tof)
                else:
                    tofs_e.append(tof)
                p_tofs.append(prompt_tof)
                rel_tofs.append(rel_tof)
                nus.append(nu)
                #print(t_n)
                tphs.append(t_ph)
                trs.append(tr * (10 ** 9))
                tzs.append(tz * (10 ** 9))

                info['eta']=nu
                info['phi']=phi
                info['z_origin'] = abs(c_z[-1])
                info['rel_tof'] = rel_tof

                dicts.append(info)

    #print(f'Detected photons in ATLAS: {counter}') #Commented by JJP
    
    #Convierto el diccionario en un dataframe
    #Dataframe = tablas tipo excel con propiedades propias e identificadores en los rows y columns
    #Se extrae la info con pandas
    dicts = pd.DataFrame(dicts)
    #print(dicts) #Commented by JJP
    
    #Guardo el dataframe del diccionario en un pickle
    #pickle: forma de guardar los dataframe adecuados para evitar errorres con xlsx(excel) 
    dicts.to_pickle(destiny_info+f'photon_df-{base_out}.pickle')
    #print('df saved!') #Commented by JJP
    return


ATLASdet_radius= 1.5 #radio del detector de ATLAS
ATLASdet_semilength = 3.512 #Mitad de la longitud del radio de atlas (metros) (z_atlas)

#Conversiones para unidades(util para los calculos de tiempo de llegada, etc):

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s

types = ["TTH"] #por ahora solo queremos un punto
#types = ["ZH","WH","TTH"]
tevs = [13]
destiny_info = '/Collider/scripts_2208/data/clean/'

allcases = [] #inicializamos allcases como lista vacia
for type in types[:]: #Iteramos en todos los types 
    for tev in tevs[:]: #Iteramos en todos los posibles tevs (en este caso, solo hay una opcion: 13)
        mwpairs = set(re.search(f'({type}.+)\-', x).group(1) for x in
                      glob.glob(f'/Collider/scripts_2208/data/clean/recollection_photons-{type}*{tev}-*.json'))
        #mwpair es una lista de etiquetas
        #set usa reg ex para extraer lo siguiente la parte del nombre del archivo que se encuentra entre "type" y "-"
        #Con esto, se crea una lista de labels.

        for base_out in sorted(list(mwpairs))[:]:
            allcases.append(
                [sorted(glob.glob(f'/Collider/scripts_2208/data/clean/recollection_photons-{base_out}-*.json')),base_out]
            )
            #allcases es una lista de listas ; cada lista interna tiene dos elementos.
            #El primer elemento de allcases es una lista, el segundo es una etiqueta.
            #La primera es una lista con todos los json (notar el *) que corresponden a esa etiqueta. El segundo elemento es una etiqueta.

            #glob.glob -> Permite iterar a lo largo de todos los archivos, en cierto folder, que le des como input y los vuelve en lista
            
if __name__ == '__main__':
    with Pool(1) as pool:
        pool.map(main, allcases)
        
        
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")


