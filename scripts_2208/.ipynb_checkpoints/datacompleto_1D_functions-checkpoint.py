from pathlib import Path
import json
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool
import numpy as np
from my_funcs import my_arctan

'''
La idea ahora es no hacer un hepmc desde cero, si no editar solo las lineas extras del hepmc original.
Este codigo optimizando diccionarios demora X segundos en compilar
'''


import time

# Record the start time
start_time = time.time()

def timepositive(pgamma, rn, Ri):
    normpgamma = np.linalg.norm(pgamma)
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    rx_n, ry_n = rn[:2]
    px_gamma, py_gamma = pgamma_xy
    
    term1 = -(rx_n*px_gamma + ry_n*py_gamma)

    term1_c = (rx_n*py_gamma - ry_n*px_gamma)
    term2 = Ri*pt_gamma*np.sqrt(1-pow(term1_c/Ri*pt_gamma, 2))
    totalterm = term1 + term2
    return normpgamma*totalterm/pow(pt_gamma,2)

def timenegative(pgamma, rn, Ri):
    normpgamma = np.linalg.norm(pgamma)
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    rx_n, ry_n = rn[:2]
    px_gamma, py_gamma = pgamma_xy
    
    term1 = -(rx_n*px_gamma + ry_n*py_gamma)

    term1_c = (rx_n*py_gamma - ry_n*px_gamma)
    term2 = -Ri*pt_gamma*np.sqrt(1-pow(term1_c/Ri*pt_gamma, 2))
    totalterm = term1 + term2
    return normpgamma*totalterm/pow(pt_gamma,2)

def epsilonsevaluator(epsilon1,epsilon2,timepositiveR1, timenegativeR1, timepositiveR2,  timenegativeR2):
    if(timepositiveR1<timepositiveR2):
        epsilon1 = 1
        epsilon2 = 1
    elif(timepsitiveR1<timenegativeR2):
        epsilon1 = 1
        epsilon2 = -1
    elif(timenegativeR1<timepositiveR2):
        epsilon1 = -1
        epsilon2 = 1
    elif(timenegativeR1<timenegativeR2):
        epsilon1 = -1
        epsilon2 = -1
    return epsilon1, epsilon2

def zsimpl(pgamma, rn):
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    rx_n, ry_n, rz_n = rn[:]
    px_gamma, py_gamma = pgamma_xy
    pz_gamma = pgamma[-1]
    return (rz_n - ((pz_gamma/pow(pt_gamma,2))*(rx_n*px_gamma + ry_n*py_gamma)))

def z_atlas(pgamma, rn, R1, R2, epsilon1, epsilon2):
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    pt_gamma_s = pow(pt_gamma,2)
    R1_s = pow(R1,2)
    R2_s = pow(R2,2)
    rx_n, ry_n, rz_n = rn[:]
    px_gamma, py_gamma = pgamma_xy
    pz_gamma = pgamma[-1]
    term1_c = (rx_n*py_gamma - ry_n*px_gamma)
    multiplier = (pz_gamma/pt_gamma)*((R1*R2)/(R2-R1))

    term1 = epsilon1*np.sqrt(1-pow(term1_c/R1_s*pt_gamma_s,2))
    term2 = -epsilon2*np.sqrt(1-pow(term1_c/R2_s*pt_gamma_s,2))
    z_simpl_v = zsimpl(pgamma, rn)
    z_atlas_v = z_simpl_v + multiplier*(term1+term2)
    return z_atlas_v


def main(parameters):
    
    global t_n

    file_in, type = parameters
    
    #Extraemos el baseout y ponemos el nombre del archivo de salida (complete...hepmc)
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'full_op_{base_out}.hepmc'

    #print(f'\nRUNNING: {base_out}') #Commented by JJP

    #importante para emular codigo de Cristian
    it = 0 # Iterator for unnecessary lines
    i = 0
    limit = 2

    it_start = 0
    batch = 5000
    corte_inf = it_start * batch
    corte_sup = corte_inf + batch * 99999
    final_part_active = True
    
    #Abrimos los hepmc. Uno para leer y otro para escribir
    #df = open(file_in, 'r')
    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')
    #Estamos creando desde cero un nuevo hepmc
    

    
    sentences = ''
    
    data = dict()
    num = 0
    p_scaler = None
    d_scaler = None
    
    
    t_n = None

    for sentence in hepmc:     

        line = sentence.split()

        if len(line) > 0:

            if line[0] == 'E':
                holder = {'v': dict(), 'a': [], 'n5': dict()}
                new_hepmc.write(sentence)

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
                
                #if (event % 1000) == 0: #loading bar
                #    print(f'{base_out}: Event {event}')
                new_hepmc.write(sentence)
                    #sentences = ''
                #print(event)
            
            elif line[0] == 'V': #si la linea es un vertice
                outg = int(line[1])
                info = *[float(x) for x in line[3:7]], int(line[8])  # x,y,z,ctau,number of outgoing particles
                info=list(info)
                holder['v'][outg] = info
                
                new_hepmc.write(sentence)
                
            elif line[0] == 'P':       
                
                pid = line[1]
                pdg = line[2]
                in_vertex = int(line[11])
                vertex = outg

                if (abs(int(pdg)) == 22) and (in_vertex == 0):
            
                    info = int(pid), *[float(x) for x in line[3:8]], outg  # id px,py,pz,E,m,vertex from where it comes
                    holder['a'].append(list(info))
                    #selection.add(outg)

                elif abs(int(pdg)) in neutralinos:
                    info = *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,out_vertex
                    holder['n5'][in_vertex] = list(info)
    
                
                es_foton_final = (abs(int(pdg)) == 22) and (in_vertex == 0)
                

                if(es_foton_final):
                    pid = int(line[1]) #el id lo hacemos integer porque esta como string
                    pdg = line[2]
                    in_vertex = line[11]
            
                    x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                    px, py, pz = float(line[3])* p_scaler, float(line[4])* p_scaler, float(line[5])* p_scaler
                    mass_ph = float(line[7]) * p_scaler          
                    
                    r = np.sqrt(x ** 2 + y ** 2) #Radius of trajectory
                    
                    pt = np.sqrt(px ** 2 + py ** 2)
                    Et = np.sqrt(mass_ph ** 2 + pt ** 2)
                    E = np.sqrt(mass_ph ** 2 + pt ** 2 + pz ** 2)

                    corte_inical =  pt >= 10.0 and not (r >= (r_detec) or abs(z) >= (z_detec))
                    
                    #es_foton_final = (abs(int(line[2])) == 22) and (int(line[11]) == 0)

                    realizar_analisis_cumple = corte_inical
                    

                    if (realizar_analisis_cumple): #Si es que la particula es foton y es particula final (si no es negativo)    
                        v_z = np.array([0, 0, 1])  # point in the z axis
                        d_z = np.array([0, 0, 1])  # z axis vector

                        v_ph = np.array([x, y, z])
                        d_ph = np.array([px, py, pz])

                        n = np.cross(d_z, d_ph)

                        n_ph = np.cross(d_ph, n)
                
                        c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z

                        #calculamos t_n
                        try:                                                        
                            vertex_n = int(holder['n5'][vertex][-1])

                            mass_n = holder['n5'][vertex][-2] * p_scaler

                            
                            px_n, py_n, pz_n = [p_scaler*ix for ix in holder['n5'][vertex][0:3]]                         
                            x_n, y_n, z_n = [d_scaler*ix for ix in holder['v'][vertex_n][0:3]]
                            dist_n = np.sqrt((x - x_n) ** 2 + (y - y_n) ** 2 + (z - z_n) ** 2)                           
                            p_n = np.sqrt(px_n ** 2 + py_n ** 2 + pz_n ** 2)

                            #definimos lo necesario para operar en la funcion
                            p_vector = np.array([px, py, pz])
                            r_n_vector = np.array([x_n, y_n, z_n])
                            R1 = 1.5453
                            R2 = 1.7592

                            timepositiveR1 = timepositive(p_vector, r_n_vector, R1)
                            timenegativeR1 = timenegative(p_vector, r_n_vector, R1)

                            timepositiveR2 = timepositive(p_vector, r_n_vector, R2)
                            timenegativeR2 = timenegative(p_vector, r_n_vector, R2)

                            print(timepositiveR1, timenegativeR1, timepositiveR2, timenegativeR2)

                            if((timepositiveR1 < 0 and timenegativeR1 < 0) or (timepositiveR2 < 0 and timenegativeR2 < 0)):
                                sys.exit("t < 0 salimos para analizar el problema")

                            epsilon1 = 0
                            epsilon2 = 0

                            epsilon1, epsilon2 = epsilonsevaluator(epsilon1,epsilon2,timepositiveR1, timenegativeR1, timepositiveR2,  timenegativeR2)

                            print(epsilon1, epsilon2)

                            zsimpl_value = zsimpl(p_vector, r_n_vector)
                            z_atlas_value = z_atlas(p_vector, r_n_vector, R1, R2, epsilon1, epsilon2)
                            print("zsimpl: ", zsimpl_value)
                            print("zcrist: ", abs(c_z[-1]))
                            print("z_atlas_value: ", z_atlas_value)

                            sys.exit("Salimos")

                            conversionmanual = p_conversion/mass_conversion
                            prev_n2= p_n / mass_n
                            prev_n = prev_n2*conversionmanual
                            
                            v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                            
                            # Dividimos la distancia entre la rapidez del NP
                            t_n = dist_n / v_n  # s
                            
                            t_n = t_n * (10 ** 9)  # ns

                            ic = 0

                        except KeyError:
                        
                            t_n = 0.0
                            ic = 1

                        #fin calculo t_n

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
                        
                        tof = t_ph + t_n
                            
                        prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
                        rel_tof = tof - prompt_tof
                       
                        z_origin = abs(c_z[-1])

                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))

                        sentence = ' '.join(line) + '\n'
                        new_hepmc.write(sentence)
                        
                    else:
                        
                        rel_tof = 0.0        
                        z_origin = 0.0
                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))
                        
                        sentence = ' '.join(line) + '\n'
                        new_hepmc.write(sentence)
                else:
                        
                        rel_tof = 0.0        
                        z_origin = 0.0
                        line.insert(13, str(rel_tof))
                        line.insert(13, str(z_origin))
                        
                        sentence = ' '.join(line) + '\n'
                        new_hepmc.write(sentence)
            else: 
                
                new_hepmc.write(sentence)
        
        else:
            new_hepmc.write(sentence)

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