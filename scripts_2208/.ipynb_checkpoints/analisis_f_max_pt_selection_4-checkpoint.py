from pathlib import Path
import json
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool
import numpy as np
from my_funcs import my_arctan
import eta_functions_R_abs
from collections import defaultdict


'''
This code does not make a hepmc file from scratch; it aims to only edit the extra lines of the original file.
'''


import time

# Record the start time
start_time = time.time()

def calculate_sqrt(value):
    try:
        if value < 0:
            raise ValueError("Input value is less than 0. Cannot compute the square root of a negative number.")
        result = np.sqrt(value)
        return result
    except ValueError as e:
        print(e)
        sys.exit(1)


def timepositive(pgamma, rn, Ri):
    #print("pgamma: ", pgamma)
    normpgamma = np.linalg.norm(pgamma)
    pgamma_xy = pgamma[:2]
    #print("pgamma_xy: ", pgamma_xy)
    pt_gamma = np.linalg.norm(pgamma_xy)
    
    #print("nrom pt gamma: ", pt_gamma)
    rx_n, ry_n = rn[:2]
    px_gamma, py_gamma = pgamma_xy

    #print("sqrt pt gamma: ", np.sqrt(px_gamma**2 + py_gamma**2))
    #print("px_gamma, py_gamma", px_gamma, py_gamma)
    
    term1 = -(rx_n*px_gamma + ry_n*py_gamma)

    term1_c = (rx_n*py_gamma - ry_n*px_gamma)

    #print("term1c: ", pow(term1_c,2))
    #print("den: ", pow(Ri*pt_gamma,2))
    value_pro = pow(term1_c/(Ri*pt_gamma),2)
    #print("pow: ", 1 - pow(term1_c/(Ri*pt_gamma),2))

    term2 = Ri*pt_gamma*calculate_sqrt(1-value_pro)
    totalterm = term1 + term2
    #pt_Squared = pow(pt_gamma,2)
    #print("div pow: ", normpgamma*totalterm/pow(pt_gamma,2))
    #print("div no pow: ", normpgamma*totalterm/pt_Squared)
    return normpgamma*totalterm/pow(pt_gamma,2)

def timenegative(pgamma, rn, Ri):
    normpgamma = np.linalg.norm(pgamma)
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    rx_n, ry_n = rn[:2]
    px_gamma, py_gamma = pgamma_xy
    
    term1 = -(rx_n*px_gamma + ry_n*py_gamma)

    term1_c = (rx_n*py_gamma - ry_n*px_gamma)

    value_pro = pow(term1_c/(Ri*pt_gamma),2)

    #print("pow: ", 1 - pow(term1_c/(Ri*pt_gamma),2))
    term2 = -Ri*pt_gamma*calculate_sqrt(1-value_pro)
    totalterm = term1 + term2
    return normpgamma*totalterm/pow(pt_gamma,2)

def epsilonsevaluator(epsilon1,epsilon2,timepositiveR1, timenegativeR1, timepositiveR2,  timenegativeR2):

    no_entro_a_ninguno = True
    if(timepositiveR1<timepositiveR2 and timepositiveR1 >= 0):
        epsilon1 = 1
        epsilon2 = 1
        no_entro_a_ninguno = False
    elif(timepositiveR1<timenegativeR2 and timepositiveR1>= 0):
        epsilon1 = 1
        epsilon2 = -1
        no_entro_a_ninguno = False
    elif(timenegativeR1<timepositiveR2 and timenegativeR1>= 0):
        epsilon1 = -1
        epsilon2 = 1
        no_entro_a_ninguno = False
    elif(timenegativeR1<timenegativeR2 and timenegativeR1>= 0):
        epsilon1 = -1
        epsilon2 = -1
        no_entro_a_ninguno = False

    if no_entro_a_ninguno:
        sys.exit("Error. No time option was chosen")

    return epsilon1, epsilon2

def zsimpl(pgamma, rn):
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    rx_n, ry_n, rz_n = rn[:]
    px_gamma, py_gamma = pgamma_xy
    pz_gamma = pgamma[-1]
    #print("pz fuera zsimp: ",pz_gamma)
    return (rz_n - ((pz_gamma/pow(pt_gamma,2))*(rx_n*px_gamma + ry_n*py_gamma)))

def z_atlas(pgamma, rn, R1, R2, epsilon1, epsilon2):
    pgamma_xy = pgamma[:2]
    pt_gamma = np.linalg.norm(pgamma_xy)
    #("pt_gamma", pt_gamma)
    #pt_gamma_s = pow(pt_gamma,2)
    #R1_s = pow(R1,2)
    #R2_s = pow(R2,2)
    rx_n, ry_n, rz_n = rn[:]
    px_gamma, py_gamma = pgamma_xy
    pz_gamma = pgamma[-1]
    term1_c = (rx_n*py_gamma - ry_n*px_gamma)
    denomi = pow(R1*pt_gamma,2)
    termaux = term1_c/denomi
    multiplier = (pz_gamma/pt_gamma)*((R1*R2)/(R2-R1))

    #print("term1c: ", pow(term1_c,2))
    #print("den: ", pow(R1*pt_gamma,2))
    #print("pow: ", pow(term1_c/(R1*pt_gamma),2))
    value_pro1 = pow(term1_c/(R1*pt_gamma),2)
    value_pro2 = pow(term1_c/(R2*pt_gamma),2)
    term1 = epsilon1*calculate_sqrt(1-value_pro1)
    term2 = -epsilon2*calculate_sqrt(1-value_pro2)
    z_simpl_v = zsimpl(pgamma, rn)
    z_atlas_v = z_simpl_v + multiplier*(term1+term2)
    return z_atlas_v


def functiontheta(timei, r_n, p_gamma):
    r_vector = r_n + timei*p_gamma
    #print("r_Vector: ",r_vector)
    vectorz = np.array([0, 0, 1])
    dot_product = np.dot(vectorz, r_vector)
    #print("dot_prod: ",dot_product)
    normr = np.linalg.norm(r_vector)
    #print("normr: ",normr)
    theta = np.arccos(dot_product/normr)

    #print("theta function 1: ", theta)
    if(theta < np.pi):
        return theta
    else:
        return theta - np.pi
"""
def functiontheta(timei, r_n, p_gamma):
    r_vector = r_n + timei*p_gamma
    #print(r_vector)
    r_z = r_vector[-1]
    normr = np.linalg.norm(r_vector)
    theta = np.arccos(r_z/normr)
    if(theta < np.pi):
        return theta
    else:
        return theta - np.pi
"""

def eta_function(theta):
    tange = np.tan(theta/2)
    #print("log e",np.log(2.71))
    return -np.log(tange)


def main(parameters):
    
    global t_n

    file_in, type = parameters
    
    #We extract the baseout and name the output file (full_op_...hepmc)
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'full_op_{base_out}.hepmc'

    #print(f'\nRUNNING: {base_out}') #Commented by JJP

    it = 0 # Iterator for unnecessary lines
    i = 0
    limit = 2

    it_start = 0
    batch = 5000
    corte_inf = it_start * batch
    corte_sup = corte_inf + batch * 99999
    final_part_active = True
    
    #We open the hepmc. One to read and the other to write.
    #df = open(file_in, 'r')
    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')
    #We are creating a hepmc from scratch
    
    newstring = f"etaforptmax{number_alpha56}.txt"
    f = open(newstring, 'w')

    
    sentences = ''
    
    data = dict()
    num = 0
    p_scaler = None
    d_scaler = None
    
    iterator_Event = 0
    
    t_n = None

    for sentence in hepmc:     

        line = sentence.split()

        if len(line) > 0:

            if line[0] == 'E':
                holder = {'v': dict(), 'a': [], 'n5': dict()}
                new_hepmc.write(sentence)
                #With the following, we make sure the dictionary is not empty (pt_dict is true if its not empty)
                if(iterator_Event >= 1 and pt_dict):
                    max_key = max(pt_dict.keys())
                    max_value_vector = pt_dict[max_key]
                    #print(f"The key with the highest value is: {max_key}")
                    #print(f"The vector associated with this key is: {max_value_vector}")
                    # Extract the first (and only) sublist

                    sublist = max_value_vector[0]

                    # Convert the sublist to a string with the desired format
                    result_string_yes_commas = ', '.join(map(str, sublist))

                    result_string = result_string_yes_commas.replace(',', '')

                    f.write(result_string + '\n')
                    #print("result string: ", result_string)
                pt_dict = defaultdict(list)
                iterator_Event = iterator_Event + 1

                if(iterator_Event%100 == 0):
                    print("Event number: ", iterator_Event)


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
            
            elif line[0] == 'V': #If the line is a vertex
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
                    pid = int(line[1]) #We make id an integer(its originally a string)
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
                    

                    if (realizar_analisis_cumple): #If the particle is a photon and its final (if its not negative)
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

                            #Variable definition to work the function
                            p_vector = np.array([px, py, pz])
                            r_n_vector = np.array([x, y, z])

                            

                            R1_meters = 1.500
                            R2_meters = 1.590
                            #We change to milimeters, because thats the stadard unit. Otherwise,
                            # we multiply d_scaler, which changes units to cm
                            R1 = R1_meters * 1000 * d_scaler
                            R2 = R2_meters * 1000 * d_scaler
                            #print("R1 unidades: ", R1)
                            #print("R2 unidades: ", R2)

                            #sys.exit("Salimos")

                            #print("pz fuera zsimp: ",pz)
                            timepositiveR1 = timepositive(p_vector, r_n_vector, R1)
                            timenegativeR1 = timenegative(p_vector, r_n_vector, R1)

                            timepositiveR2 = timepositive(p_vector, r_n_vector, R2)
                            timenegativeR2 = timenegative(p_vector, r_n_vector, R2)

                            #print(timepositiveR1, timenegativeR1, timepositiveR2, timenegativeR2)

                            if((timepositiveR1 < 0 and timenegativeR1 < 0) or (timepositiveR2 < 0 and timenegativeR2 < 0)):
                                sys.exit("t < 0 We exit to analyse the individual problem")

                            epsilon1 = 0
                            epsilon2 = 0

                            epsilon1, epsilon2 = epsilonsevaluator(epsilon1,epsilon2,timepositiveR1, timenegativeR1, timepositiveR2,  timenegativeR2)

                            #print("epsilon1, epsilon2: ",epsilon1, epsilon2)

                            #We define a real sign for time
                            if(epsilon1 ==1):
                                time1 = timepositiveR1
                            else:
                                time1 = timenegativeR1
                                print("case epsilon1 = -1")
                            
                            if(epsilon2 ==1):
                                time2 = timepositiveR2
                            else:
                                time2 = timenegativeR2
                                print("case epsilon2 = -1")

                            #print("time1: ", time1)
                            angletheta1 = functiontheta(time1, r_n_vector, p_vector)
                            #print("time2: ", time2)
                            angletheta2 = functiontheta(time2, r_n_vector, p_vector)
                            eta1 = eta_function(angletheta1)
                            eta2 = eta_function(angletheta2)
                            
                            #print("eta1, eta2 main: ", eta1,eta2)
                            R1, R2 = eta_functions_R_abs.eta_func_R_abs(eta1,eta2)*d_scaler

                            #print("R1 y R2: ",R1,R2)

                            zsimpl_value = abs(zsimpl(p_vector, r_n_vector))
                            z_atlas_value = abs(z_atlas(p_vector, r_n_vector, R1, R2, epsilon1, epsilon2))
                            zcrist = abs(c_z[-1])
                            #print("zsimpl: ", zsimpl_value)
                            #print("zcrist: ", abs(c_z[-1]))
                            #print("z_atlas_value: ", z_atlas_value)

                            deltaz = np.abs((zsimpl_value - z_atlas_value)/z_atlas_value)*100

                           
                            #print("values insdide loop: ",pt, R1,R2, eta1, eta2, zsimpl_value, z_atlas_value)
                            #sys.exit("salimos")
                            pt_dict[pt].append([R1,R2, eta1, eta2, zsimpl_value, zcrist, z_atlas_value])

                            #for pt, values in pt_dict.items():
                            #    print(f"pt: {pt}, values: {values}")
                            #sys.exit("salimos")
                            
                            
                            #hay varios casos que parecieran no tener mucho error
                           
                            #if(deltaz > 10 ** 4 and np.abs(eta1) < 1.4 and np.abs(eta2) < 1.4):
                            #    print("Caso anomalo")
                            #    print("zsimpl_value, z_atlas_value",zsimpl_value, z_atlas_value)
                            #    print("p_vector, z_atlas_value",p_vector, r_n_vector)
                            #    print("R1, R2",R1, R2)

                            #i = 0
                            #print("casei: " ,i)
                            #i = i + 1
                            #if(i==10):
                            #   sys.exit("Salimos")

                            conversionmanual = p_conversion/mass_conversion
                            prev_n2= p_n / mass_n
                            prev_n = prev_n2*conversionmanual
                            
                            v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                            
                            # We divide the distance over the HN speed
                            t_n = dist_n / v_n  # s
                            
                            t_n = t_n * (10 ** 9)  # ns

                            ic = 0

                        except KeyError:
                        
                            t_n = 0.0
                            ic = 1

                        #End of t_n calculation

                        #t_ph calculation
                        
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
number_alpha56 = None
ATLASdet_radius= 1.5 #ATLAS detector radius
ATLASdet_semilength = 3.512 #Half the lenght of ATLAS radius (meters) (z_atlas)

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
        #Again, we open hepmc files to overwrite them.
        for file_inx in sorted(glob.glob(f"/Collider/scripts_2208/data/raw/run_{typex}*{tevx}.hepmc"))[:]:
            allcases.append([file_inx, typex])

#print(allcases)
# Redefine with only components at indices 0, 3, and 6
selected_cases = [allcases[i] for i in [0, 3, 6]]

# Extract the number 4 from the fourth part of the filename
for case in selected_cases:
    path = case[0]
    # Split the path to get the file name
    filename = path.split('/')[-1]
    # Split the filename to extract the fourth part
    fourth_part = filename.split('_')[3]
    # Extract the number from the fourth part
    number = ''.join(filter(str.isdigit, fourth_part))
    if number:
        extracted_number = number
        break

# Print the result
print(selected_cases)
print(extracted_number)

number_alpha56 = extracted_number


if __name__ == '__main__':
    with Pool(1) as pool:
        #print(allcases[-1:])
        pool.map(main, selected_cases)

        
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

#f.close()