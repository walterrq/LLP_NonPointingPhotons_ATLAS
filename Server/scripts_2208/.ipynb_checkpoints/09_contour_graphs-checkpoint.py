import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from math import floor, ceil
import scipy.ndimage
import sys

#Con este codigo solo generamos la data. Luego se la podemos pasar a Mathematica para los graficos. 
#Mathematica grafica mejor que python.

#Sucede que para los contour graphs, matplotlib necesita puntos que esten separados por un mismo intervalo.
#No simpre se tiene este caso.

#Al final tendremos csv: formato que Mathematica puede leer.

k_factors = {'ZH':1.491,'WH':1.253,'TTH':1.15}

#Configuramos el deltas, porque ahora solo tenemos delta 15 y queremos variacion en alfas
deltas = ['01','15']

#COMENTADO MIO
#deltas = ['15']
#alphas = ['4','5','6']

#Los graficos seran del BR del neutrino pesado decayendo en foton y neutrino ligero

#Sabemos que tenemos la formula para hallar eventos reales:
##s=L*BR(H->n5 n6)*cross_sec*BR(n5->gamma nu)
#Recordemos:
#scale = scales[type + '_' + base_out] * 1000 * 0.2* 139 / n_events
#aca hemos asumido el BR(n5->gamma nu) =1

#tenemos valores estadisticos (el maximo que se puede obtener como para decir que viene de la señal que queremos)
#Hay una cota superior de que signo o cuantos eventos debes medir

#Queremos despejar el BR y ver cual seria el BR maximo para llegar a esta cota
#dado el numero de eventos que obtuvimos, cual es la cota del BR para que se llegue a esa cota maxima.
#el BR hasta ahora lo consideramos 1

#tenemos "s" del paper de ATLAS y L*BR(H->n5 n6)*cross_sec es el resultado de numero de evento que estamos teniendo,

events_up = {6.84578:['1'],3.84578:['2+'],6.84844:['1','2+']}
sigmas_up = {0.042:['1'],0.022:['2+'],0.041:['1','2+']}

#Los expected background (sigmas), salen del paper Search for displaced photons produced in exotic decays of the Higgs boson using 13 TeV pp collisions with the ATLAS detector.
#Esto representa el tope de eventos o sigmas maximos
#Hacemos esta comparacion con el sigma de -> BR * Lum * sigma = N
#Luego dividimos los sigmas
#Events es lo que extrapolamos en base a los sigmas (Lo hizo Lucia Duarte)

for delta in deltas[1:]:
    #origin = f"./data/matrices_{delta}GeV/" 
    origin = f"./data/matrices_{delta}/" 
    #Alteramos la direccion ahora
    
    #origin = f"./data/matrices_15/"
    destiny = f"./data/"
    names = list(sorted(glob.glob(origin + f"bin_*.json")))
    #for n_up, channels in events_up.items():
    print('0')
    print(names)
    for s_up, channels in sigmas_up.items():
        print('0.5')
        print(channels)
        ## Opening the files and assigning mass and alpha as tag
        values=[]
        for name in names[:]:
            #Con reg ex sacamos los siguientes datos del nombre del proceso (names)
            mass = float(re.search(f'/.*M(\d+,?\d+|\d+)_', name).group(1).replace(',','.'))
            alpha = float(re.search(f'/.*Alpha(\d+,?\d+|\d+)_', name).group(1).replace(',','.'))
            
            print("printeamos alpha")
            print(alpha)
            
            #We know alpha and have delta fixed
            proccess = re.search(f'/.*_13-(.*).json', name).group(1)
            #print(mass, alpha)
            with open(name, 'r') as file:
                info = json.load(file)
                #de info solo nos interesa el canal
                #queremos el ultimo bin de z, t y met
            print('0.75')
            print(info[ch])
            info = [np.asarray(info[ch])[-1,-1,-1] for ch in channels]
            print('1')
            print(info)
            #sys.exit()
            values.append([(mass,alpha),(proccess,sum(info))])
            #En sum, se esta sumando los eventos del ultimo bin de z, t con el met signal.
            #La suma es sobre los canales.

        # Grouping the list of values by same mass and alpha
        
        #Se esta creando un diccionario de la siguiente forma:
        # {(m,a): [("WH", 2.4), ("ZH", 1.2), ("TTH", 0.5)]}
        
        print('2')
        print(values)
        
        points = {}
        for value in values:
            #Se hace una lista vacia por default.
            points.setdefault(value[0], []).append(value[-1])
        print('3')
        print(points)
        
        
        '''
        Lo de abajo fue un codigo para los graficos. Como ya no hacemos estos en python, se puede comentar
        pre_params = [set(i for i,j in points.keys()), set(j for i,j in points.keys())]
        masses, alphas = [sorted(list(x)) for x in pre_params]
        print(masses, alphas)
        '''
        
        
        # Keeping only the ones that have three elements
        #Multiplicamos el kfactor por el numero de evento para cada proceso.
        #Hacemos un dic comprehension:
        #Para key y value, conserva solo los que tengan un name igual a 3. Es decir, que se tengan los tres procesos.
        #Lo hacemos como safeguard.
        #No tendremos una comparacion correcta de BR si es que una de la combinacion de las masas y alphas solo tienen TTH o solo TTZ WH, o solo WH ZH, etc
        #Se hace un analisis de todos los procesos
        #Necesitamos que hayan los 3 
        #Si no hay, descartamos la combinacion de masa y alfa. Descartamos ese punto.
        #Yo quiero 3 tipos de señales y si no lo tengo descarto esa punto del espacio de parametros.
        points = {key: [k_factors[proc[0]]*proc[1] for proc in val] for key, val in points.items() if len(val) == 3}
        #Comentamos por recomendacion de cristian
        #points = {key: val for key, val in points.items()}# if key[0] >2 and key[1] < 8}
        
        print('4')
        print(points)
        #sys.exit()
        # Sum all the channels
        points = {key: sum(val) for key, val in points.items()}

        # Get the branching ratio
        #COMENTAMOS EL NON_POINTS PORQUE SEGUN CRISTIAN NO ES IMPORTANTE
        #not_points = [key for key, val in points.items() if val == 0]
        #points = {key: 100 * n_up/(val/0.2) for key, val in points.items() if val > 0 }#and key[0]%1==0 and key[1]%1==0}
        points = {key: 100 * s_up/(val/(139 * 0.2)) for key, val in points.items() if val > 0 }#and key[0]%1==0 and key[1]%1==0}
        #Lo que obtenemos de 100 * s_up/(val/(139 * 0.2) es el BR correspondiente al de higgs a neutrinos pesados maximo para llegar a l verdadero.
        #*100 porque queremos en porcentaje
        #Solo hacemos si el numero de eventos es mayor a 0
        
        #Guardamos en una lista de listas
        data = [[*key,val] for key, val in points.items()]
        #[[m, a, 98.7], ...]    
        
        #pd.DataFrame(data).to_csv(destiny + f'datapoints_{alpha}GeV-{"_".join(channels)}-Sigma.dat',index=False)
        pd.DataFrame(data).to_csv(destiny + f'datapoints_{delta}GeV-{"_".join(channels)}-Sigma.dat',index=False)
        print('5')
        print(data)
        
        #Aca generamos la masa con alfa y el BR maximo
        
    
        
#sys.exit()
#
# print(f'points considered: {len(points)}')
# print(f'points not considered: {len(not_points)}')
#
# zoom = 1
# x,y = np.meshgrid(masses,alphas)
# z = np.full((len(alphas),len(masses)),max(points.values()))#max(points.values()))
# print(len(masses),len(alphas),z.shape)
#
# for xi, mass in enumerate(masses):
#     for yi, alpha in enumerate(alphas):
#         if (mass, alpha) in points:
#             val = points[(mass, alpha)]
#             #if val <= 500.:
#             if val > 0.:
#                 z[yi, xi] = val
#
# #print(z[9,1])
# #z = 10.**scipy.ndimage.zoom(np.log10(z),zoom)
# #print(z)
# color_min = floor(np.log10(min([x for x in points.values() if x > 0])))
# color_max = ceil(np.log10(max(points.values())))
#
# levels = 10. ** np.arange(color_min,color_max+1)
# #levels = 10. ** np.array([-1,0,1,2,5,19])
# plt.contourf(x,y,z,levels=levels,locator=ticker.LogLocator())
# plt.colorbar()
#
# not_px = [zoom*x for x,y in not_points]
# not_py = [zoom*y for x,y in not_points]
# px = [zoom*x for x,y in points.keys()]
# py = [zoom*y for x,y in points.keys()]
#
# #plt.scatter(px, py, color='pink')
# #plt.scatter(not_px, not_py, color='orange')
#
# plt.xlabel('MASS')
# plt.ylabel('ALPHA')
# plt.xlim(1,10)
# plt.ylim(1,10)
# plt.show()