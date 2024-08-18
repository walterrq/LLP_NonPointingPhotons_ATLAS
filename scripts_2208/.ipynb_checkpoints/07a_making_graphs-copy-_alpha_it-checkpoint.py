import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import numpy as np

#Este se usa mas.

destiny = f"./data/"

#tenemos dos carpetas distintas para depositar los json. 
#Por ahora, solo haremos ploteos con delta = 15. Por ello, comentamos la linea de abajo y la editamos
#deltas = ['01','15']
deltas = ['15']
met_labels = ['BKG', 'CR', 'SR']
#El vround no se usa. Se podria eliminar
vround = np.vectorize(round)
#round es una funcion de python:
#x = round(5.76543, 2)
#print(x) -> 5.77
#Lo que hacemos con np.vectorize es que la funcion round pueda trabajar con vector, pues por default solo lo hace con escalares
colores = {'60':'r','50':'g','40':'b','30':'m'}
#PREGUNTARLE AL PROFESOR JONES QUE ES EL KFACTOR
#En principio, es un valor experimental. Es un valor como para normalizar el numero de eventos. Cambia en base al proceso.
k_factors = {'ZH':1.491,'WH':1.253,'TTH':1.15}

for delta in deltas[:]:
    #Ahora la matriz depende del delta
    origin = f"./data/matrices_{delta}/"
    #Generamos el grid de canvas(vacio por ahora) con plt.subplots
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
    
    #Por ahora no estamos graficando todas las masas.
    #Por ahora solo graficamos 2: masa 5 y 8 y Alpha5.
    #reverse es porque queremos que esten en cierto orden, pero no es tan importante en este caso.
    
    #En nuestro caso, con nuestros json, varia el alpha. 
    #Tambien podriamos alterar los benchmarks y editar los archivos
    #El camino largo seria ejecutar las simulaciones y cambiar los param_cards para dejar el alpha 5 y cambiamos las masas.
    #Camino corto: editar el codigo y que se ejecute el for en terminos de los alphas.
    
    for alpha in ['4','5','6']:
        burrito = []
        
        #lo que hacemos a continuacion es pasar por todos los json de una masa y un alpha pero con distintos procesos (WH,ZH,TTH)
        #Nuestro resultado final sera la suma de todos estos procesos
        #Esto es porque los resultados experimentales verdaderos no diferencian entre estos tipos de procesos. tenemos la suma de todo
        
        input_files = list(reversed(sorted(glob.glob(origin + f"bin_*M9_*Alpha{alpha}_*.json"))))
        for input_file in input_files:
            process = re.search(f'/.*_13-(.*).json', input_file).group(1)
            with open(input_file, 'r') as file:
                cofre = json.load(file)
            cofre = {key: k_factors[process]*np.asarray(matrix) for key, matrix in cofre.items()}
            if burrito == []:
                burrito = cofre
            else:
                burrito = {key: burrito[key] + cofre[key] for key in cofre.keys()}
            #print(burrito)
        #sys.exit()
        norm = sum([x[:,:,-1].sum() for x in burrito.values()])
        burrito = {key: value[:,:,-1]/norm for key, value in burrito.items()}
        #print(sum([x.sum() for x in burrito.values()]))
        #sys.exit()

        ymax_p = []
        ymin_p = []

        for key, matrix in burrito.items():
            #definimos el numero de bins. Definimos los limites de nuestros bines.
            #matrix.shape[1] son los bines del t_gamma.
            #+1 en un range para que vaya de 0 a un numero mas. 
            #Luego, el 0.5 permite que tengamos los siguientes bins
            #|0.5|----|1.5|----|2.5|----|3.5|
            #Esto lo hacemos para lograr que los bins esten centrado en 1, 2 y 3.
            nbins = np.array(range(matrix.shape[1] + 1)) + 0.5
            ix = int(key[0]) - 1
            ir = 0
            #print(nbins)
            #print(matrix[ir][:,-1])
            #sys.exit()
            for row in axs:
                row[ix].hist(nbins[:-1],
                             bins=nbins, weights=matrix[ir], histtype='step', label=f'Alpha {alpha}')
                row[ix].set_yscale('log')
                row[ix].set_xticks(np.array(range(matrix.shape[1])) + 1)
                row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                row[ix].legend()
                row[ix].secondary_yaxis("right")
                ymax_p.append(row[ix].get_ylim()[1])
                ymin_p.append(row[ix].get_ylim()[0])
                ir += 1
                
    #aca en las axis debo poner como limites el mayor valor de la lista ymax_p y el menor valor de la lista ymin_p
    #usando esa lista, busco su max y min y seteo el bin
    
    #Advertencia: siempre puede que salgan numeros = 0 y no se puede hacer el log con ese valor. No hay problema.

    plt.setp(axs, ylim=(10**(-4),0.4))
    plt.suptitle(f'Mass 9 - Delta {delta}')
    #plt.show()
    fig.savefig(destiny + f'validation_graphs-Delta{delta}.png')
    plt.close()