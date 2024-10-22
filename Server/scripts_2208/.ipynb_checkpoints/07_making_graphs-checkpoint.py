import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import numpy as np

origin = f"./data/matrices/"
destiny = f"./cases/"

#La unica diferencia entre el numero de casos.El 07 normal es cuando solo hay un caso. 
#Por ejemplo, podemos tener una separacion de masa de n5 y n6 de 1GeV. Luego, con una separacion de masa de 15GeV.
#Esto iba a hacer que se generaran matrices tanto como para 1GeV como para 15GeV.
#Por esto, en 07a tenemos mas de una carpeta de matrices.
#Tengo 2 grupos.
#Cada delta de masas generara distintos json. Las masas y alfas seran los mismos.

#En este script no necesito hacer pool. No necesito paralelizacion.

met_labels = ['BKG', 'CR', 'SR']
vround = np.vectorize(round)
#round es una funcion de python:
#x = round(5.76543, 2)
#print(x) -> 5.77
#Lo que hacemos con np.vectorize es que la funcion round pueda trabajar con vector, pues por default solo lo hace con escalares

colores = {'60':'r','50':'g','40':'b','30':'m'}
#definimos nuestro plot(axis): 5 filas(# de filas en el delta z), #columnas 2(solo una columna para cada canal)
#Recordemos que habra una columna con el channel 1, otra con el +2. En cada columna habra 5 filas de graficos. 10 graficos en total.
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
#Ajustamos los espacios del ploteo
plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)

#Hacemos un glob para ver todos los json que hay en el origin.
#glob.glob -> Permite iterar a lo largo de todos los archivos, en cierto folder, que le des como input y los vuelve en lista
for input_file in list(reversed(sorted(glob.glob(origin + f"bin_*.json"))))[:]:
    
    #Matplotlib te genera sus propios axis por default.
    #Nosotros queremos comparar histogramas necesitamos que los axis esten en la misma escala.
    ymax_p = []
    ymin_p = []
    
    #el base_out contiene, por ejemplo: M9_Alpha4_13-ZH
    base_out = re.search(f'/.*bin_matrices-(.+)\.json', input_file).group(1)
    print(base_out)
    
    #Abrimos el archivo con open. 
    #Transformamos las listas (nuevamente) a arrays, pues json no soporta numpy arrays. 
    with open(input_file, 'r') as file:
        burrito = json.load(file)
    
    #Recoramos que los keys son los canales. 1 o 2+
    burrito = {key: np.asarray(matrix) for key, matrix in burrito.items()}

    for key, matrix in burrito.items():
        #definimos el numero de bins. Definimos los limites de nuestros bines.
        #matrix.shape[1] son los bines del t_gamma.
        #+1 en un range para que vaya de 0 a un numero mas. 
        #Luego, el 0.5 permite que tengamos los siguientes bins
        #|0.5|----|1.5|----|2.5|----|3.5|
        #Esto lo hacemos para lograr que los bins esten centrado en 1, 2 y 3.
        nbins = np.array(range(matrix.shape[1] + 1)) + 0.5
        #ix-> columna de los graficos
        ix = int(key[0]) - 1
        #ir representa el row de la matriz
        ir = 0
        print('1')
        print(matrix[ir][:,-1])
        print('2')
        print(nbins)
        #sys.exit()
        
        
        #Recordamos la definicion de axs:
        #fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
        #axs es un grid de 10 canvases para hacer los graficos
        #Como se considera un array de numpy, al hacer el for row en axis pasamos por todos los rows.
        #El ix fija el analisis en una columna: 1 o 2+
        #como ix = int(key[0]) - 1, el canal 1 corresponde a la columna 0; el canal 2, a la columna 1
        for row in axs:
            row[ix].hist(nbins[:-1],
                         bins=nbins, weights=matrix[ir][:,-1], histtype='step',
                         stacked=False, label=base_out, color=colores[re.search(f'MN(.+)_ML', base_out).group(1)])
            row[ix].set_yscale('log')
            row[ix].set_xticks(np.array(range(matrix.shape[1])) + 1)
            row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
            row[ix].legend()
            row[ix].secondary_yaxis("right")
            ymax_p.append(row[ix].get_ylim()[1])
            ymin_p.append(row[ix].get_ylim()[0])
            ir += 1

plt.setp(axs, ylim=(10**-2, 2*10**6))
#plt.suptitle(f'{base_out} GeV\n{events} events')
#plt.show()
fig.savefig(destiny + f'validation_graphs.png')
plt.close()