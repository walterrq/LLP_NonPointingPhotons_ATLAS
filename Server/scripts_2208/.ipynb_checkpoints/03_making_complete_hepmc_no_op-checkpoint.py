from pathlib import Path
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool

def main(parameters):
    
    #De los dataframes, en base al evento y al id, seleccionamos un registro y sacamos datos.
    
    #parameters, nuevamente, son el nombre del archivo y el tipo

    file_in, type = parameters
    
    #Extraemos el baseout y ponemos el nombre del archivo de salida (complete...hepmc)
    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'complete_{base_out}.hepmc'

    #print(f'\nRUNNING: {base_out}') #Commented by JJP
    
    #Abrimos el pickle(dataframe) que guardamos en el anterior
    
    #Podemos intentar dropear (borrar) linea por linea analizada para que la busqueda no demore tanto.
    #Esto lo hariamos alterando la variable new_observs despues de cada vez que analicemos las variables.
    #Tenemos que hacer un codigo que elimine al final del ultimo if elimina todos los registros del primer evento de new_observs
    new_observs = pd.read_pickle(f'/Collider/scripts_2208/data/clean/photon_df-{base_out}.pickle')

    #keys = new_observs[['Event','pid']].values.tolist() #NO SE USA
    
    #Antes del comando set_index tenemos un index generico (0,1,2,...)
    #Luego de hacer el set index seteamos las variables de Event y pid como un multiindex
    #El nombre de cada registro estara basado en el numero de su evento y el id del foton
    #Notamos que la etiqueta 0,1,2,3,etc de la primera columna original desaparece porque la estamos sobreescribiendo con el index de las columnas especificas evento e id
    new_observs = new_observs.set_index(['Event','pid'])
    
    print(new_observs)
    
    #Abrimos los hepmc. Uno para leer y otro para escribir
    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')

    event = -1
    sentences = ''
    #Leemos linea por linea del hepmc
    #Cuando se abre un archivo, el for lo interpreta como readline
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
                #if (event % 100) == 0:
                #    print(f'{base_out}: Event {event}')
                if (event % 1000) == 0:
                    new_hepmc.write(sentences)
                    sentences = ''
                #print(event)
            elif line[0] == 'P':
                pid = int(line[1]) #el id lo hacemos integer porque esta como string
                if (abs(int(line[2])) == 22) and (int(line[11]) == 0): #Si es que la particula es foton y es particula final (si no es negativo)
                    try: #Queremos ponerle un z origin y tiempo relativo al foton
                        #Se intentara encontrar el registro de estos obserevables en nuestra dataframe llamando a evento y pid
                        
                        #Multiindex: cuando se tiene mas de una columna como indice
                        #En ese caso, podemos buscarlo usando una tupla
                        #el .loc te devuelve todos los datos guardados (ptx,zoirigin,trelat,etc) en formato de serie
                        this = new_observs.loc[(event,pid)]
                        
                        #Este loc demora cuando tenemos varios eventos, pues tenemos 10000 eventos con 2 fotones por evento en promedio
                        
                        zorigin = this.z_origin
                        relt = this.rel_tof
                        #new_observs = new_observs.drop(index = (event,pid))
                    except KeyError: 
                        #Si no existe la combinacion de evento y pid(porque el foton no pasa los cortes), dar el valor de 0 a las variables
                        zorigin = 0.0
                        relt = 0.0
                        
                #Agregamos los valores anteriores a la linea donde estan nuestros nuevos datos
                #El primer input de insert (13) indicara el espacio de la lista donde se agrega
                #El segundo input implica lo que se esta añadiendo(relt o zorigin) en esta posicion 13
                #El segundo input empuja el primero. zorigin termina en la posicion 13 y relt en la posicion 14
                line.insert(13, str(relt))
                line.insert(13, str(zorigin))
                
                #Ojo que bastantes valores seran 0, pues solo, en promedio, hay 2 fotones de nuestro interes con los valores distintos de 0
                
                #Construimos nuevamente el sentence agregando el elemento separandolo con espacios en blanco y ponindo un newline.
                #El join junta los elementos de la lista en base al string (en este caso, el espacio '')
                #si usaramos el join con una x en lugar de un espacio en blanco tendriamos lo siguiente:
                    # [1,2] -> 1x2
                #Recordemos line = sentence.split()
                #Entonces, convertimos una lista en un string unido por espacios. Importante para mantener la estructura del hepmc
                sentence = ' '.join(line) + '\n'
        
        #En lugar de escribir linea por linea, que es pesado en memoria, juntamos linea por linea y luego, cuando la linea sea 1000 o 2000 lo escribimos todo en newhepmc y reseteamos sentences.
        #Mejor es escribir por banches cada 1000 o 2000 elementos
        sentences += sentence
    new_hepmc.write(sentences)
    hepmc.close()
    new_hepmc.close()
    return


destiny = "/Collider/scripts_2208/data/raw/"
types = ["TTH"] #por ahora solo queremos un punto
#types = ["ZH","WH","TTH"]
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

        
#Para revisar la efectividad del tiempo tenemos que hacer lo siguiente
#import time
#start = time.time()
#end = time.time()
#print(end-start)