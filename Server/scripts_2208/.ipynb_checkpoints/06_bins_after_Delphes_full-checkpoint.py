import sys
import numpy as np
import re
import glob
import pandas as pd
from scipy.interpolate import interp1d
from my_funcs import isolation
from pathlib import Path
import json
import sys
from multiprocessing import Pool
#import os

#Numero de eventos (recordamos que este archivo tiene un input, que es el # de eventos)
n_events = int(sys.argv[1])

#La eficiencia se maneja de la siguiente forma
#Supongamos que tenemos el foton con una eficiencia de deteccion de 0.85 y sacamos un random number
#si el random number es menor que 0.85 guardas el foton, si es mayor, lo descartas. 

#Importamos eficiencias obtenidas de los graficos de los papers de atlas
#les damos forma de funcion con set_axis
#En el caso del foton, se busca la eficiencia en terminos de z_origin. Mientras mas lejos, disminuye la eficiencia (no tanto)
ph_eff_zo = pd.read_csv(f'./data/z0_eff_points.csv', delimiter=',', header=None).set_axis(['zorigin','eff'], axis=1)
#la eficiencia del muon se hace en base a pt
mu_eff_pt = pd.read_csv('./data/muon_eff_pt.csv',header=0)
#el electron tiene eficiencia por pt y por eta
el_eff_pt = pd.read_csv('./data/electron_eff_pt.csv',header=0)
el_eff_eta = pd.read_csv('./data/electron_eff_eta.csv',header=0)
zorigin_res= pd.read_csv(f'./data/z0_res_points.csv', delimiter=',', header=None).set_axis(['zorigin','res'], axis=1)
reltof_res= pd.read_csv(f'./data/z0_res_points.csv', delimiter=',', header=None).set_axis(['zorigin','res'], axis=1)
cutflow_path = "./data/clean/cutflow/"

#Guardamos el crossection al ejecutar madgraph ubicado en el master
#Usamos la formula: #eventos generados = #eventos reales * luminosidad * cross section
scales = pd.read_csv("/Collider/scripts_2208/data/cross_section.dat",delimiter="\t",index_col=0,header=None,squeeze=True)

np.random.seed(0)

## For photon efficiency
#Experimentalmente, mientras aumenta el z_origin, baja la eficiencia
#hacemos una interpolacion en base al zorigin y la eficiencia
#no es una interpolacion lineal. 
#Solo usamos la interpolacion para que la funcion no tenga problemas si le pasamos un punto no definido en la data
#fill_value llena la informacion con el ultimo valor disponible
photon_eff_zo = interp1d(ph_eff_zo.zorigin, ph_eff_zo.eff, fill_value=tuple(ph_eff_zo.eff.iloc[[0,-1]]),
                        bounds_error=False)

#Hay un factor de normalizacion de eficiencia para cuando se utilizan 2.

## For muon efficiency
mu_func = interp1d(mu_eff_pt.pt,mu_eff_pt.eff, fill_value=tuple(mu_eff_pt.eff.iloc[[0,-1]]), bounds_error=False)
## For electron efciency
el_pt_func = interp1d(el_eff_pt.BinLeft,el_eff_pt.Efficiency, fill_value=tuple(el_eff_pt.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')
el_eta_func = interp1d(el_eff_eta.BinLeft,el_eff_eta.Efficiency, fill_value=tuple(el_eff_eta.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')
el_normal_factor = 1/0.85
## For comparing with the Z mass
m_Z = 91.1876 #GeV
Ecell_factor = 0.35 #factor experimental del deposito en el calorimetro 
#No usamos todo el et, si no mayor deposito de energia transversal en el calorimetro. Consideramos que es el 35% de la energia total
#Trabajamos no con la energia total, si no con la energia de la celda en donde se ha depositado mas energia
#La celda con mayor deposito de energia medira el 35% de la energia de la particula

## For photon's z origin resolution
#Interpolacion de resolucion
zorigin_res_func = interp1d(zorigin_res.zorigin, zorigin_res.res, fill_value=tuple(zorigin_res.res.iloc[[0,-1]]),
                        bounds_error=False)

## For photon's relative tof resolution
p0_h = 1.962
p1_h = 0.262

p0_m = 3.650
p1_m = 0.223

#Formula para la resolucion del tiempo relativo
#Se explica esta formula en la tesis
def t_res(ecell):
    if ecell >= 25:
        resol= np.sqrt((p0_m/ecell)**2 + p1_m**2)
    else:
        resol= min(np.sqrt((p0_h / ecell) ** 2 + p1_h ** 2), 0.57)

    return resol

def main(variables):

    type = variables[0]
    base_out = variables[1]

    #bin_matrix = dict()
    #for key, t_bin in t_bins.items():
    #    bin_matrix[key] = np.zeros((len(z_bins) - 1, len(t_bin) - 1, lpen(met_bins) - 1)).tolist()
    #print(bin_matrix)
    #sys.exit()
    
    #El bin_matrix contendra la matriz
    #En el json tendremos un diccionario de dos matrices 
    #El json tendra 2 entradas. Cada entrada con una matrix
    #{1:M3, 2+:M3} -> 2 entradas correspondientes a los canales de fotones
    #La matrix sera 3x3: z_origin, tgamma y MET
    bin_matrix = dict()
    for key, t_bin in t_bins.items(): 
        #El t_bins es un diccionario. Extraemos sus keys y sus values. El key es el channel
        #Recordemos: t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,12.1]}
        
        #Para cada canal haz una matriz de ceros con las dimensiones de la longitud de zbin, tbin y metbins.
        #Luego lo pasamos a lista pues json no recibe bien los arrays.
        #Recordemos que se crea una matriz de dimensiones iguales a la dimension de los arrays
        #Por ejemplo z_bins = [0,50,100,200,300,2000.1], entonces, la primera dimension sera 5.
        bin_matrix[key] = np.zeros((len(z_bins) - 1, len(t_bin) - 1, len(met_bins) - 1)).tolist()
        #bin_matrix[key] = np.zeros((len(z_bins) - 1, len(t_bin) - 1, len(met_bins) - 1))

    cutflows = dict()
    #en SCALE estamos multiplicando los eventos generados para obtener el numero de eventos experimentales (reales) que se detectarian
    #Este scale es un porcentaje del numero de eventos total. Por eso es importante el n_events
    #Liminosidad: 139
    #Para la validacion: 0.2 (branching ratio usado para la validacion. Dummy value)
    #Pasamos de picobarns a fentobarn con 1000, pues la unidad de luminosidad es en fentobarns^-1
    #scales -> Cross section
    scale = scales[type + '_' + base_out] * 1000 * 0.2 * 139 / n_events
    #print(n_events)
    #sys.exit()
    #with open(destiny + f'bin_matrices-{base_out}-{type}.1.txt', 'w') as file:
    #    file.write(scale)
    #with open(destiny + f'bin_matrices-{base_out}-{type}.2.txt', 'w') as file:
    #    file.write(n_events)

    #print(f'RUNNING: {base_out} - {type}')
    
    #Definimos el inputfile
    input_file = origin + f"complete_{type}_{base_out}_photons.pickle";
    photons = pd.read_pickle(input_file)
    leptons = pd.read_pickle(input_file.replace('photons', 'leptons')) #Lo mismo pero en lugar de photons dice leptones
    jets = pd.read_pickle(input_file.replace('photons', 'jets'))
    #print(photons)
    #sys.exit()
    
    #Si no tengo leptones o no tengo fotones no puedo hacer el analisis, entonces se termina el analisis.
    #esto es un lepton signal trigger
    if leptons.size == 0 or photons.size == 0:
        return
    #print(photons.shape[0])
    ### Aplying resolutions

    ## Z Origin
    #Aplicamos la resolucion experimental
    #Creamos una nueva columna generada en base a los datos de cada columna del photons.
    #Estamos agregando la incertidumbre. Le estamos dando una forma normal.
    #axis=1 -> eje de los rows. Los datos los extraemos de las filas
    photons['zo_smeared'] = \
        photons.apply(lambda row:
                      np.abs(row['z_origin'] + zorigin_res_func(row['z_origin']) * np.random.normal(0, 1)),
                    axis=1)

    ## relative time of flight
    #Recordemos que el relative tof se calcula con la energia de la celda. Por ello, multiplicamos el row de la energia 'E' por el Ecell_factor.
    #'rel_tof' -> nombre que le hemos dado al relative time of flight
    
    #----------------------------- ENTIENDO LOS TERMINOS PERO NO SE POR QUE SUMAN RELTOF O ZORIGIN CON RES PREGUNTAR!-----------------------------------
    
    #Hallamos el tiempo de resolucion en base a la energia, pero la energia se multiplica por el factor 0.35 recogido del detector.
    #eso lo sumamos al relative time of flight y lo normalizamos
    photons['rt_smeared'] = \
        photons.apply(lambda row: row['rel_tof'] + t_res(Ecell_factor * row['E']) * np.random.normal(0, 1), axis=1)

    ### Applying efficiencies

    ## leptons
    #Los leptones tienen 2 formas de eficiencia: para muones y electrones
    #Primero para los electrones:
    #En delphes, los leptons tiene la opcion loc que permite seleccionar solo los que cumplen con los datos dentro del parentesis (que el pdg sea 11).
    #Cambiaremos el valor de eff_value
    #Sobre los electrones aplicamos el apply sobre su columna.
    #Multiplicamos el normal factor(1/0.85) por la eficiencia obtenida del pt por la eficiencia del eta
    #IMPORTANTE: Recordemos que el eff_value es el threshold(0.85) bajo el que tiene que estar el random value para que se considere como detectado.
    #La informacion resultante de las operaciones que estamos haciendo sera guardada en la nueva columna con label 'eff_value'
    leptons.loc[(leptons.pdg==11),'eff_value'] = \
        leptons[leptons.pdg==11].apply(lambda row:
                                       el_normal_factor*el_pt_func(row.pt)*el_eta_func(row.eta), axis=1)
    #Para los muones hacemos lo mismo pero solo tenemos un tipo de eficiencia (pt)
    leptons.loc[(leptons.pdg == 13), 'eff_value'] = \
        leptons[leptons.pdg == 13].apply(lambda row: mu_func(row.pt), axis=1)
    
    #Creamos 'detected' como boolean (true o false)
    #Para el row 'eff_value' genera un random value (random_sample) y dime si es menor que el valor de la eficiencia o no.
    #Hay cierta aleatoriedad en lo que se va a detectar y lo que no
    leptons['detected'] = leptons.apply(lambda row: np.random.random_sample() < row['eff_value'], axis=1)
    
    #Me quedo solo con los valores verdaderos: aquellos en donde el random value es menor que la eficiencia
    #Conservamos los trues.
    #a leptons[] le estoy pasando la posicion en el dataframe de los detectados
    leptons = leptons[leptons.detected]

    ## photons
    #Analogamente en este caso con los fotones. Queremos que el numero random sea menor que la eficiencia calculada usando el zorigin
    photons['detected'] = \
        photons.apply(lambda row: np.random.random_sample() < photon_eff_zo(row['zo_smeared']), axis=1)
    # print(df[['zo_smeared','detected']])

    photons = photons[photons['detected']] #Esta ultima sintaxis es equivalente a  = photons[photons.detected]
    #print(photons.shape[0])
    #print(photons)
    #sys.exit()

    ## Overlapping
    
    #Filtro para confirmar que las particulas esten aisladas.
    
    ### Primero electrones
    #Quiero que no haya ningun foton dentro de un cono de radio 0.4 para los electrones
    #Quiero conservar electrones que esten aislados de fotones
    #Esto se traduce en conservar valores == 0
    leptons.loc[(leptons.pdg==11),'el_iso_ph'] = isolation(leptons[leptons.pdg==11],photons,'pt',same=False,dR=0.4)
    #print('1')
    #print(leptons)
    #print('2')
    #print(leptons.loc[(leptons.pdg==11),'el_iso_ph'])
    #Conservo los leptones que tengan el pt=0 sean muones
    leptons = leptons[(leptons.pdg==13)|(leptons['el_iso_ph']==0)]
    #print('3')
    #print(leptons)

    ## Luego jets
    #Quiero que en el cono de mi jet no haya electrones(0.2) ni fotones(0.4)
    jets['jet_iso_ph'] = isolation(jets,photons,'pt',same=False,dR=0.4)
    jets['jet_iso_e'] = isolation(jets, leptons[leptons.pdg==11], 'pt', same=False, dR=0.2)
    jets = jets[jets['jet_iso_e'] + jets['jet_iso_ph']==0]

    ## Electrones de nuevo
    leptons.loc[(leptons.pdg == 11), 'el_iso_j'] = isolation(leptons[leptons.pdg == 11], jets, 'pt', same=False,
                                                              dR=0.4)
    leptons = leptons[(leptons.pdg == 13) | (leptons['el_iso_j'] == 0)]

    ## Finalmente, muones
    jets['jet_iso_mu'] = isolation(jets, leptons[leptons.pdg == 13], 'pt', same=False, dR=0.01)
    jets = jets[jets['jet_iso_mu'] == 0]

    leptons.loc[(leptons.pdg == 13), 'mu_iso_j'] = isolation(leptons[leptons.pdg == 13], jets, 'pt', same=False,
                                                                 dR=0.4)
    leptons.loc[(leptons.pdg == 13), 'mu_iso_ph'] = isolation(leptons[leptons.pdg == 13], photons, 'pt', same=False,
                                                                 dR=0.4)
    leptons = leptons[(leptons.pdg == 11) | ((leptons['mu_iso_j'] + leptons['mu_iso_ph']) == 0)]
    
    #Comenzamos con los trigger
    
    ##### De ahí leptones con pt > 27
    #Pedimos que haya al menos un electron o muon con pt mayor a 27 GeV
    leptons = leptons[leptons.pt > 27]
    #Este comando borra filas que no tengan pt mayor a 27 GeV
    #print('4')
    #print(leptons)
    
    #Como segundo trigger, queremos eliminar eventos con diferencia entre masa invariante y masa del electron sea menor a 15 GeV
    ### Invariant mass
    #Me quedo con el foton mas energetico
    #Calculo px, py y pz en base al pt y phi. Esto porque root no brinda estos valores como tal
    photons0 = photons.groupby(['N']).nth(0)
    photons0['px'] = photons0.pt * np.cos(photons0.phi)
    photons0['py'] = photons0.pt * np.sin(photons0.phi)
    photons0['pz'] = photons0.pt / np.tan(2 * np.arctan(np.exp(photons0.eta)))
    photons0 = photons0[['E', 'px', 'py', 'pz']]

    leptons0 = leptons.groupby(['N']).nth(0)
    leptons0['px'] = leptons0.pt * np.cos(leptons0.phi)
    leptons0['py'] = leptons0.pt * np.sin(leptons0.phi)
    leptons0['pz'] = leptons0.pt / np.tan(2 * np.arctan(np.exp(leptons0.eta)))
    leptons0['E'] = np.sqrt(leptons0.mass**2 + leptons0.pt**2 + leptons0.pz**2)
    leptons0 = leptons0[['E','px','py','pz','pdg']]
    
    #Estamos haciendo un join de dos dataframes. Nos quedamos con la interseccion de los dataframes.
    #Solo queremos conservar los eventos que tienen tanto fotones como leptones
    #Ambos solo tendrian un indice: N -> de los eventos
    #print('6')
    #print(photons0)
    #print('7')
    #print(leptons0)
    final_particles = photons0.join(leptons0,how='inner',lsuffix='_ph',rsuffix='_l')
    #Hacemos una nueva columna 'M_eg', que es la masa invariante entre electrones y gammas.
    #Si el lepton es un muon, esto no cuenta.
    
    #print('8')
    #print(final_particles)
    
    final_particles['M_eg'] = np.sqrt((final_particles.E_ph + final_particles.E_l) ** 2 -
                    ((final_particles.px_ph + final_particles.px_l) ** 2 +
                     (final_particles.py_ph + final_particles.py_l) ** 2 +
                         (final_particles.pz_ph + final_particles.pz_l) ** 2))
    
    #Me quedo eventos que o su lepton es un muon o, si es un electron, que la masa invariante este el electron(recientemente hallada) y el gamma es mayor a 15 GeV
    final_particles = final_particles[(final_particles.pdg == 13) | (np.abs(final_particles.M_eg - m_Z) > 15)]
    
    #Nos quedamos con los eventos que sobrevivieron a todo lo anterior.
    photons = photons[photons.index.get_level_values(0).isin(
            list(final_particles.index.get_level_values(0)))]
    
    print('9')
    print(photons)

    ### CLaasifying in channels
    #Dividimos el dataframe de fotones. Separamos por canales. Canal 1: tamaño del evento 1 foton. Canal 2: los que son mas de un foton
    ph_num = photons.groupby(['N']).size() #busco el size del numero de fotones por evento
    dfs = {'1': photons.loc[ph_num[ph_num == 1].index], '2+': photons.loc[ph_num[ph_num > 1].index]}
    
    print(dfs)

    #print(scale)
    #with open(destiny + f'bin_matrices-{base_out}-{type}.txt', 'w') as file:
    #    file.write(scale)
    
    #Recordemos que Ecell se define como el 35% de la energia del foton
    for channel, phs in dfs.items():
        ## Keeping the most energetic photon (si hay mas de uno)
        phs = phs.groupby(['N']).nth(0)
        ## Filtering Ecell, zorigin and reltof
        #Aplicamos el corte de que el Ecell debe ser mayor a 10 GeV
        phs = phs[(Ecell_factor * phs['E']) > 10]
        #Recordemos que analizamos el modulo de z origin. Queremos que este de 0 a 2000
        phs = phs[phs['zo_smeared'] < 2000]
        #t_gamma debe ser mayor a cero y menor que 12 nanosegundos
        phs = phs[(0 < phs['rt_smeared']) & (phs['rt_smeared'] < 12)]
        
        ## Classifying in bins
        #El np.digitize recibe como input el valor y los bines. Si recibe un valor entre el bin n y n+1, lo pone el en bin en el que n estaria.
        #Le resto -1 a cada uno porque luego lo pondre en matrix. El binado empieza en 1, pero las matrices estan indexadas en 0 asi que le resto 1.
        phs['z_binned'] = np.digitize(phs['zo_smeared'], z_bins) - 1
        phs['t_binned'] = np.digitize(phs['rt_smeared'], t_bins[channel]) - 1
        phs['met_binned'] = np.digitize(phs['MET'], met_bins) - 1
        #print(phs[phs['z_binned'] == 5])
        ixs, tallies = np.unique(phs[['z_binned','t_binned','met_binned']].values,
                        return_counts=True, axis=0)
        #Numpy.unique-> Find the unique elements of an array.
        #If return_counts True, also return the number of times each unique item appears in ar.
        
        #Busco las combinaciones que hay y busco cuantas veces aparece esa combinacion,
        #Para esos casos, lo sumo a su bin correspondiente.
        
        print('5')
        #Se asigna al binado correspondiente
        print(ixs)
        #Si le hago un print a ixs da una lista de 3 elementos: el primero es un bin en z. El segundo el bin en t y el tercero el bin en met
        #Hay vario 0 presentes en el array

            
        #print(ixs, tallies)
        #Recordamos que bin_matrix es un diccionario con 2 keys. Cada key tiene una matrix. 
        #Primero accedemos al key correspondiente.
        #Los tres ultimos corchetes son el indice de la matrix.
        #key -> channel. Matrix-> z,t,met
        
        print('5')
        print(bin_matrix)
        
        for ix, tally in zip(ixs, tallies):
            z, t, met = ix
            bin_matrix[channel][z][t][met] += tally * scale
            #print(scale)
            
        print('6')
        print(bin_matrix)

        #np.save(destiny + f'bin_matrices-{base_out}-{type}-ch{channel}.npy', bin_matrix[channel])

    #bin_matrix = {k: v.tolist() for k, v in bin_matrix.items()}
    
    #Guardamos output en el destiny 
    with open(destiny + f'bin_matrices-{base_out}-{type}.json', 'w') as file:
        json.dump(bin_matrix, file)
    #print(bin_matrix)
    #os.system(f'echo {bin_matrix}')
    #print('Matrix saved!')

    #with open(destiny + f'bin_matrices-{base_out}.json', 'w') as file:
    #    json.dump(bin_matrix, file)
    return


# For bin classification
#z esta en milimetros
z_bins = [0,50,100,200,300,2000.1]
#Recordemos que el analisis se dividio en 2 canales: canal con 1 foton y canal con 2 a mas fotones
#t_gamma esta en nanosegundos
t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,12.1]}
#de 0 a 30, de 30 a 50, de 50 hasta arriba
#met en GeV
met_bins = [0, 30, 50, np.inf]

origin = f"/Collider/scripts_2208/data/clean/"
#origin = "/Collider/2023_LLHN_CONCYTEC/"
destiny = f"./data/matrices_15/"
types = ['ZH', 'WH', 'TTH']
tevs = [13]

#creamos el destiny folder con mkdir
Path(destiny).mkdir(exist_ok=True, parents=True)

#bases es una acumulacion de newcases
#newcases toma 3 valores diferentes: ZH, WH, TTH
#newcases es una lista con 3 elementos. Hay 3 variaciones del newcases: primero toma los ZH, el segundo los WH y el tercero los TTH
#bases junta todo estas 3 listas

#glob.glob -> Permite iterar a lo largo de todos los archivos, en cierto folder, que le des como input y los vuelve en lista

bases = []
for xx in types:
    files_in = glob.glob(origin + f"complete_{xx}*photons.pickle")
    #Recordemos que la estructura de los pickles antes creados son:
    #complete_TTH_M3_Alpha1_13_photons.pickle
    #print(files_in)
    newcases=sorted([[xx, re.search(f'/.*{xx}_(.+)_photons', x).group(1)] for x in files_in])
    #print(newcases)
    
    #The print of newcases is the following:
    #[['ZH', 'M3_Alpha1_13'], ['ZH', 'M3_Alpha2_13']]
    #[['WH', 'M3_Alpha1_13'], ['WH', 'M3_Alpha2_13']]
    #[['TTH', 'M3_Alpha1_13'], ['TTH', 'M3_Alpha2_13']]

    bases.extend(newcases)
#print(bases)

if __name__ == '__main__':
    with Pool(1) as pool:
        pool.map(main, bases)
