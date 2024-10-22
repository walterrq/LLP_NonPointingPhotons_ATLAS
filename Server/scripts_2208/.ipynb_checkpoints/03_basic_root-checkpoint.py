import sys
import os
import pandas as pd
import glob
#Importamos root
import ROOT
import numpy as np
from multiprocessing import Pool

#Este script extrae los datos del Root que bota Delphes
#Al final tendremos 3 dataframes: uno para leptones, otro para jets y otro para fotones

def main(input_file):
    
    #El output file tiene el mismo nombre del input file, pero cambiando la parte final de .root a _photons.pickle
    out_file = input_file.replace('.root','_photons.pickle')
    #out_file = out_file.replace(origin, destiny)

    #Create chain of root trees
    chain = ROOT.TChain("Delphes")
    chain.Add(input_file) #Add permite que se lea el root
    
    # Create object of class ExRootTreeReader
    treeReader = ROOT.ExRootTreeReader(chain) #treeReader lee el root
    numberOfEntries = treeReader.GetEntries() #da el numero de eventos que hay en este root en especifico
    
    #Definimos cada una de las ramas
    #Recordemos que los root tienen hojas y ramas
    #Esto guardamos la direccion en memoria
    met = treeReader.UseBranch("MissingET")
    branchPhoton = treeReader.UseBranch("Photon")
    branchJet = treeReader.UseBranch("Jet")
    branchElectron = treeReader.UseBranch("Electron")
    branchMuon = treeReader.UseBranch("Muon")
    branchECal = treeReader.UseBranch("EFlowTrackECAL")
    branchTrack = treeReader.UseBranch("Track")
    

    # Loop over all events
    photons = []
    jets = []
    leptons = []
    tracks = []
    ecals = []
    #print(f"Number of Entries: {numberOfEntries}") #Commented by JJP
    for entry in range(numberOfEntries):
        # Load selected branches with data from specified event
        treeReader.ReadEntry(entry)
        #Con este comando de ReadEntry automaticamente todas las branches dan la informacion de ese evento
        #Lo que hace es que, por ejemplo, cuando llamemos a Branch photon solo te de los fotones de ese evento, pues branchphoton tiene los fotones de todos los eventos.
        #Al correr el ReadEntry hace que cuando le hacemos el for, solo haga un loop sobre los fotones de ese evento en especifico.
        
        miss = met[0].MET 
        #con el comando anterior obtenemos el missing energy del evento en especifico ([0]->primer evento)
        #print(entry0)

        #print(branchPhoton, branchElectron, branchMuon)
        for ph in branchPhoton:
            photons.append({"N": entry, "E":ph.E, "pt":ph.PT, "eta":ph.Eta, 'phi': ph.Phi,
                                'z_origin': ph.ZOrigin, 'rel_tof': ph.RelativeT,'MET': miss})

        for jet in branchJet:        
            jets.append({"N": entry, "pt": jet.PT, "eta": jet.Eta, 'phi': jet.Phi})

        for e in branchElectron:
            leptons.append({"N": entry, 'pdg': 11, "pt":e.PT,
                                "eta":e.Eta, 'phi': e.Phi, 'mass': 0.000511})

        for mu in branchMuon:
            #El append genera una lista de diccionarios (para leptons, para jets, etc)
            leptons.append({"N": entry, 'pdg': 13, "pt": mu.PT,
                                "eta": mu.Eta, 'phi': mu.Phi, 'mass': 0.10566})
                        
        for track in branchTrack:
            tracks.append({"N": entry, "pt":track.PT, "eta":track.Eta, 'phi': track.Phi,})
            
        for ecal in branchECal:
            ecals.append({"N": entry, "pt":ecal.PT, "eta":ecal.Eta, 'phi': ecal.Phi,})


    #input_file.close()
    #Vaciamos de la memoria todo lo relacionado al chain
    chain.Clear()
    
    #Guardamos los datos en 3 distintos dataframes
    df = pd.DataFrame(photons)
    df_jets = pd.DataFrame(jets)
    df_leps = pd.DataFrame(leptons)
    df_tracks = pd.DataFrame(tracks)
    df_ecals = pd.DataFrame(ecals)
    
    #Si no hay particulas, hacemos un dataframe con el formato pero vacio, sin informacion
    if df.shape[0] == 0:
        df = pd.DataFrame(columns=["N", "E", "pt", "eta", 'phi', 'z_origin', 'rel_tof', 'MET'])
    if df_jets.shape[0] == 0:
        df_jets = pd.DataFrame(columns=["N", "pt", "eta", 'phi', 'MET'])
    if df_leps.shape[0] == 0:
        df_leps = pd.DataFrame(columns=["N", "pdg", "pt", "eta", 'phi','mass'])
    if df_tracks.shape[0] == 0:
        df_tracks = pd.DataFrame(columns=["N", "pt", "eta", 'phi', 'MET'])
    if df_ecals.shape[0] == 0:
        df_ecals = pd.DataFrame(columns=["N", "pt", "eta", 'phi', 'MET'])
        #print(df_jets)
    
    #El comando sort_values ordena los valores primero en base al numero de eventos, y luego, en base al pt (de mayor a menor(del mas energetico al menos energetico))
    df = df.sort_values(by=['N', 'pt'], ascending=[True, False])
    #El groupby le da la etiqueta
    #Al mas energetico le pone 0. Al segundo menos energetico le pone un id de 1
    #Generamos un id en base al pt. 
    
    #La razon porque hacemos el sortby con N y pt juntos es para que ordene los pts pero dentro de los que tienen el mismo N. 
    #El orden de prioridad es que primero ordene por N. Luego de eso, de entre los que tienen el mismo N, ordenamos el pt.
    
    #Tendremos entonces, que la anterior linea de codigo hace lo siguiente:
    #   N  pt
    #0  1  20
    #1  1  10
    #2  2  50
    #3  2  40
    #4  2  30
    
    g = df.groupby('N', as_index=False).cumcount()
    #df[id] guarda g en una columna
    df['id'] = g
    #Seteamos el index(un multiindex) para que esten dado en base al N y al id
    df = df.set_index(['N', 'id'])
    #print(f'{100 * df.index.unique(0).size / numberOfEntries:2f} %') #Commented by JJP
    df.to_pickle(out_file)

    df_jets = df_jets.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_jets.groupby('N', as_index=False).cumcount()
    df_jets['id'] = g
    df_jets = df_jets.set_index(['N', 'id'])
    df_jets.to_pickle(out_file.replace('_photons','_jets'))

    df_leps = df_leps.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_leps.groupby('N', as_index=False).cumcount()
    df_leps['id'] = g
    df_leps = df_leps.set_index(['N', 'id'])
    df_leps.to_pickle(out_file.replace('_photons', '_leptons'))
    
    #Probablemente se necesite calbiar para tener en cuenta el R min 2do !!
    df_tracks = df_tracks.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_tracks.groupby('N', as_index=False).cumcount()
    df_tracks['id'] = g
    df_tracks = df_tracks.set_index(['N', 'id'])
    df_tracks.to_pickle(out_file.replace('_photons', '_tracks'))
    
    df_ecals = df_ecals.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_ecals.groupby('N', as_index=False).cumcount()
    df_ecals['id'] = g
    df_ecals = df_ecals.set_index(['N', 'id'])
    df_ecals.to_pickle(out_file.replace('_photons', '_ecals'))
    
    #print(df) #Commented by JJP
    return


#Commented by JJP
#print('ROOT FIRST ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
#print('DELPHES CLASSES   :',ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"'))
#print('EXRROT TREE READER:',ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"'))

#Cargamos librerias necesarias para leer root (delphesclasses, exrootreader)
ROOT.gSystem.Load("libDelphes") #cargamos una libreria especifica
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"') #Interpreto una linea de c++ en python
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')


origin = f"/Collider/scripts_2208/data/clean/"
destiny = f"/Collider/scripts_2208/data/clean/"
#destiny = f"/Collider/2023_LLHN_CONCYTEC/"

types = ['ZH', "WH", "TTH"]
tevs = [13]

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(origin + f"*{typex}*{tevx}.root"))[:]:
            allcases.append(file_inx)

if __name__ == '__main__':
    #print(allcases) #Commented by JJP
    with Pool(1) as pool:
        pool.map(main, allcases)
