import sys
import os
import pandas as pd
import glob
#Importamos root
import ROOT
import numpy as np
from multiprocessing import Pool

'''
We extract info to validate isolation parameters. This does not represent the final information 
the general code is working with. For an updated version check 03_extracting_root.py
'''


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
    branchFlowTrack = treeReader.UseBranch("EFlowTrackECAL")
    branchFlowPhoton = treeReader.UseBranch("EFlowPhoton")
    

    # Loop over all events
    photons = []
    jets = []
    leptons = []
    eftrack = []
    efphoton = []
    depo_ecal = []
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
                        
        for track in branchFlowTrack:
            eftrack.append({"N": entry, "eta":track.Eta, 'phi': track.Phi, 'E': track.PT})
            depo_ecal.append({"N": entry, "eta":track.Eta, 'phi': track.Phi, 'E': track.PT})
            
        for fph in branchFlowPhoton:
            efphoton.append({"N": entry, "eta":fph.Eta, 'phi': fph.Phi, 'E': fph.ET})
            depo_ecal.append({"N": entry, "eta":fph.Eta, 'phi': fph.Phi, 'E': fph.ET})

    #input_file.close()
    #Vaciamos de la memoria todo lo relacionado al chain
    chain.Clear()
    
    #Guardamos los datos en 3 distintos dataframes
    df = pd.DataFrame(photons)
    df_jets = pd.DataFrame(jets)
    df_leps = pd.DataFrame(leptons)
    df_eftrack = pd.DataFrame(eftrack)
    df_efphoton = pd.DataFrame(efphoton)
    df_ecal = pd.DataFrame(depo_ecal)

    
    #Si no hay particulas, hacemos un dataframe con el formato pero vacio, sin informacion
    if df.shape[0] == 0:
        df = pd.DataFrame(columns=["N", "E", "pt", "eta", 'phi', 'z_origin', 'rel_tof', 'MET'])
    if df_jets.shape[0] == 0:
        df_jets = pd.DataFrame(columns=["N", "pt", "eta", 'phi', 'MET'])
    if df_leps.shape[0] == 0:
        df_leps = pd.DataFrame(columns=["N", "pdg", "pt", "eta", 'phi','mass'])
    if df_eftrack.shape[0] == 0:
        df_eftrack = pd.DataFrame(columns=["N", "eta", 'phi', 'E'])
    if df_efphoton.shape[0] == 0:
        df_efphoton = pd.DataFrame(columns=["N", "eta", 'phi', 'E'])
    if df_ecal.shape[0] == 0:
        df_ecal = pd.DataFrame(columns=["N", "eta", 'phi', 'E'])
    
    
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
    
    #The following lists wont be sorted by momentum.
    df_eftrack = df_eftrack.sort_values(by=['N', 'eta'], ascending=[True, False])
    g = df_eftrack.groupby('N', as_index=False).cumcount()
    df_eftrack['id'] = g
    df_eftrack = df_eftrack.set_index(['N', 'id'])
    df_eftrack.to_pickle(out_file.replace('_photons', '_eftracks'))

    df_efphoton = df_efphoton.sort_values(by=['N', 'eta'], ascending=[True, False])
    g = df_efphoton.groupby('N', as_index=False).cumcount()
    df_efphoton['id'] = g
    df_efphoton = df_efphoton.set_index(['N', 'id'])
    df_efphoton.to_pickle(out_file.replace('_photons', '_efphotons'))

    df_ecal = df_ecal.sort_values(by=['N', 'eta'], ascending=[True, False])
    g = df_ecal.groupby('N', as_index=False).cumcount()
    df_ecal['id'] = g
    df_ecal = df_ecal.set_index(['N', 'id'])
    df_ecal.to_pickle(out_file.replace('_photons', '_ecals'))
    
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
