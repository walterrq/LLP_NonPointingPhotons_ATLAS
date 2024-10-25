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

    print("Estamos generando: ", out_file)
    #print("Entramos al main")
    # Create chain of root trees
    chain = ROOT.TChain("Delphes")
    chain.Add(input_file) #Add permite que se lea el root
    
    # Create object of class ExRootTreeReader
    treeReader = ROOT.ExRootTreeReader(chain) #treeReader lee el root
    numberOfEntries = treeReader.GetEntries() #da el numero de eventos que hay en este root en especifico
    
    #Definimos cada una de las ramas
    #Recordemos que los root tienen hojas y ramas
    #Esto guardamos la direccion en memoria

    """
    met = treeReader.UseBranch("MissingET")
    branchPhoton = treeReader.UseBranch("Photon")
    branchJet = treeReader.UseBranch("Jet")
    branchElectron = treeReader.UseBranch("Electron")
    branchMuon = treeReader.UseBranch("Muon")
    branchFlowTrack = treeReader.UseBranch("EFlowTrackECAL")
    branchFlowPhoton = treeReader.UseBranch("EFlowPhoton")
    """


    branchElectron = treeReader.UseBranch("Electron")

    branchFlowTrackE = treeReader.UseBranch("EFlowTrackECAL")
    
    # HCal energy flow tracks (hadronic calorimeter tracks)
    branchFlowTrackH = treeReader.UseBranch("EFlowTrack")

    # ECal photons (energy flow photons in the electromagnetic calorimeter)
    branchFlowPhoton = treeReader.UseBranch("EFlowPhoton")

    # HCal neutral hadrons (hadronic calorimeter neutral hadrons)
    branchFlowNeutralHadron = treeReader.UseBranch("EFlowNeutralHadron")

    # Jets (reconstructed jets)
    branchJet = treeReader.UseBranch("Jet")

    # Photons after isolation (reconstructed isolated photons)
    branchPhoton = treeReader.UseBranch("Photon")

    # Muons after efficiency cuts (reconstructed muons)
    branchMuon = treeReader.UseBranch("Muon")

    # Missing transverse energy (MET)
    met = treeReader.UseBranch("MissingET")

    # Loop over all events
    photons = []
    jets = []
    leptons = []
    tracks = []
    towers = []

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
        if(entry%100 == 0):
            print("We are in the event: ", entry)

        #We create tracks dataframes

        for etrack in branchFlowTrackE:
            #We need the tracks in order to do the muon isolation.
            #We only save information regarding tracks from ecal for muons and electrons
            if abs(int(etrack.PID)) == 13 or abs(int(etrack.PID)) == 11:
                # print("printeamos el pid: deberia ser 11 o 13")
                # print(f'N: {entry}, pdg: {etrack.PID}, pt: {etrack.PT}, eta: {etrack.Eta}, phi: {etrack.Phi}')
                # print("tracks \n")
                # print(tracks)
                # print(type(etrack.PID))
                tracks.append({"N": entry, 'pdg': etrack.PID, "pt": etrack.PT,
                                "eta": etrack.Eta, 'phi': etrack.Phi}) #We consider particles and antiparticles
        # print("tracks \n")
        # print(tracks)
        for htrack in branchFlowTrackH:
            if abs(int(htrack.PID)) != 13 and abs(int(htrack.PID)) != 11: #we do not consider tau, because they decay quickly 
                # print("printeamos el pid: NO deberia ser 11 o 13")
                # print(etrack.PID)
                tracks.append({"N": entry, 'pdg': htrack.PID, "pt": htrack.PT,
                                "eta": htrack.Eta, 'phi': htrack.Phi}) #We consider particles and antiparticles

        #We create towers dataframes
            
        for etower in branchFlowPhoton:
            #I dont impose a requirement for now for towers
            towers.append({"N": entry, "et": etower.ET, "eta": etower.Eta, 'phi': etower.Phi}) 

        for htower in branchFlowNeutralHadron:
            #I dont impose a requirement for now for towers
            towers.append({"N": entry, "et": htower.ET, "eta": htower.Eta, 'phi': htower.Phi})
        

        # for muon in branchMuon:
        #     if muon.PT > 25:

        for jet in branchJet:
            #Condiciones para jets
            #print("Entramos al jet")
            if jet.PT > 25:
                #Formula del rapidity en base a los atributos que si tenemos
                y = np.log((jet.PT * np.sinh(jet.Eta) + np.sqrt(jet.Mass**2 +
                    (jet.PT * np.cosh(jet.Eta))**2)) / (np.sqrt(jet.Mass**2 + jet.PT**2)))
                if abs(y) < 4.4: 
                    
                    ConstList = 0
                    
                    if jet.Constituents.GetEntries() == 1:

                        obj = jet.Constituents.At(0)                        
                        strings=str(jet.Constituents.At(0))
                        #print(strings)
                        if strings == "Name: Muon Title: ":
                            #print("Tenemos un muon")
                            ConstList = 13
                        else:
                            ConstList = 0
                    else:
                        ConstList = 1           
                    jets.append({"N": entry, "pt": jet.PT, "eta": jet.Eta, 'phi': jet.Phi, 'Constituents': ConstList})
            #print(jets)


        for ph in branchPhoton:
            #print(ph.PT, ph.Eta)
            #Ponemos condiciones para que este dentro del barrel o en los endcaps
            if ph.PT > 10 and (abs(ph.Eta) < 1.37 or 1.52 < abs(ph.Eta) < 2.37): 
                #print(ph.Eta)
                #Si se cumplen las condiciones anteriores, guarda en la lista de fotones en diccionarios
                photons.append({"N": entry, "E":ph.E, "pt":ph.PT, "eta":ph.Eta, 'phi': ph.Phi,
                                'z_origin': ph.ZOrigin, 'rel_tof': ph.RelativeT,'MET': miss})
        
        for e in branchElectron:
            #Condiciones para el electron
            if e.PT > 10 and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.47):
                leptons.append({"N": entry, 'pdg': 11, "pt":e.PT,
                                "eta":e.Eta, 'phi': e.Phi, 'mass': 0.000511})

        for mu in branchMuon:
            #Condiciones para el muon
            #El append genera una lista de diccionarios (para leptons, para jets, etc)
            if mu.PT > 10 and abs(mu.Eta) < 2.7:
                leptons.append({"N": entry, 'pdg': 13, "pt": mu.PT,
                                "eta": mu.Eta, 'phi': mu.Phi, 'mass': 0.10566})
   #   N  pt
    #0  1  20
    #1  1  10
    #2  2  50
    #3  2  40
    #4  2  30
    

    
    #input_file.close()
    #Vaciamos de la memoria todo lo relacionado al chain
    chain.Clear()
    
    #Guardamos los datos en 3 distintos dataframes
    df = pd.DataFrame(photons)
    df_jets = pd.DataFrame(jets)
    df_leps = pd.DataFrame(leptons)
    df_tracks = pd.DataFrame(tracks)
    df_towers = pd.DataFrame(towers)
    
    #Si el dataframe de fotones o leptones tiene 0 elementos, entonces no podemos hacer analiis sobre ese evento.
    #Si sucede esto, imprimimos el shape y salimos.
    #Si no hay jets, hacemos un dataframe con el formato pero vacio, sin informacion
    if (df.shape[0] == 0) or (df_leps.shape[0] == 0):
        print(df.shape,df_leps.shape)
        return
    
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

    print("Generando pickle fotones")
    df.to_pickle(out_file)

    print("Generando pickle jets")
    df_jets = df_jets.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_jets.groupby('N', as_index=False).cumcount()
    df_jets['id'] = g
    df_jets = df_jets.set_index(['N', 'id'])
    df_jets.to_pickle(out_file.replace('_photons','_jets'))

    print("Generando pickle leptones")
    df_leps = df_leps.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_leps.groupby('N', as_index=False).cumcount()
    df_leps['id'] = g
    df_leps = df_leps.set_index(['N', 'id'])
    df_leps.to_pickle(out_file.replace('_photons', '_leptons'))

    print("Generando pickle tracks")
    df_tracks = df_tracks.sort_values(by=['N', 'pt'], ascending=[True, False])
    g = df_tracks.groupby('N', as_index=False).cumcount()
    df_tracks['id'] = g
    df_tracks = df_tracks.set_index(['N', 'id'])
    df_tracks.to_pickle(out_file.replace('_photons', '_tracks'))

    print("Generando pickle towers")
    df_towers = df_towers.sort_values(by=['N', 'et'], ascending=[True, False])
    g = df_towers.groupby('N', as_index=False).cumcount()
    df_towers['id'] = g
    df_towers = df_towers.set_index(['N', 'id'])
    df_towers.to_pickle(out_file.replace('_photons', '_towers'))
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
