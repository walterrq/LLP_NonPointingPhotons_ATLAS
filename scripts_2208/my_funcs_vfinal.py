import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

cross_sec = {13:{'VBF': {50: 0.231009, 30: 0.233562, 10: 0.234305}},
             8: {'VBF': {50: 1.12885, 30: 1.14555, 10: 1.13663},'GF': {50: 7.4004, 30: 7.4165, 10: 7.409}}} # pb
l = {8: 20.3,13: 139} #fb-1
br_nn = 0.21
br_np = {8: {50: 0.719,30: 0.935,10: 0.960}, 13:{50: 0.799, 30: 0.938, 10: 0.960}}

#Tenemos como input para isolation lo siguiente:
#phs = Dataset de lo que queremos aislar u observar (fotones)
#surr = surrounding. Particulas que estan alrededor. Dataset de las particulas de las que quiero aislar mi phs.
#obs = observable que usaremos para medir.
    #Basicamente, nos dice que si hay una cantidad de pt o alguna cantidad que se sume, se considera que no esta aislado
#same = usamos esto porque queremos que, por ejemplo, un foton este aislado de otros fotones.
#Si comparo fotones tengo que poner el value same en true para que elimine el foton dentro del cono y no lo considere.
#dR = delta R (separacion angular). Util para definir el aislamiento. Definido en base a phi y eta
#Hay casos en los que quiero fotones que esten aislados de todo(incluyendo fotones). En ese caso, se cambia el same a True

def isolation(phs, surr, obs, same=False, dR=0.2):
    phs_list = []
    for ix in phs.index.get_level_values(0).unique()[:]:
        event_ph = phs.loc[ix]

        # print(f"Evento numero: '{ix}'")

        # print("Jets")
        # print(event_ph)

        try:
            event_surr = surr.loc[ix]
            # print(event_surr)

            # print("Muones")
            # print(event_surr)

            for index_ph, row_ph in event_ph.iterrows():
                # print(row_ph)

                # print(f"Estamos analizando el cono generado por el jet con indice '{index_ph}'\n")

                #Se inicializa energia del cono en 0
                cone = 0
                #Analizo particula por particula
                for index_d, row_d in event_surr.iterrows():
                    dr = np.sqrt((row_d.phi - row_ph.phi) ** 2 + (row_d.eta - row_ph.eta) ** 2)
                    #Quiero que la energia del cono, sin considerar el foton,sea el 5% de la energia del foton.
                    #Si considero al foton no será el 5% de la energia.
                    if same and (index_ph == index_d):
                        dr = dR*1000
                    if dr < dR:
                        cone += row_d[obs]
                        # print("obs")
                        # print(obs)
                        # print("Row_d")
                        # print(row_d[obs])
                        # print("cone")
                        # print(cone)

                    #     print(f"Muon '{index_d}' que esta dentro del cono generado por el jet '{index_ph}'")
                    #     row_isolated_muon = surr.loc[(ix, index_d)]
                    #     print(row_isolated_muon)
                    #     print("Jet el cual genera el cono")
                    #     print(row_ph)
                    # else:
                    #     print(f"Muon '{index_d}' que esta fuera del cono generado por el jet '{index_ph}'")
                    #     row_non_isolated_muon = surr.loc[(ix, index_d)]
                    #     print(row_non_isolated_muon)
                    #     print("Jet el cual genera el cono")
                    #     print(row_ph)

                phs_list.append(cone)
        except KeyError:
            phs_list.extend([0]*len(event_ph))

    return phs_list

def my_arctan(y,x):

    arctan = np.arctan2(y,x)
    if not isinstance(x, (pd.Series, np.ndarray)):
        arctan = np.asarray(arctan)
    arctan[arctan < 0] += 2*np.pi
    return arctan

def get_scale(tev,type,mass,nevents=100000):
    return ((cross_sec[tev][type][mass]*1000)*l[tev]*br_nn*(br_np[tev][mass])**2)/nevents

def format_exponent(ax, axis='y', size=13):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
    else:
        offset_text = "   "
    # Turn off the offset text that's calculated automatically
    ax_axis.offsetText.set_visible(False)

    # Add in a text box at the top of the y axis
    ax.text(x_pos, y_pos, offset_text, fontsize=size, transform=ax.transAxes,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)

    return ax

def deltaRcalculation(event_muon, event_track, event_tower):
    '''
    Funtion that calculates 2 delta R: between first input (muon dataframe) and second input (tracks dataframe)
    and between first input and third input(towers dataframe). 
    There is 2 output: deltaR(muon, tracks) and deltaR(muon, towers) 
    '''
    muon_phi = event_muon['phi'].values
    muon_eta = event_muon['eta'].values

    if event_track is not None:
        # Handle the case when event_track is provided
        #print("Using event_track for calculation.")
         # Use event_track variables
        track_phi = event_track['phi'].values
        track_eta = event_track['eta'].values

        # Calculate Δphi and Δη using numpy broadcasting (for muons and tracks)
        # We substract matrix elements:
        # say muon_phi = (1 , 2) and track_phi = (3, 4)
        # then np.subtract.outer(muon_phi, track_phi) gives [(-2, -3),
        #                                                   (-1, -2)] 
        # Therefore, we successfully have [(phiMu1Track1, phiMu1Track2),
        #                                  (phiMu2Track1, phiMu2Track2)] 
        delta_phi_tracks = np.subtract.outer(muon_phi, track_phi)
        delta_eta_tracks = np.subtract.outer(muon_eta, track_eta)

        # Calculate ΔR for all muon-track pairs
        delta_r_tracks = np.sqrt(delta_phi_tracks**2 + delta_eta_tracks**2)
        #print("ΔR for muon-track pairs:\n", delta_r_tracks)
  
    if event_tower is not None:
        # Handle the case when event_tower is provided
        #print("Using event_tower for calculation.")
        tower_phi = event_tower['phi'].values
        tower_eta = event_tower['eta'].values

        # Calculate Δphi and Δη using numpy broadcasting (for muons and towers)
        delta_phi_towers = np.subtract.outer(muon_phi, tower_phi)
        delta_eta_towers = np.subtract.outer(muon_eta, tower_eta)

        # Calculate ΔR for all muon-tower pairs
        delta_r_towers = np.sqrt(delta_phi_towers**2 + delta_eta_towers**2)
        #print("ΔR for muon-tower pairs:\n", delta_r_towers)
    
     # Return a tuple with both delta_r_tracks and delta_r_towers if both are defined
    if event_track is not None and event_tower is not None:
        return delta_r_tracks, delta_r_towers

    # If only one is defined, return that, otherwise return None for both
    return delta_r_tracks if event_track is not None else None, \
           delta_r_towers if event_tower is not None else None

def cone_isolation(track_pt, tower_et, muon_pt, track_pdg, delta_r_tracks, delta_r_towers, delta_r_max= 0.2, pt_min = 0.5):

    '''
    cone_isolation receives data from a single event and calculates the ratio of[ p_T cone20 / p_T muon ] for tracks and 
    [e_T cone 20 / p_T muon] for towers. These two values are the output.
    '''
    # Determine if there are no leptons within the ΔR max condition

    #print("Printing delta_r_towers")
    #print(delta_r_towers)
    if delta_r_tracks is not None:

        # Apply the ΔR max condition for tracks
        # within_cone_tracks gives a matrix of trues and falses depending on the condition
        within_cone_tracks = (delta_r_tracks < delta_r_max)

        #print("within_cone_tracks")
        #print(within_cone_tracks)

        # Apply the pT min condition to the tracks, setting tracks below pt_min to 0
        # say track_pt = (1,2,3,4,5) and pt_min = 3 then result is (0, 0, 0, 4, 5)
        track_pt_filtered = np.where(track_pt > pt_min, track_pt, 0)

        track_pt_filtered = np.array(track_pt_filtered)

        #print("track_pt_filtered")
        #print(track_pt_filtered)

        # Convert within_cone_tracks to an integer mask (1 for True, 0 for False)
        within_cone_tracks_int = within_cone_tracks.astype(int)

        #print("within_cone_tracks_int")
        #print(within_cone_tracks_int)

        # Multiply the track pT values by the within-cone condition to filter out tracks outside the cone
        # Now, trackfiltered has only the PT information of the tracks that satisfy the condition > pt_min and are within the cone
        track_filtered = within_cone_tracks_int * track_pt_filtered

        #! We do not consider in the analysis the same muon 
        # print("track_pdg")
        # print(track_pdg)
        track_pdg_array = np.array(track_pdg)
        track_pdg_mask = np.where((track_pdg_array == 13) | (track_pdg_array == -13), 1, 0)
        
        # print("track_pdg_array")
        # print(track_pdg_array)

        # print("track_pdg_mask")
        # print(track_pdg_mask)

        # print("within_cone_tracks_int")
        # print(type(within_cone_tracks_int))

        # pdg_within_mask is a mask where only the tracks within the cone that have |id|= 13 have 1. others have 0
        pdg_within_mask = within_cone_tracks_int * track_pdg_mask

        # print("pdg_within_mask")
        # print(pdg_within_mask)

        # print("delta_r_tracks")
        # print(delta_r_tracks)

        # delta_r_filtered will return a matrix containing only the delta_r_tracks of the muon with tracks with id =13 
        pdg_within_mask = np.where(pdg_within_mask == 0, np.inf, pdg_within_mask)
        # Any track that is not a muon or is not within the cone is labeled as infinity. Muons within cone are 1

        delta_r_filtered = np.array(delta_r_tracks) * pdg_within_mask #We use the element multiplication
        # print("delta_r_filtered")
        # print(delta_r_filtered)

        # Replace zeros with np.inf temporarily to avoid selecting as minimum the 0 from the mask
        if np.any(np.all(delta_r_filtered == np.inf, axis=1)):
            # Find the indices of rows that are all zeros. We should always have a track within the radius of the muon,
            # because muons always have their associated tracks
            zero_rows = np.where(np.all(delta_r_filtered == np.inf, axis=1))[0]
            print(f"Error: Row(s) {zero_rows} contain only infinites. Aborting operation.")
            return
        else:

            # Find the index of the minimum value in each row now that all tracks with pdg=13 within the cone are not infinite
            min_deltar = np.argmin(delta_r_filtered, axis=1)

            # print("Printing min_deltar")
            # print(min_deltar)

            min_deltar_row = np.arange(min_deltar.shape[0])  # Generate an array of row indices [0, 1, 2]
            track_filtered[min_deltar_row, min_deltar] = 0.0  # Set the corresponding elements to 0.0

        #print("track_filtered")
        #print(track_filtered)

        # Calculate the sum of pT of tracks within the cone for each muon
        # Here we have an array of the sum of PT of every track satisfying the conditions 
        # (pt1_track1+pt1_track2[related to mu1], pt2_track1+pt2_track2+pt2_track3[related to mu2], pt3_track1[related to mu3], etc)
        # This corresponds to p_T cone20 from 2012.00578
        sum_pt_within_cone_tracks = np.sum(track_filtered, axis=1)

        #print("sum_pt_within_cone_tracks")
        #print(sum_pt_within_cone_tracks)

        # Calculate the isolation ratio for each muon based on tracks
        # p_T cone20 / muon_pt
        isolation_ratio_tracks = sum_pt_within_cone_tracks / muon_pt

        # print("isolation_ratio_tracks")
        # print(isolation_ratio_tracks)


        
    if delta_r_towers is not None:

        # Apply the ΔR max condition
        within_cone_towers = (delta_r_towers < delta_r_max)

        # print("within_cone_towers")
        # print(within_cone_towers)

        # print("tower_et before filter")
        # print(tower_et)
        
        # Apply the ET min condition to the towers
        #We have no minimum condition, so we use et > 0
        tower_et_filtered = np.where(tower_et > 0.0, tower_et, 0)

        tower_et_filtered = np.array(tower_et_filtered)

        # print("tower_et_filtered")
        # print(tower_et_filtered)

        within_cone_towers_int = within_cone_towers.astype(int)

        # print("within_cone_towers_int")

        # print(within_cone_towers_int)

        tower_filtered = within_cone_towers_int*tower_et_filtered

        # Calculate the sum of ET of towers within the cone for each muon
       
        # partimos del a mascara que podemos observar que se transforma en 1 si es true y 0 si es false
        # eso nos permite solo considerar los pt de las particulas dentro del cono y si estan fuera no sumar
        # entonces definimos el vector tower_et_filtered que multiplicara toda la columna
        #  t1 t2                                   t1 t2           
        #u1 T T       ----> se transforma a     u1    1 1    ----> 2 +5 = 7
        #t2 F F                                 u2    0 0    ----> 0
        #                                      (et1= 2 et2 =5) multiplicamos esto por la columna

        # ahora en caso et por ejemplo sea menor a el et min entonces lo mandamos a cero para que asi no contribuya
        #  t1 t2                                   t1 t2           
        #u1 T T       ----> se transforma a     u1    1 1    ----> 0 +5 = 5
        #t2 F F                                 u2    0 0    ----> 0
        #                                      (et1= 0 t2 =5)
        #ceste es un caso donde t1 y t2 estan dentro del cono de u1

        # print("tower_filtered")
        # print(tower_filtered)

        sum_et_within_cone_towers = np.sum(tower_filtered, axis=1)

        # print("sum_et_within_cone_towers")
        # print(sum_et_within_cone_towers)

        # Calculate the isolation ratio for each muon
        # print("muon_pt")
        # print(muon_pt)

        isolation_ratio_towers = sum_et_within_cone_towers/muon_pt
    
    if( (delta_r_tracks is not None) and (delta_r_towers is not None)):
        return isolation_ratio_tracks, isolation_ratio_towers

    # If only one is defined, return that, otherwise return None for both
    if((delta_r_tracks is not None) and (delta_r_towers is None)):
        return isolation_ratio_tracks, None
    
    if((delta_r_tracks is None) and (delta_r_towers is not None)): 
        return None, isolation_ratio_towers

    #return isolation_ratio_tracks, isolation_ratio_towers

def muon_isolation(df_muons, df_tracks, df_towers, pt_ratio_max=0.16):

    """
    Calculates muon isolation using the previously defined functions. Gives a new list with m
    muons that dont contain df_tracks and df_towers within a radious according to
    the analysis in DOI:10.1140/epjc/s10052-021-09233-2 and DOI:10.7916/d8-n5sm-qj56
    """

    #0 significa que no es un jet impostor, 1 significa que si lo es
   
    for ix in df_muons.index.get_level_values(0).unique()[:]:
        event_muon = df_muons.loc[ix]

        #print("Jets")
        #print(event_muon)
        # print("ix", ix)

        try:
            event_track = df_tracks.loc[ix]

            try:
                #Tenemos tanto Tracks como Towers
                event_tower = df_towers.loc[ix]

                # print("Printing event_track")
                # print(event_track)

                # print("Printing event_tower")
                # print(event_tower)

                # print(f"Evento que tiene track y tower: '{ix}'")
                
                delta_r_tracks, delta_r_towers = deltaRcalculation(event_muon, event_track, event_tower)

                #print("delta_r_tracks")
                #print(delta_r_tracks)

                # print("delta_r_towers")
                # print(delta_r_towers)

                track_pt = event_track['pt'].values
                tower_et = event_tower['et'].values
                muon_pt = event_muon['pt'].values
                track_pdg = event_track['pdg'].values
                
                isolation_ratio_tracks, isolation_ratio_towers = \
                cone_isolation(track_pt, tower_et, muon_pt, track_pdg, delta_r_tracks, delta_r_towers)

                # print("isolation_ratio_tracks")
                # print(isolation_ratio_tracks)

                # print("isolation_ratio_towers")
                # print(isolation_ratio_towers)
                
                #[ p_T cone20 / p_T muon ] + 0.4*[e_T cone 20 / p_T muon] 
                isolation_ratio = isolation_ratio_tracks + 0.4 * isolation_ratio_towers

                #print("isolation_ratio")
                # print(isolation_ratio)

                # Determine if each muon is isolated based on the isolation ratio
                # If isolation_ratio < 0.16 then the muon is isolated
                # isolated_muon_mask is an array of trues and falses depending if is isolated or not
                isolated_muon_mask = (isolation_ratio < pt_ratio_max)

                # Print statements for all variables
                not_isolated_muon_mask = ~isolated_muon_mask
                
                # print("isolated_muon_mask")

                # print(isolated_muon_mask)

                # print("not_isolated_muon_mask")

                # print(not_isolated_muon_mask)
                
                # Filter and store the non-isolated muons with the event number (N) and muon id
                if any(not_isolated_muon_mask):

                    # Filter non-isolated muons
                    # We generate a subdataset with only the information from the event muons that are not isolated
                    not_isolated_muons = event_muon[not_isolated_muon_mask].copy()

                    # print("not_isolated_muons")
                    # print(not_isolated_muons)

                    # Get the index list for non-isolated muons
                    index_list = not_isolated_muons.index.tolist()

                    # print("index_list")
                    # print(index_list)
                    # Loop through the index list and remove the non-isolated muons from the dataframe
                    #recordar que ix es el numero del evento
                    for index_event in index_list:
                        df_muons = df_muons.drop((ix, index_event))
                                   
            except KeyError:
                #Tenemos solo tracks y no towers
                
                # print(f"Evento que tiene solo track'{ix}'")
                
                
                # print("event_track")
                # print(event_track)

                #print(deltaRcalculation(event_muon, event_track, None))
                delta_r_tracks, delta_r_towers = deltaRcalculation(event_muon, event_track, None)
                
                
                # print("delta_r_tracks")
                # print(delta_r_tracks)

                # print("delta_r_towers")
                # print(delta_r_towers)
                
                
                track_pt = event_track['pt'].values
                tower_et = None
                muon_pt = event_muon['pt'].values
                track_pdg = event_track['pdg'].values
                
                isolation_ratio_tracks, isolation_ratio_towers = \
                cone_isolation(track_pt, tower_et, muon_pt, track_pdg, delta_r_tracks, delta_r_towers)

                
                isolation_ratio = isolation_ratio_tracks

                isolated_muon_mask = (isolation_ratio < pt_ratio_max)

                not_isolated_muon_mask = ~isolated_muon_mask

                
                if any(not_isolated_muon_mask):
 
                    not_isolated_muons = event_muon[not_isolated_muon_mask].copy()

                    index_list = not_isolated_muons.index.tolist()

                    for index_event in index_list:
                        df_muons = df_muons.drop((ix, index_event))
                        
                
        except KeyError:
            

            #si no tiene tracks, puede que si tenga towers
            try:
                event_tower = df_towers.loc[ix]
                #event_muon = df_muons.loc[ix]

                print(f"Evento que tiene solo tower'{ix}'")
                
                
                print("event_track")
                print(event_track)

                #print(deltaRcalculation(event_muon, None, event_tower))
                delta_r_tracks, delta_r_towers = deltaRcalculation(event_muon, None, event_tower)
                
                
                print("delta_r_tracks")
                print(delta_r_tracks)

                print("delta_r_towers")
                print(delta_r_towers)
                
                #print("Hola")
                track_pt = None
                tower_et = event_tower['et'].values
                muon_pt = event_muon['pt'].values
                track_pdg = None
                
                #cone_isolation(track_pt, tower_et, muon_pt, delta_r_tracks, delta_r_towers, delta_r_max= 0.2, pt_min = 0.5)
                isolation_ratio_tracks, isolation_ratio_towers = \
                cone_isolation(track_pt, tower_et, muon_pt, track_pdg, delta_r_tracks, delta_r_towers)
                
                #print("Hola 2")
                print("isolation_ratio_tracks")
                print(isolation_ratio_tracks)

                print("isolation_ratio_towers")
                print(isolation_ratio_towers)
                
                isolation_ratio = 0.4 * isolation_ratio_towers

                isolated_muon_mask = (isolation_ratio < pt_ratio_max)

                not_isolated_muon_mask = ~isolated_muon_mask


                print("isolated_muon_mask")

                print(isolated_muon_mask)

                print("not_isolated_muon_mask")

                print(not_isolated_muon_mask)
                
                if any(not_isolated_muon_mask):
 
                    not_isolated_muons = event_muon[not_isolated_muon_mask].copy()

                    index_list = not_isolated_muons.index.tolist()

                    for index_event in index_list:
                        df_muons = df_muons.drop((ix, index_event))
                
            except KeyError:
                #no tiene ninguno de los dos (no elimina nada)
                a = 1
                print("No hay nada uwu")
        
    return df_muons

def format_exponent(ax, axis='y', size=13):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
    else:
        offset_text = "   "
    # Turn off the offset text that's calculated automatically
    ax_axis.offsetText.set_visible(False)

    # Add in a text box at the top of the y axis
    ax.text(x_pos, y_pos, offset_text, fontsize=size, transform=ax.transAxes,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)

    return ax

def overlap_removal_muon_jet(phs, surr, obs, same=False, dR=0.2):
    phs_list = []
 
    for ix in phs.index.get_level_values(0).unique()[:]:
        event_ph = phs.loc[ix]
  
        try:
            event_surr = surr.loc[ix]
            total_surr = event_surr.shape[0]
            #print("event_surr")
            #print(event_surr)
            #print("Total Surr")
            #print(total_surr)
            #sys.exit("Salimos")
            #
            for index_ph, row_ph in event_ph.iterrows():
                # print(row_ph)
                cone = 0

                for index_d, row_d in event_surr.iterrows():
                    #para un valor de ph fijo en un evento, lo comparamos con las demas particulas
                    dr = np.sqrt((row_d.phi - row_ph.phi) ** 2 + (row_d.eta - row_ph.eta) ** 2)

                    if same and (index_ph == index_d):
                        dr = dR*1000
                    
                    if(total_surr == 1):
                        
                        #print("Hubo solo un jet, pero entre este y el muon tiene dr < 0.2 asi que no le hacemos nada")
                        if(dr > 0.2):
                            #print("Estamos en un caso donde dr>0.2 y solo hay un jet")
                            cone += row_d[obs]
                    
                    else:
                        if dr < dR:
                            #normalmente pt es el observable
                            cone += row_d[obs]

                #sys.exit("Salimos")
                #print("Cone:", cone)
                phs_list.append(cone)
                #print("Cone list:", phs_list)
        except KeyError:
            #en que caso podria surgir este error?
            #puede que el evento numero 5 no hayan habido particulas surr y por lo tanto sale error
            #por que se extiende la lista con ceros?
            phs_list.extend([0]*len(event_ph))
    
    return phs_list