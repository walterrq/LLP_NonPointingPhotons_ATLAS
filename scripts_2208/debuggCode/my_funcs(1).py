import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

cross_sec = {13:{'VBF': {50: 0.231009, 30: 0.233562, 10: 0.234305}},
             8: {'VBF': {50: 1.12885, 30: 1.14555, 10: 1.13663},'GF': {50: 7.4004, 30: 7.4165, 10: 7.409}}} # pb
l = {8: 20.3,13: 139} #fb-1
br_nn = 0.21
br_np = {8: {50: 0.719,30: 0.935,10: 0.960}, 13:{50: 0.799, 30: 0.938, 10: 0.960}}

#phs es el data set que quiero analizar
#surr es lo que queremos aislar
def isolation(phs, surr, obs, same=False, dR=0.2):
    phs_list = []
    #print(phs)
    #con el codigo de abajo se repite los indices, por ejemplo:
    #                value  other_value
    #Level_1 Level_2                  
    #A       1         10          100
    #        2         20          200
    #B       1         30          300
    #        2         40          400
    #C       1         50          500
    #nos dara
    #Index(['A', 'A', 'B', 'B', 'C'], dtype='object', name='Level_1')
    #print(phs.index.get_level_values(0))
    #el unique permite no repetirlos
    #print(phs.index.get_level_values(0).unique()[:])
    #ix sera el numero del evento
    for ix in phs.index.get_level_values(0).unique()[:]:
        event_ph = phs.loc[ix]
        #print(f"DataFrame for index '{ix}':\n{event_ph}\n")
        #print(len(event_ph))
        #con este print obtendremos los siguiente:
        #para el ix = 263, tendremos el siguiente data frame
        #DataFrame for index '263':
        #    pdg         pt       eta       phi      mass eff_value  detected
        #id                                                                  
        #0    11  64.123085  0.186741 -0.068175  0.000511  0.974118      True
        #1    11  29.816454  1.769422 -0.714027  0.000511  0.849529      True

        try:
            event_surr = surr.loc[ix]
            # print(event_surr)
            #
            for index_ph, row_ph in event_ph.iterrows():
                # print(row_ph)
                cone = 0
                #print("Index ph:", index_ph)
                #print("Row ph:", row_ph)
                #el indice sera el asociado al numero de la particula en un determinado evento
                #row_ph es lo siguiente, si por ejemplo tenemos
                #id   A  B
                #0    1  4
                #1    2  5
                #2    3  6
                #row ph sera por cada indice
                #Index: 0
                #Row: A    1
                #B    4
                #Name: 0, dtype: int64
                for index_d, row_d in event_surr.iterrows():
                    #para un valor de ph fijo en un evento, lo comparamos con las demas particulas
                    dr = np.sqrt((row_d.phi - row_ph.phi) ** 2 + (row_d.eta - row_ph.eta) ** 2)
                    #consideramos el caso que tengamos que aislar foton de foton
                    #si las particulas son iguales, entonces no es necesario aislarlas
                    #print("Index surr:", index_d)
                    #print("Row surr:", row_d)
                    if same and (index_ph == index_d):
                        dr = dR*1000
                    if dr < dR:
                        #normalmente pt es el observable
                        cone += row_d[obs]
            
                #print("Cone:", cone)
                phs_list.append(cone)
                #print("Cone list:", phs_list)
        except KeyError:
            #en que caso podria surgir este error?
            #puede que el evento numero 5 no hayan habido particulas surr y por lo tanto sale error
            #por que se extiende la lista con ceros?
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


def isolation_muon(phs, surr, obs, same=False, dR=0.2):
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
                        
                        #print("Hubo solo un jet, pero entre est y el muon tiene dr < 0.2 asi que no le hacemos nada")
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