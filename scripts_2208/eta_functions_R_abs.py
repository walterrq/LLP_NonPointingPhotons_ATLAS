#variable definition (to be placed outside the main)

#Values for parametrization for eta < 0.8
#R1
import numpy as np
def eta_func_R_abs(eta_ini,eta_mid):

    eta_1 = np.abs(eta_ini)
    eta_2 = np.abs(eta_mid)

    #print("eta1, eta2: ", eta_1, eta_2)
    #Values for parametrization for eta < 0.8
    #R1
    as_nblw = 1567.8
    bs_nblw = -18.975
    cs_nblw = -17.668
    #R2
    am_nblw = 1697.1
    bm_nblw = -15.311
    cm_nblw = -64.153

    #Values for parametrization for eta > 0.8
    #R1
    as_nabv = 1503.2
    bs_nabv = 71.716
    cs_nabv = -41.008
    #R2
    am_nabv = 1739.1
    bm_nabv = -75.648
    cm_nabv = -18.501

    '''
    Parametrization for R1 and R2 (to be inserted inside the main)
    '''

    if eta_1 < 0.8 and eta_2 < 0.8:
        R1 = as_nblw + bs_nblw*eta_1 + cs_nblw*(eta_1**2)
        R2 = am_nblw + bm_nblw*eta_2 + cm_nblw*(eta_2**2)
    elif eta_1 < 0.8 and eta_2 > 0.8:
        R1 = as_nblw + bs_nblw*eta_1 + cs_nblw*(eta_1**2)
        R2 = am_nabv + bm_nabv*eta_2 + cm_nabv*(eta_2**2)
    elif eta_1 > 0.8 and eta_2 < 0.8:
        R1 = as_nabv + bs_nabv*eta_1 + cs_nabv*(eta_1**2)
        R2 = am_nblw + bm_nblw*eta_2 + cm_nblw*(eta_2**2)
    elif eta_1 > 0.8 and eta_2 > 0.8:
        R1 = as_nabv + bs_nabv*eta_1 + cs_nabv*(eta_1**2)
        R2 = am_nabv + bm_nabv*eta_2 + cm_nabv*(eta_2**2)
        
    return R1, R2
    #What if eta =0.8?