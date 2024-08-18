#variable definition (to be placed outside the main)

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

if eta < 0.8:
    R1 = as_nblw + bs_nblw*eta + cs_nblw*(eta^2)
    R2 = am_nblw + bm_nblw*eta + cm_nblw*(eta^2)

else:
    R1 = as_nabv + bs_nabv*eta + cs_nabv*(eta^2)
    R2 = am_nabv + bm_nabv*eta + cm_nabv*(eta^2)
    
#What if eta =0.8?