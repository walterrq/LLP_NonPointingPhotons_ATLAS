#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

echo "Analysis master"

#echo "01"
#python 01_cases_hepmc_reader.py              ## Gets raw data from .hepmc
# echo "02"
#python 02_cases_photons_data.py             ## Use the data from hepmc to generate observables: z_origin and t_gamma
# echo "03"
#python 03_making_complete_hepmc.py         ## Rewrite .hepmc adding observables z_origin and t_gamma
# echo "04"
#python 04_run_Delphes.py "$3" "$2"         ## Run the modified Delphes through .hepmc
# echo "05"
#python 05_extracting_root.py               ## Extract important data from .root generated by Delphes
# echo "06"
#python 06_bins_after_Delphes.py "$1"       ## Apply the analysis over "experimental" data

#todo codigo optimizado
# echo "01, 02, 03"
#python 03_making_full_exec_hepmc.py

# echo "4 full"
#python 04_run_Delphes_full.py "$3" "$2"

# echo "danilohepmc"
#python masparecido.py

# echo "05 full"
#python 05_extracting_root_full.py

 echo "cod con R y abs solo para foton pt max"
 #python ultimo_1D_functions.py
#python "test_mayorPT.py" 
python "analisis_f_max_pt.py"


#!FALTA
# echo "06 full"
#python 06_bins_after_Delphes_full.py "$1"

# echo "01 02 03 con fe"
#python 03_making_full_if_exec_hepmc.py

# echo "01 02 03 ahora si"
#python 03_making_full_par_if_exec_hepmc.py

# echo "comparando ctaus con L"
#python 03_making_full_ct_exec_hepmc.py

# echo "comparando ctau sin beta"
# python 03_making_full_ctnob_exec_hepmc.py

# echo "comparando ctau con beta"
#python 03_making_full_ct_exec_hepmc.py

#echo "ayuda"
#python 03_making_full_selection_hepmc.py

#echo "cositas"
#python 03_danilo_full_selection_hepmc.py

#echo "ultimo"
#python 03_ultimo_full_selection_hepmc.py

#!echo "funcional"    
#python datacompleto_1.py
#python datacompleto_2F.py
#python datacompleto_1D.py

#todo compare
#python comparehepmc.py
#python comparewrite.py

#!final

#python comparewriteoptm.py

#python comparehepmc_new.py
#python comparewrite.py
# echo "07"
#python 07_making_graphs.py
# echo "07a"
#python "07a_making_graphs (copy).py"
# echo "07a_alpha_it"
#python 07a_making_graphs-copy-_alpha_it.py

# echo "09"
#python 09_contour_graphs.py