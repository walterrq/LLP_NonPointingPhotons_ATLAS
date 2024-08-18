#!/bin/bash

cd limon #esta linea manda al bash dentro de la carpeta
# se corre independientemente si se corre en docker o la computadora
#cuando se corra en el dokcer, quizas si es necesario el SHELL
#$SHELL

x1=100000 #esto indica el numero de eventos que genera madgraph

#folder donde tenemos madgraph
madgraph_folder="/Collider/MG5_aMC_v2_9_11"
#folder dond tenemos delphes
delphes_folder="/Collider/MG5_aMC_v2_9_11/Delphes"
#donde se va a crear el folder que deseamos
destiny_folder="/Collider"


#removemos las carpetas
#The code removes the directory located at "/Collider/scripts_2208" and all its contents, 
#and it does so without asking for confirmation (-rf options). 
#The destiny_folder variable is used to specify the base directory where the "scripts_2208" directory is located.

#rm -rf "${destiny_folder}/scripts_2208"

# When you run this command, it creates the directory structure:

#/Collider
# scripts_2208
#  data
#   raw

#mkdir -p "${destiny_folder}/scripts_2208/data/raw" #crea la estructura para guardar los datos del analisis

#the command extracts the contents of the specified compressed archive 
#file (heavNeff4_UFO.tar.xz) that is in limon into the directory ${madgraph_folder}/models/
#tar -xf heavNeff4_UFO.tar.xz -C "${madgraph_folder}/models/"

#el sed remplaza patrones (edita archivos de texto)
#normalmente se pone la canitdad de cores de 10 el cual se puede variar si deseas
#con sed buscamos en el archivo input/mg5_configuration.txt el string run_mode=2 y lo remplaza por run_mode=0
# en el archivo mg5_configuration.txt se tiene lo siguiente:
##! Default Running mode
#!  0: single machine/ 1: cluster / 2: multicore
#run_mode = 2

#el codigo de abajo solo es necesario si se tiene una nueva imagen de docker ya que esta cambiara las interfaces
#mv "$madgraph_folder/madgraph/interface/madevent_interface.py" "$madgraph_folder/madgraph/interface/madevent_interface-default.py"
#cp "./madevent_interface.py" "$madgraph_folder/madgraph/interface/madevent_interface.py"

#sed -i 's+run_mode = 2+run_mode = 2+' ${madgraph_folder}/input/mg5_configuration.txt
#sed -i 's+nb_core = 4+nb_core = 1+' ${madgraph_folder}/input/mg5_configuration.txt

#seteamos en madgraph folder pues cuando abrimos el archivo como tal, tiene 
#una direccion ficticia que se llama output FOLDER/ la cual es reemplzada
#la variable madgraph_folder es madgraph_folder="/Collider/MG5_aMC_v3_3_2"
# esta se remplaza por la palabra FOLDER en mg5_launches.txt y se crea un nuevo
#txt con esta edicion llamado mg5_launches_proper.txt. La g hace referencia a global
#por lo que el cambio se hace en todas las palabras FOlDER que aparezcan
#sed "s|FOLDER|$madgraph_folder|g" mg5_launches.txt > mg5_launches_proper.txt

#este codigo solo genera los esqueletos, todavia no afecta los paramcards
#${madgraph_folder}/bin/mg5_aMC mg5_launches_proper.txt #> /dev/null 2>&1  

#bash benchsZH.sh "$x1" "$madgraph_folder"
#bash hepmc_dist.sh "$madgraph_folder" "$destiny_folder"
#bash crossec_distZH.sh "$destiny_folder" "$madgraph_folder"

source ~/.bashrc
cd ./scripts_2208/
echo $PYTHONPATH
bash analysis_master.sh "$x1" "$delphes_folder"  "$destiny_folder"

echo "Done!"