#!/bin/bash

#todo ejecutar desde collider
cd llpatlas #!Comentamos esto porque ya se monto en esta ubicacion
# $SHELL

#Debuggeo simple
echo "hello"

# *Le bajamos el numero de eventos solo para practicar
#x1=500
x1=10000

#!Las versiones de MG5 deben coincidir con las que se este usando
madgraph_folder="/Collider/MG5_aMC_v2_9_11"
delphes_folder="/Collider/MG5_aMC_v2_9_11/Delphes"
destiny_folder="/Collider"

#The code removes the directory located at "/Collider/scripts_2208" and all its contents, 
#and it does so without asking for confirmation (-rf options). 
#The destiny_folder variable is used to specify the base directory where the "scripts_2208" directory is located.

rm -rf "${destiny_folder}/scripts_2208"
mkdir -p "${destiny_folder}/scripts_2208/data/raw"

tar -xf heavNeff4_UFO.tar.xz -C "${madgraph_folder}/models/" #Descarga el UFO en las carpetas models de MG5

##? BEGIN FOR AWS INSTANCE
sed -i 's+run_mode = 2+run_mode = 7+' ${madgraph_folder}/input/mg5_configuration.txt
##TODO > La anterior linea reemplaza el multicore mode(2) con sigle-machine mode(0)
##sed -i 's+nb_core = 4+nb_core = 1+' ${madgraph_folder}/input/mg5_configuration.txt
##TODO > La anterior linea cambia el numero de cores usadas(de 4 a 1)
##? END FOR AWS INSTANCE

##todo Procedemos a cambiar el madevent_interface.py porque queremos que tenga otros comandos. En primer lugar, no queremos que se compriman los hepmc porque toma mucho tiempo. En segundo lugar, queremos que estos hepmc se guarden en una carpeta fuera de MG5, la cual es run... Esto se logra comentando la linea de comando que pide comprimir el hepmc

##mv "$madgraph_folder/madgraph/interface/madevent_interface.py" "$madgraph_folder/madgraph/interface/madevent_interface-default.py" ################
##cp "./madevent_interface.py" "$madgraph_folder/madgraph/interface/madevent_interface.py" ##################


##! nb core dice cuantos cores se usan
##! el comando sed reemplaza patrones con otros patrones
##! el -i reemplaza in place: el mismo archivo que ingresas como input, lo regresa como output

sed "s|FOLDER|$madgraph_folder|g" mg5_launches.txt > mg5_launches_proper.txt
##! la linea | tiene la misma funcion de separar como el +. Pero se usan distintas para evitar confusiones
##! En este caso no usamos -i porque hacemos un cambio pero queremos que el output no se sobreescriba, si no, se genere en otra carpeta
##todo Es importante notar que FOLDER es una variable. El sed cambia esta variable por madgraph_folder, que fue definida al inicio del documento. todo esto sucede usando de input mg5_launches.txt

${madgraph_folder}/bin/mg5_aMC mg5_launches_proper.txt #> /dev/null 2>&1
##todo Este comando de arriba ejecuta madgraph usando como input el launches proper generando los eventos. Es como si a mano generamos los eventos. (generate)

bash benchsZH.sh "$x1" "$madgraph_folder"
##todo SINTAXIS: cuando hacemos bash, podemos poner los argumentos de benchsZH. Los argumentos son inputs para el archivo benchsZH. (x1=numero de eventos, madgraph_folder=variable de direccion)

##todo Mantenemos comentado la linea de abajo puesto que la no descompresion ya lo hicimos cambiando el madevent_interface
bash hepmc_dist.sh "$madgraph_folder" "$destiny_folder"

#todo con este comando extraemos los crossections
bash crossec_distZH.sh "$destiny_folder" "$madgraph_folder"

source ~/.bashrc 
#todo Descomentamos esta linea para correr root en docker en la imagen especifica de lpphotons. Con esta linea se hace un update al bash para activar que el docker pueda correr root.
#Con esto podemos delphes y root desde python

cd ./scripts_2208/
#todo Estamos en llpatlas y entramos al scripts_2208 de llpatlas

echo $PYTHONPATH
bash analysis_master.sh "$x1" "$delphes_folder"  "$destiny_folder"

echo "Done!"
