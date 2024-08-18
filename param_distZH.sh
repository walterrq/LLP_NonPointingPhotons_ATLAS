#!/bin/bash

#todo Queremos que las funciones de nuestros scripts sean darle el nombre al tag y los valores de los cortes al run_card para finalmente correr madgraph

function changing () {
	x=$(find|grep "$1" "${run_path}")
	sed -i "s/$x/$2/g" "${run_path}" > /dev/null 2>&1
	#echo "$x"
}

function run_mg5 () {
	for tev in $tevs
		do
		tev_="$((tev*1000/2))"
		# Define las energias de los beams en el run_card
		beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
		beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
		changing " = ebeam1 " "$beam1"
		changing " = ebeam2 " "$beam2"
		
		# Le da el tag apropiado al run
		tag="  ${channel}_${mindex}_${aindex}_${tev}     = run_tag ! name of the run "
		#todo descomentar el echo si se desea debuggear
		#echo $tag
		#exit 0
		changing " = run_tag " "$tag"
			
		#Copia el param_card correspondiente
		filename_d="${folder_destiny}/param_card.dat"
		cp "${filename_o}" "${filename_d}" 
		#todo recordamos filename_o="$4"
			
		# Correr el run
		cd "${folder_destiny}"
		cd ..
		./bin/madevent "${config_path}" #> /dev/null 2>&1
	done
}

#echo "Param_dist"
#echo $PWD

#!Recordamos: $1: vacio, $2:Mass ,$3:Alpha ,$4:ubicacion global (absolute path), $5: numero de eventos, $6: folder de madgraph(para este caso se tiene: madgraph_folder="/Collider/MG5_aMC_v3_1_0")

mindex="$2" #todo Esto fue especificado en benchesZH
aindex="$3" #todo Esto fue especificado en benchesZH, que a su vez fue especificado en master
filename_o="$4"
config_path="${PWD}/HN_run_config.txt"

tevs="13"

small="  1e-12 = small_width_treatment"
nevents="  ${5} = nevents ! Number of unweighted events requested "
ct="  0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)"
#todo con -1 no se sobreescribe la varialbe. Con 0 si se guarda la distancia
decay="   True  = cut_decays    ! Cut decay products "

pta_min=" 10.0  = pta       ! minimum pt for the photons "
ptl_min=" 10.0  = ptl       ! minimum pt for the charged leptons "
ptl_min_WH=" 5.0  = ptl       ! minimum pt for the charged leptons "
ptj_min=" 25.0  = ptj       ! minimum pt for the jets "
etaa_max=" 2.4  = etaa    ! max rap for the photons "
etal_max="# 2.5  = etal    ! max rap for the charged leptons"
etapdg_max=" {11: 2.5, 13: 2.7, 15: 5.0} = eta_max_pdg ! rap cut for other particles (syntax e.g. {6: 2.5, 23: 5})"
ptcl_min=" 27.0  = xptl ! minimum pt for at least one charged lepton "
etaj_max=" -1.0 = etaj    ! max rap for the jets "
drjj_min=" 0.0 = drjj    ! min distance between jets "
drjl_min=" 0.0 = drjl    ! min distance between jet and lepton "
r0gamma="  0.0 = R0gamma ! Radius of isolation code"
 
###################

tipos="ZH WH TTH" #todo 3 procesos con los que hemos estado trabajando
#! Recordemos que en masterZH ya pudimos generar el evento deseado, a los cuales llamamos val-HN_ZH, val-HN_WH y val-HN_TTH

for channel in ${tipos} 
#todo Se corren los tres procesos. for por default separa por white space los string, por eso se evaluan los 3 tipos de procesos 1 por 1
	do
	folder_destiny="${6}/val-HN_${channel}/Cards" 
	#todo UBICACION: recordamos que los eventos creados reciben de nombre val-HN_ZH o cualquiera de los tipos posibles. Como esta direccion final(channel) depende de tipo, entoces esto peromite que nos ubiquemos en la carpeta Cards del evento del tipo que queremos analizar.
	run_path="${folder_destiny}/run_card.dat"
	
	#TODO La funcion changing busca el argumento en el archivo que yo le de (run_card en este caso), saca la linea completa y reemplaza por lo que yo le de como segundo argumento

	changing " = small_width_treatment "  "$small"
	#TODO changing busca que linea en todo el run_card dice small width treatment. 
	#!Luego se reemplaza toda esa linea por lo que yo le paso como segundo argumento, que es la variable small(en nuestro caso, tal como se definio arriba se tiene: small="  1e-12 = small_width_treatment")
	changing " = nevents "  "$nevents"
	changing " = time_of_flight "  "$ct"
	changing " = cut_decays "  "$decay"
	changing " = pta "  "$pta_min"
	changing " = ptl "  "$ptl_min"
	#Todo caso especial se da en el WH, que se necesita un ptl especifico para este canal
	#todo si no encuentra dicha linea, no cambia nada
	if [ $channel == "WH" ]
		then
		changing " = ptl "  "$ptl_min_WH"
	fi
	changing " = ptj "  "$ptj_min"
	changing " = etaa "  "$etaa_max"
	changing " = etal "  "$etal_max"
	changing " = eta_max_pdg "  "$etapdg_max"
	changing " = xptl"  "$ptcl_max"
	changing " = etaj " "$etaj_max"
	changing " = drjj " "$drjj_min"
	changing " = drjl " "$drjl_min"
	changing " = R0gamma " "$r0gamma"
	
	run_mg5 "$channel"

done
