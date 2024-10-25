#!/bin/bash

echo "crosssec"
echo $PWD

tipos="ZH WH TTH"
#tipos="ZH"

echo '' > "${1}/scripts_2208/data/cross_section.dat"
#! > sobreescribe
#! >> append: escribe ademas de lo que ya hay

for tipo in ${tipos}
	do
	#declare -a arr
	folder_origin="${2}/val-HN_${tipo}/Events"
	#todo la anterior linea nos ubica en la carpeta events de ZH, WH o TTH
	cd ${folder_origin}
	runs=( $(ls -d */) )
	#todo la anterior linea me da array con TODOS los run dentro de evens: run_01, run_02, etc
	for run in "${runs[@]}"
		do
		cd "${run}"
		file_mc=("$(ls -d *_banner.txt)")
		#TODO dame todos los archivos(en realidad solo hay uno) que tengan como nombre (cualquiercosa)_banner.txt
		#!Notemos que file_mc tiene comillas, por lo que solo le estamos pasando texto.
		#!file_mc ES UN STRING

		run="${run::-1}_"
		#TODO run es un directorio y por ende su nombre termina con /
		#TODO queremos que no tenga ese /, por ende, lo quitamos con ::-1. Ademas, le añadimos _. De este modo, me quedaria la direccion + _

		cross=$(find|grep "Integrated" "${file_mc}")
		#todo la anterior linea ingresa a file_mc y busca la palabra Integrated. Luego me da como output la linea integrated.
		#! Recordemos que la estructura de find|grep recibe la palabra a buscar y el documento a revisar por default.
		cross=$(sed 's| Integrated weight (pb) |''|g' <<<"$cross")
		#todo En la anterior linea hacemos que del output de la linea que tiene la palabra integrated le eliminemos Integrated weight (pb) y lo reemplacemos por vacio ('')
		#! sed por definicion considera un filename. Con los <<<  decimos a sed que no lo considere como filename, si no que lo tome como el string sobre el cual debemos trabajar
		cross=$(sed 's|\#|''|g' <<<"$cross")
		cross=$(sed 's|\: |''|g' <<<"$cross")
		file_mc="${file_mc/_banner.txt/''}"
		file_mc="${file_mc/$run/''}"
		echo "${file_mc}	${cross}" >> "${1}/scripts_2208/data/cross_section.dat"
		echo "${file_mc}        ${cross}" >> "/Collider/2023_LLHN_CONCYTEC/cross_section.dat"
		cd ..
	done
done
cp ${1}/scripts_2208/data/cross_section.dat /Collider/2023_LLHN_CONCYTEC/
