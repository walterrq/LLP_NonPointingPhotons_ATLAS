#!/bin/bash

#TODO FUNCION DE ESTE ARCHIVO: este archivo se encarga de hacer un scan de distintos de param_cards. Hace un for barriendo todos los param_cards de interes y los usa

#echo "In benchs"
#echo $PWD

###?AWS

#line=$(($AWS_BATCH_JOB_ARRAY_INDEX + 1))
#benches=()
#benches+=$(sed "$line!d" benchmarks.txt)
#todo AWS_BATCH_JOB_ARRAY_INDEX es una variable por default cuando se corre una instancia en el amazon web service.  Si corremos 60 computadoras al mismo tiempo, AWS_BATCH_JOB_ARRAY_INDEX = 0 la primera vez, =1 la segunda, y asi sucesivamente
# line=1


benches=() #Inicializa lista de nombres de param_cards
#benches1=$(sed -n '131,+8p' benchmarksX.txt)
#benches1=$(sed -n '46,+2p' benchmarks.txt)
#!Por ahora solo nos interesa la simulacion sobre un punto.
benches1=$(sed -n '46,+0p' benchmarks.txt)
#todo The variable benches1 will contain the text of the first line and the second line from the benchmarks.txt file. 
#todo si bien la funcion principal de sed es reemplazar, al añadir -n pedimos que su comportamiento original cambie y solo lea las lineas que indicamos.
#!Comenzamos en la instancia 1 y una mas. Solo usamos, por ende, dos cores. si hago '1,+4p' significa que uso 5 cores: la linea 1 mas 4 lineas mas.
#* Lo que hace el sed es generar una lista con 
IFS=$'\n' benches=( $benches1 )
echo $benches1
#todo la anterior linea separa el string por newline y logra que benches sea un array separados por /n
#todo Es decir, tenemos arrays de benchmarks con dos elementos

#!Queremos generar un array de los param_cards que vamos a correr


for vars in "${benches[@]}" 
#todo Sintaxis de for: vars es una variable que tomara cada valor del benches
do
	origin="${PWD}/${vars}"  #* absolute path
	#todo origin es una nueva variable: toma el working directory y vars
	echo $origin

	#x=$(find|grep "# gNh55" "${origin}")
	#todo En esta primera linea, find|grep Estructura: encuentra la linea en donde se encuentra el pedazo de string # gNh55 en el archivo origin(param_card)
	#! find|grep da el string y se guarda en la variable x
	#sed -i "s/$x/  4 0e00  # gNh55/" "${origin}"
	#todo En esta segunda linea, reemplaza TODA la linea por 4 0e00
	#!se toma como input la variable x, que es la linea que queremos reemplazar
	
	#x=$(find|grep "# gNh56" "${origin}")
       #sed -i "s/$x/  5 2.000000e-1  # gNh56/" "${origin}"

	#todo ids: identificadores con los que diferenciaremos estos param_cards de otros param_cards
	ids=$(sed 's|.dat|''|g' <<< "$vars")
	#todo Estructura: agarra .dat y borralo(para eso se usa '') 
	#todo el g representa global
	#todo global hace que cada instancia que encuentres de esto reemplazalo, pues si no usamos g, el set por default solo reemplaza la primera que encuentre
	#! Se agarra cada param_card(vars) y le dices que de ese string, agarre .dat y lo reemplace(sed) por nada
	#todo 's indica que se haga el replace
	#todo le estoy mandando vars y la estructura es <<< porque la variable vars es una direccion
	#todo los <<< es como decir: toma esa variable (mg5launches) y no intentes leer el archivo; si no, tomalo y trabajalo como un string

	ids=$(sed 's|.*/param_card.SeesawSM|''|g' <<< "$ids")
	#todo notemos que ids es el string sin .dat y luego he borrado todo lo anterior de param_card.SeesawSM incluido esto ultimo (lo reemplazamos por nada '') y me quedo con los numeros descriptivos (eso es mi identificador(10.1)). Son tags.

	#todo debuggeo
	echo $ids

	#todo string split usando el punto
	IFS="." read -r -a array <<< "$ids"
	#todo IFS actua como un string split: corta la palabra en el caracter que yo desee
	#todo IFS separa usando el punto como separador. Luego lo vuelve un array llamado array

	#! Luegoa asignamos las etiquetas o id(identifier) a los componentes de este array. El primer componente del array hace referencia a la masa; el segundo, al valor alfa
	
	mass=${array[0]}
	alpha=${array[1]}

	echo "$mass $alpha"

	#TODO como ultimo paso, llamamos a param_distZH y recibe los inputs M, Alpha, etc
	#TODO Si suponemos que mass=10 y alpha=3, se le esta creando los strings M10, Alpha3, solo para indicar que numero es que. Creo un label para alpha y la masa.
	bash param_distZH.sh "" "M${mass}" "Alpha${alpha}" "${origin}" "$1" "$2"
	#! "$1" es el input 1 y "$2" es el input 2. Recordamos que en masterZH se definieron estos al llamar a benchZH con bash: bash benchsZH.sh "$x1" "$madgraph_folder". El input 1 es el numero de eventos y el input 2 es la direccion de madgraph.

	#! Aca, siempre los inputs descritos al llamar a aun archivo se describen dentro del archivo de entrada como $n. Por ejemplo, en master se pusieron dos inputs para el bash de benches, esos dos inputs son $1 y $2 en orden. Lo mismo sera para param_dist; se tendra #1,#2,...,#5

done

