# LLP_NonPointingPhotons_ATLAS

## Overview
This repository contains the code for a generalized search of non-pointing parameters and time of flight of photons from Long-Lived Particles at the ATLAS ECal. It imports the UFO of interest, generates the param_card.dat files necesary for the analysis and executes the run of the analysis based on a specific parameter space given by a change in the copling (alpha), and the mass. 

## Features
The python analysis files edits the HepMC with a linear efficiency by overwriting the file line by line adding the parameters of interest to the photon information. With this information, deltaZ and t_gamma are calculated. Then, a .txt file with this information is created in order study the difference between delta Z given by ATLAS and the a delta Z calculated in a simple way. The main difference with this two approaches resides in the values of R1 and R2, which are the positions in the Electromagnetic Calorimeter that the photon leaves most of its energy: The ATLAS approach parametrizes this value by the pseudorapidity; meanwhile, the simpler way takes R1 and R2 as fixed.

## Acknowledgements
This work is based uppon the code developed in the [2023_LLHN_CONCYTEC](https://github.com/VelvetBucket/2023_LLHN_CONCYTEC) repository. 

## Contact
For any questions or suggestions, please open an issue or contact me at [walter.rodq@pucp.edu.pe] or the contributor [danilo.zegarra@pucp.edu.pe].

