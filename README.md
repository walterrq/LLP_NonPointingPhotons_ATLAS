# LLP_NonPointingPhotons_ATLAS

## Overview
This repository contains the code for a generalized search of non-pointing parameters and the time of flight of photons from Long-Lived Particles at the ATLAS ECal. It imports the UFO of interest, generates the `param_card.dat` files necessary for the analysis, and executes the analysis based on a specific parameter space defined by variations in the coupling constant (alpha) and mass.

## Features
The Python analysis files edit the HepMC data with linear efficiency by overwriting the file line by line, adding the parameters of interest to the photon information. With this information, `deltaZ` and `t_gamma` are calculated. A `.txt` file containing this information is created to study the difference between `deltaZ` given by ATLAS and `deltaZ` calculated in a simpler way. The main difference between these two approaches lies in the values of `R1` and `R2`, which are the positions in the Electromagnetic Calorimeter where the photon deposits most of its energy: 
- The ATLAS approach parameterizes this value using the pseudorapidity.
- The simpler method assumes `R1` and `R2` as fixed values.

## Acknowledgements
This work is based on the code developed in the [2023_LLHN_CONCYTEC](https://github.com/VelvetBucket/2023_LLHN_CONCYTEC) repository.

## Contact
For any questions or suggestions, please open an issue or contact me at [walter.rodriguez@pucp.edu.pe](mailto:walter.rodriguez@pucp.edu.pe) or [danilo.zegarra@pucp.edu.pe](mailto:danilo.zegarra@pucp.edu.pe).
