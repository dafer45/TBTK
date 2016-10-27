[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)

Welcome to the Tight Binding Toolkit - TBTK

For an introduction to the library, see http://dafer45.github.io/TBTK

###########################
# Installation instructions
###########################
Without CUDA (gpu)  
source init_session.sh  
./install.sh

With CUDA (gpu)  
source init_session.sh  
./install -CUDA

#############
# Update TBTK
#############
After pulling the latest version of TBTK, execute the following command from the TBTK root folder to update the library:  
./update.sh

####################
# Initialize session
####################
Each time a new terminal session is opened, execute the following command from the TBTK root folder:  
source init_session.sh

#########
# License
#########
TBTK is free to use under the Appache 2.0 license (see the file 'License'). Please give attribution in accordance with the 'Cite' section below.

######
# Cite
######
####To cite TBTK, mention TBTK in the text and cite the DOI for this project.  
[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)  
Kristofer Björnson, & glaurung24. (2016). dafer45/TBTK: Initial release [Data set].  
Zenodo. http://doi.org/10.5281/zenodo.162730

####If you use the ChebyshevSolver to produce results, please also cite the following references  
A. Weiße, G. Wellein, A. Alvermann, and H. Fehske,  
Rev. Mod. Phys. 78, 275 (2006).

L. Covaci, F. M. Peeters, and M. Berciu,  
Phys. Rev. Lett. 105, 167006 (2010).
