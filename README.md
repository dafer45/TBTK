Welcome to the Tight Binding Toolkit - TBTK

###########################
# Installation instructions
###########################
module load gcc/4.9
module load cuda/7.0

bash install.sh

####################
# Initialize session
####################
module load gcc/4.9
module load cuda/7.0
LD_LIBRARY_PATH+=:/path_to_here/hdf5/hdf5-build/hdf5/lib
