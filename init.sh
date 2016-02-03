module load gcc/4.9
module load cuda/7.0

LD_LIBRARY_PATH+=":$PWD/hdf5/hdf5-build/hdf5/lib"
LD_LIBRARY_PATH+=":$PWD/TBTK/calc/TightBindingLib/build"

LIBRARY_PATH+="$PWD/TBTK/calc/TightBindingLib/build"
export LIBRARY_PATH

CPLUS_INCLUDE_PATH="/$PWD/TBTK/calc/TightBindingLib/include"
export CPLUS_INCLUDE_PATH
