##########################################################
# Check that gcc 4.9.0 or above is the current compiler. #
##########################################################
gcc_version="$(gcc -dumpversion)"
gcc_required_version="4.9.0"
if [ "$(printf "$gcc_required_version\n$gcc_version" | sort -V | head -n1)" == "$gcc_version" ] && [ "$gcc_version" != "$gcc_required_version" ]; then
        echo "Error: Unable to detect gcc compiler with minimum version 4.9."
        return
fi

####################################################################
# Check wether nvcc is available and print warning message if not. #
####################################################################
if hash nvcc 2>/dev/null; then
	:
else
        echo "Warning: No nvcc compiler found. CUDA not supported."
fi

#module load gcc/4.9
#module load cuda/7.0

TBTK_dir=$PWD
export TBTK_dir

#################
# Include paths #
#################
# Native TBTK
if [ -z "$CPLUS_INCLUDE_PATH" ]
then
	CPLUS_INCLUDE_PATH=${TBTK_dir}/Lib/include/Builders;
else
	CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Builders;
fi
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Elements;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Core;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Lattices;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Lattices/D2;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Lattices/D3;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/LinearAlgebra;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/ManyBody;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/ManyBody/FockStateMap;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/ManyBody/FockStateRule;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Properties;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/PropertyExtractors;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Solvers;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/StatesAndOperators;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Utilities;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Uncategorized;
# Third party libraries
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/json;
export CPLUS_INCLUDE_PATH

#########################
# Build time link paths #
#########################
if [ -z "$LIBRARY_PATH" ]
then
	LIBRARY_PATH=${TBTK_dir}/Lib/build;
else
	LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
fi
export LIBRARY_PATH

#######################
# Run time link paths #
#######################
if [ -z "$LD_LIBRARY_PATH" ]
then
	LD_LIBRARY_PATH=${TBTK_dir}/hdf5/hdf5-build/hdf5/lib;
	LD_LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
else
	LD_LIBRARY_PATH+=:${TBTK_dir}/hdf5/hdf5-build/hdf5/lib;
	LD_LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
fi
export LD_LIBRARY_PATH

################
# Binary paths #
################
if [ -z "$PATH" ]
then
	PATH=${TBTK_dir}/Tools/bin;
else
	PATH+=:${TBTK_dir}/Tools/bin;
fi
PATH+=:${TBTK_dir}/Visualization/python;
export PATH

##################
# Man page paths #
##################
MANPATH+=:${TBTK_dir}/Tools/man;
export MANPATH
