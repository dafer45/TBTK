module load gcc/4.9
module load cuda/7.0

TBTK_dir=$PWD
export TBTK_dir

if [ -z "$CPLUS_INCLUDE_PATH" ]
then
	CPLUS_INCLUDE_PATH=${TBTK_dir}/Lib/include/Builders;
else
	CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Builders;
fi
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Core;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Properties;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/PropertyExtractors;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Solvers;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/StatesAndOperators;
CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include/Utilities;
export CPLUS_INCLUDE_PATH

if [ -z "$LIBRARY_PATH" ]
then
	LIBRARY_PATH=${TBTK_dir}/Lib/build;
else
	LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
fi
export LIBRARY_PATH

if [ -z "$LD_LIBRARY_PATH" ]
then
	LD_LIBRARY_PATH=${TBTK_dir}/hdf5/hdf5-build/hdf5/lib;
	LD_LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
else
	LD_LIBRARY_PATH+=:${TBTK_dir}/hdf5/hdf5-build/hdf5/lib;
	LD_LIBRARY_PATH+=:${TBTK_dir}/Lib/build;
fi
export LD_LIBRARY_PATH

if [ -z "$PATH" ]
then
	PATH=${TBTK_dir}/Tools/bin;
	PATH+=:${TBTK_dir}/Visualization/python;
else
	PATH+=:${TBTK_dir}/Tools/bin;
	PATH+=:${TBTK_dir}/Visualization/python;
fi
export PATH

#if [ -z "$MANPATH" ]
#then
#	MANPATH=${TBTK_dir}/Tools/man;
#else
	MANPATH+=:${TBTK_dir}/Tools/man;
#fi
export MANPATH
