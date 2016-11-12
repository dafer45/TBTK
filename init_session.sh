module load gcc/4.9
module load cuda/7.0

TBTK_dir=$PWD
export TBTK_dir

if [ -z "$CPLUS_INCLUDE_PATH" ]
then
	CPLUS_INCLUDE_PATH=${TBTK_dir}/Lib/include;
else
	CPLUS_INCLUDE_PATH+=:${TBTK_dir}/Lib/include;
fi
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
	PATH+=:${TBTK_dir}/View/python;
else
	PATH+=:${TBTK_dir}/Tools/bin;
	PATH+=:${TBTK_dir}/View/python;
fi
export PATH
