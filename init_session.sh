module load gcc/4.9
module load cuda/7.0

TBTK_dir=$PWD
export TBTK_dir

LD_LIBRARY_PATH+=:${TBTK_dir}/hdf5/hdf5-build/hdf5/lib
export LD_LIBRARY_PATH
