#!/bin/bash

#Plot D_abs and D_arg in TBTKResults.h5
python ${TBTK_dir}/TBTK/view/python/plot2D.py TBTKResults.h5 D_abs
python ${TBTK_dir}/TBTK/view/python/plot2D.py TBTKResults.h5 D_arg

#Plot magnetization
python ${TBTK_dir}/TBTK/view/python/plotMAG.py TBTKResults.h5

#Plot spin-polarized local density of states for (theta,phi) = (0, 0), using
#Gaussian smoothing with sigma = 0.02
#python ${TBTK_dir}/TBTK/view/python/plotSP_LDOS.py TBTKResults.h5 0 0 0.02
