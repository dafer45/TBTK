#!/bin/bash

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.1
python ${TBTK_dir}/Visualization/python/plotDOS.py TBTKResults.h5 0.1

#Plot eigenvalues
python ${TBTK_dir}/Visualization/python/plotEV.py TBTKResults.h5
