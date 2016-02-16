#!/bin/bash

#Plot D_abs and D_arg in TBTKResults.h5
python ${TBTK_dir}/TBTK/view/python/plot2D.py TBTKResults.h5 D_abs
python ${TBTK_dir}/TBTK/view/python/plot2D.py TBTKResults.h5 D_arg
python ${TBTK_dir}/TBTK/view/python/plotMAG.py TBTKResults.h5
