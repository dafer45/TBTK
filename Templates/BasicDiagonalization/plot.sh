#!/bin/bash

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.15
python ${TBTK_dir}/TBTK/view/python/plotDOS.py TBTKResults.h5 0.15
