#!/bin/bash

#Plot LDOS in TBTKResults.h5, using gaussian smoothing with sigma=0.05
python ${TBTK_dir}/TBTK/view/python/plotLDOS.py TBTKResults.h5 0.05
