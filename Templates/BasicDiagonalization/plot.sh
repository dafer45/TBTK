#!/bin/bash

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.15
TBTKPlotDOS.py TBTKResults.h5 0.15

#Plot eigenvalues
TBTKPlotEV.py TBTKResults.h5
