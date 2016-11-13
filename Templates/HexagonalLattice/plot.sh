#!/bin/bash

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.1
TBTKPlotDOS.py TBTKResults.h5 0.1

#Plot eigenvalues
TBTKPlotEV.py TBTKResults.h5
