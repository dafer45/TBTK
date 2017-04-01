#!/bin/bash

#Plot density
TBTKPlotDensity.py TBTKResults.h5

#Plot DOS using gaussian smoothing with sigma=0.15
TBTKPlotDOS.py TBTKResults.h5 0.15

#Plot eigen values
TBTKPlotEigenValues.py TBTKResults.h5

#Plot LDOS using gaussian smoothing with sigma=0.15
TBTKPlotLDOS.py TBTKResults.h5 0.15

#Plot magnetization along direction theta=0.1, phi=0.2
TBTKPlotMagnetization.py TBTKResults.h5 0.1 0.2

#Plot spin-polarized LDOS along direction theta=0.1, phi=0.2, using gaussian smoothing with sigma=0.05
TBTKPlotSpinPolarizedLDOS.py TBTKResults.h5 0.1 0.2 0.05
