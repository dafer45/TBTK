#!/bin/bash

#Plot eigenvalues
TBTKPlotEigenValues.py TBTKResults.h5

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.1
TBTKPlotDOS.py TBTKResults.h5 0.0075

#Plot LDOS in TBTKResults.h5, using gaussian smoothing with sigma=0.1
TBTKPlotLDOS.py TBTKResults.h5 0.0075

#Plot spin-polarized LDOS for spins polarization axis theta=3.14, phi=0, using
#gaussian smoothing with sigma=0.1
TBTKPlotSpinPolarizedLDOS.py TBTKResults.h5 3.14 0 0.0075
