#!/bin/bash

#Plot DOS in TBTKResults.h5, using gaussian smoothing with sigma=0.15
TBTKPlot1D.py TBTKResults.h5 fR
TBTKPlot1D.py TBTKResults.h5 fI
TBTKPlot1D.py TBTKResults.h5 FR
TBTKPlot1D.py TBTKResults.h5 FI

#Plot eigenvalues
TBTKPlot2D.py TBTKResults.h5 gR
TBTKPlot2D.py TBTKResults.h5 gI
TBTKPlot2D.py TBTKResults.h5 GR
TBTKPlot2D.py TBTKResults.h5 GI
