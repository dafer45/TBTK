#!/bin/bash

#Plot wave function
TBTKRayTracePlotter --property WaveFunction --wave-function-state 64 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionUp --output WaveFunctionUp64.png
TBTKRayTracePlotter --property WaveFunction --wave-function-state 67 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionUp --output WaveFunctionUp67.png
TBTKRayTracePlotter --property WaveFunction --wave-function-state 73 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionUp --output WaveFunctionUp73.png

TBTKRayTracePlotter --property WaveFunction --wave-function-state 64 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionDown --output WaveFunctionDown64.png
TBTKRayTracePlotter --property WaveFunction --wave-function-state 67 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionDown --output WaveFunctionDown67.png
TBTKRayTracePlotter --property WaveFunction --wave-function-state 73 --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800 --input WaveFunctionDown --output WaveFunctionDown73.png

#Plot Density
TBTKRayTracePlotter --property Density --position "(15, 8, 40)" --focus "(15, 8, 0)" --radius 0.5 --width 1200 --height 800
