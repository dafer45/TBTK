/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Basic Chebyshev example
 *
 *  Basic example of using the Chebyshev method to solve a 2D tight-binding
 *  model with t = 1 and mu = -1. Lattice with edges and a size of 100x100
 *  sites. Using 500 Chebyshev coefficients and evaluating the Green's function
 *  with an energy resolution of 1000. Calculates LDOS along the line y =
 *  SIZE_Y/2 = 20.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Model.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 100;
	const int SIZE_Y = 100;

	//Model parameters.
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model << HoppingAmplitude(
					-mu,
					{x,	y,	s},
					{x,	y,	s}
				);

				//Add hopping parameters corresponding to t
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{(x+1)%SIZE_X,	y,	s},
						{x,		y,	s}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{x,	(y+1)%SIZE_Y,	s},
						{x,	y,		s}
					) + HC;
				}
			}
		}
	}

	//Construct model
	model.construct();

	//Chebyshev expansion parameters.
	const int NUM_COEFFICIENTS = 500;
	const int ENERGY_RESOLUTION = 1000;
	const double SCALE_FACTOR = 5.;

	//Setup Solver::ChebyshevExpander
	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setCalculateCoefficientsOnGPU(false);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(true);
	solver.setNumCoefficients(NUM_COEFFICIENTS);

	//Set up the PropertyExtractor.
	PropertyExtractor::ChebyshevExpander propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		-SCALE_FACTOR,
		SCALE_FACTOR,
		ENERGY_RESOLUTION
	);

	//Extract local density of states (LDOS).
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, SIZE_Y/2, IDX_SUM_ALL}
	});

	//Plot the LDOS.
	Plotter plotter;
	plotter.plot({_a_, SIZE_Y/2, IDX_SUM_ALL}, ldos);
	plotter.save("figures/LDOS.png");

	return 0;
}
