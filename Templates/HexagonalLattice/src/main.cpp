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
 *  @brief Hexagonal lattice using diagonalization
 *
 *  Basic example of diagonalization of a 2D tight-binding model with t = 1 and
 *  mu = 0. Hexagonal lattice with edges, four sites per unit cell, and 10x10
 *  unit cells.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Model.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>
#include <iostream>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Lattice size.
	const int SIZE_X = 10;
	const int SIZE_Y = 10;

	//Parameters.
	double mu = 0.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters.
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping parameters corresponding to t.
				model << HoppingAmplitude(
					-t,
					{x, y, 1, s},
					{x, y, 0, s}
				) + HC;
				model << HoppingAmplitude(
					-t,
					{x, y, 2, s},
					{x, y, 1, s}
				) + HC;
				model << HoppingAmplitude(
					-t,
					{x, y, 3, s},
					{x, y, 2, s}
				) + HC;
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{(x+1)%SIZE_X, y, 0, s},
						{x, y, 3, s}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{x, (y+1)%SIZE_Y, 0, s},
						{x, y, 1, s}
					) + HC;
					model << HoppingAmplitude(
						-t,
						{x, (y+1)%SIZE_Y, 3, s},
						{x, y, 2, s}
					) + HC;
				}
			}
		}
	}
	model.setChemicalPotential(mu);

	//Construct model.
	model.construct();

	//Setup and run Solver::Diagonalizer.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Create PropertyExtractor.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	//Setup energy window.
	const double LOWER_BOUND = -5.;
	const double UPPER_BOUND = 5.;
	const int RESOLUTION = 1000;
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Extract eigenvalues.
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	//Plot eigenvalues.
	Plotter plotter;
	plotter.plot(eigenValues);
	plotter.save("figures/EigenValues.png");

	//Extract the density fo states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.1;
	const unsigned int SMOOTHING_WINDOW = 51;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	plotter.clear();
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	return 0;
}
