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
 *  @brief Basic Arnoldi example
 *
 *  Basic example of Arnoldi iteration of a 2D tight-binding model with t = 1,
 *  mu = -1, and J = 0.25. Lattice with edges and a size of 40x40 sites.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/TBTK.h"
#include "TBTK/Timer.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	Timer::tick("Full calculation");

	//Lattice size.
	const int SIZE_X = 40;
	const int SIZE_Y = 40;

	//Parameters.
	complex<double> mu = -1.0 + 0.0000001;
	complex<double> t = 1.0;
	complex<double> J = 0.25;

	//Create model and set up hopping parameters.
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to
				//chemical potential.
				model << HoppingAmplitude(
					-mu + J*(1/2.-s),
					{x,	y,	s},
					{x,	y,	s}
				);

				//Add hopping parameters corresponding to t.
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

	//Construct model.
	model.construct();

	//Setup and run Solver::ArnoldiIterator.
	const int NUM_EIGEN_VALUES = 400;
	const int NUM_LANCZOS_VECTORS = 800;
	int MAX_ITERATIONS = 2000;
	Solver::ArnoldiIterator solver;
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setModel(model);
	solver.setCentralValue(1.0);
	solver.setNumEigenValues(NUM_EIGEN_VALUES);
	solver.setCalculateEigenVectors(true);
	solver.setNumLanczosVectors(NUM_LANCZOS_VECTORS);
	solver.setMaxIterations(MAX_ITERATIONS);
	solver.run();

	//Create PropertyExtractor.
	PropertyExtractor::ArnoldiIterator propertyExtractor(solver);

	//Setup energy window.
	const double UPPER_BOUND = 1.25;
	const double LOWER_BOUND = 0.75;
	const int RESOLUTION = 1000;
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Extract eigenvalues and write these to file.
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	//Plot the eigenvalues.
	Plotter plotter;
	plotter.plot(eigenValues);
	plotter.save("figures/EigenValues.png");

	//Extract the density of states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.01;
	const unsigned int SMOOTHING_WINDOW = 201;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	plotter.clear();
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Extract the local density of states (LDOS).
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, _a_, IDX_SUM_ALL}
	});

	//Smooth the LDOS.
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the LDOS.
	plotter.clear();
	plotter.setNumContours(100);
	plotter.plot({_a_, SIZE_Y/2, IDX_SUM_ALL}, ldos);
	plotter.save("figures/LDOS.png");

	//Extract the spin-polarized LDOS.
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, _a_, IDX_SPIN}
		});

	//Smooth the spin-polarized LDOS.
	spinPolarizedLDOS = Smooth::gaussian(
		spinPolarizedLDOS,
		SMOOTHING_SIGMA,
		SMOOTHING_WINDOW
	);

	//Plot the spin-polarized LDOS along the z-axis and for y=SIZE_Y/2.
	plotter.clear();
	plotter.setNumContours(100);
	plotter.plot({_a_, SIZE_Y/2, IDX_SPIN}, {0, 0, 1}, spinPolarizedLDOS);
	plotter.save("figures/SpinPolarizedLDOS.png");

	Timer::tock();

	return 0;
}
