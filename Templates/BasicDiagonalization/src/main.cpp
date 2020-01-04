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
 *  @brief Basic diagonalization example
 *
 *  Basic example of diagonalization of a 2D tight-binding model with t = 1 and
 *  mu = -1, and J = 1. Lattice with edges and a size of 21x21 sites.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Model.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Lattice size
	const int SIZE_X = 21;
	const int SIZE_Y = 21;

	//Parameters
	double mu = -1.0;
	complex<double> t = 1.0;
	complex<double> J = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int spin = 0; spin < 2; spin++){
				//Add the Zeeman term.
				model << HoppingAmplitude(
					J*(1. - 2*spin),
					{x, y, spin},
					{x, y, spin}
				);

				//Add hopping amplitudes corresponding to t.
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{(x+1)%SIZE_X,	y,	spin},
						{x,		y,	spin}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{x,	(y+1)%SIZE_Y,	spin},
						{x,	y,		spin}
					) + HC;
				}
			}
		}
	}
	model.setChemicalPotential(mu);

	//Construct the model.
	model.construct();

	//Setup and run the Solver::Diagonalizer.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	//Set the energy window.
	const double UPPER_BOUND = 6.;
	const double LOWER_BOUND = -6.;
	const int RESOLUTION = 1000;
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Extract Density.
	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, _a_, IDX_SUM_ALL}
	});

	//Plot the Density.
	Plotter plotter;
	plotter.plot({_a_, _a_, IDX_SUM_ALL}, density);
	plotter.save("figures/Density.png");

	//Extract the density of states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 101;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	plotter.clear();
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Extract eigenvalues.
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	//Plot the eigenvalues.
	plotter.clear();
	plotter.plot(eigenValues);
	plotter.save("figures/EigenValues.png");

	//Extract the local density of states (LDOS).
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, SIZE_Y/2, IDX_SUM_ALL}
	});

	//Smooth the LDOS.
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the LDOS.
	plotter.clear();
	plotter.plot({_a_, SIZE_Y/2, IDX_SUM_ALL}, ldos);
	plotter.save("figures/LDOS.png");

	//Extract the Magnetization.
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, IDX_SPIN}
		});

	//Plot the Magnetization along the z-axis.
	plotter.clear();
	plotter.plot({_a_, SIZE_Y/2, IDX_SPIN}, {0, 0, 1}, magnetization);
	plotter.save("figures/Magnetization.png");

	//Extract the spin-polarized LDOS.
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, SIZE_Y/2, IDX_SPIN}
		});

	//Smooth the spin-polarized LDOS.
	spinPolarizedLDOS = Smooth::gaussian(
		spinPolarizedLDOS,
		SMOOTHING_SIGMA,
		SMOOTHING_WINDOW
	);

	//Plot the spin-polarized LDOS along the z-axis.
	plotter.clear();
	plotter.plot({_a_, SIZE_Y/2, IDX_SPIN}, {0, 0, 1}, spinPolarizedLDOS);
	plotter.save("figures/SpinPolarizedLDOS.png");

	return 0;
}
