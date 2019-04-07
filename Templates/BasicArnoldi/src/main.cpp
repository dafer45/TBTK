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

#include "TBTK/FileWriter.h"
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Timer.h"

#include <complex>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	Timer::tick("Full calculation");
	//Lattice size
	const int SIZE_X = 80;
	const int SIZE_Y = 80;

	//Parameters
	complex<double> mu = -1.0 + 0.0000001;
	complex<double> t = 1.0;
	complex<double> J = 0.25;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model << HoppingAmplitude(
					-mu + J*(1/2.-s),
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

	//Setup and run Solver::ArnoldiIterator
	const int NUM_EIGEN_VALUES = 1600;
	const int NUM_LANCZOS_VECTORS = 3200;
	int MAX_ITERATIONS = 4000;
	Solver::ArnoldiIterator aSolver;
	aSolver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	aSolver.setModel(model);
	aSolver.setCentralValue(1.0);
	aSolver.setNumEigenValues(NUM_EIGEN_VALUES);
	aSolver.setCalculateEigenVectors(true);
	aSolver.setNumLanczosVectors(NUM_LANCZOS_VECTORS);
	aSolver.setMaxIterations(MAX_ITERATIONS);
	aSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor
	PropertyExtractor::ArnoldiIterator pe(aSolver);

	//Setup energy window
	const double UPPER_BOUND = 1.25;
	const double LOWER_BOUND = 0.75;
	const int RESOLUTION = 1000;
	pe.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, RESOLUTION);

	//Extract eigenvalues and write these to file
	Property::EigenValues ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev);

	//Extract DOS and write to file
	Property::DOS dos = pe.calculateDOS();
	FileWriter::writeDOS(dos);

	//Extract LDOS and write to file
	Property::LDOS ldos = pe.calculateLDOS(
		{IDX_X,		SIZE_Y/2,	IDX_SUM_ALL},
		{SIZE_X,	1,		2}
	);
	FileWriter::writeLDOS(ldos);

	//Extract spin-polarized LDOS and write to file
	Property::SpinPolarizedLDOS spLdos = pe.calculateSpinPolarizedLDOS(
		{IDX_X,		SIZE_Y/2,	IDX_SPIN},
		{SIZE_X,	1,		2}
	);
	FileWriter::writeSpinPolarizedLDOS(spLdos);

	Timer::tock();

	return 0;
}
