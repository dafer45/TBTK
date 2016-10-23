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
 *  mu = -1. Lattice with edges and a size of 20x20 sites.
 *
 *  @author Kristofer Björnson
 */

#include <complex>
#include "Model.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"
#include "FileWriter.h"
#include "EigenValues.h"
#include "DOS.h"

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 20;
	const int SIZE_Y = 20;

	//Parameters
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude(-mu,	{x, y, s},	{x, y, s}));

				//Add hopping parameters corresponding to t
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, s},	{x, y, s}));
				if(y+1 < SIZE_Y)
					model.addHAAndHC(HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, s},	{x, y, s}));
			}
		}
	}

	//Construct model
	model.construct();

	//Setup and run DiagonalizationSolver
	DiagonalizationSolver dSolver;
	dSolver.setModel(&model);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor
	DPropertyExtractor pe(&dSolver);

	//Extract eigenvalues and write these to file
	Property::EigenValues *ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev);
	delete ev;

	//Extract DOS and write to file
	const double UPPER_LIMIT = 6.;
	const double LOWER_LIMIT = -4.;
	const int RESOLUTION = 1000;
	Property::DOS *dos = pe.calculateDOS(LOWER_LIMIT, UPPER_LIMIT, RESOLUTION);
	FileWriter::writeDOS(dos);
	delete dos;

	return 0;
}
