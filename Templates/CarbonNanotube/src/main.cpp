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
 *  @brief Carbon nanotube using diagonalization
 *
 *  Basic example of diagonalization of a 2D tight-binding model with t = 1 and
 *  mu = 0. Carbon nanotube with 4*SIZE_X=8 atoms circumference, and SIZE_Y=20
 *  atoms long.
 *
 *  @author Kristofer Björnson
 */

#include "Solver/Diagonalizer.h"
#include "DOS.h"
#include "DPropertyExtractor.h"
#include "EigenValues.h"
#include "FileWriter.h"
#include "Model.h"

#include <complex>
#include <iostream>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 2;
	const int SIZE_Y = 20;

	//Parameters
	complex<double> mu = 0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model << HoppingAmplitude(-mu,	{x, y, 0, s},	{x, y, 0, s});
				model << HoppingAmplitude(-mu,	{x, y, 1, s},	{x, y, 1, s});
				model << HoppingAmplitude(-mu,	{x, y, 2, s},	{x, y, 2, s});
				model << HoppingAmplitude(-mu,	{x, y, 3, s},	{x, y, 3, s});

				//Add hopping parameters corresponding to t
				model << HoppingAmplitude(-t,		{x, y, 1, s},			{x, y, 0, s}) + HC;
				model << HoppingAmplitude(-t,		{x, y, 2, s},			{x, y, 1, s}) + HC;
				model << HoppingAmplitude(-t,		{x, y, 3, s},			{x, y, 2, s}) + HC;
				model << HoppingAmplitude(-t,		{(x+1)%SIZE_X, y, 0, s},	{x, y, 3, s}) + HC;
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, 0, s},	{x, y, 1, s}) + HC;
					model << HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, 3, s},	{x, y, 2, s}) + HC;
				}
			}
		}
	}

	//Construct model
	model.construct();

	//Setup and run Solver::Diagonalizer
	Solver::Diagonalizer dSolver;
	dSolver.setModel(model);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor
	DPropertyExtractor pe(dSolver);

	//Setup energy window
	const double LOWER_BOUND = -5.;
	const double UPPER_BOUND = 5.;
	const int RESOLUTION = 1000;
	pe.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, RESOLUTION);

	//Extract eigenvalues and write these to file
	Property::EigenValues ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev);

	//Extract DOS and write to file
	Property::DOS dos = pe.calculateDOS();
	FileWriter::writeDOS(dos);

	return 0;
}
