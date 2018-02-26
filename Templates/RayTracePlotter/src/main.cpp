/* Copyright 2017 Kristofer Björnson
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
 *  @brief Hexagonal lattice for ray tracing demo
 *
 *  Sets up and diagonalizes a hexagonal lattice. Specifies the lattice
 *  geometry and calculates the density and wave function for state 64, 67, and
 *  73. In combination with plot.sh, this demonstrates hom to calculate
 *  properties such that they can be plotted using the ray tracing plotter
 *  'TBTKRayTracePlotter'.
 *
 *  @author Kristofer Björnson
 */

#include "Solver/Diagonalizer.h"
#include "DOS.h"
#include "PropertyExtractor/Diagonalizer.h"
#include "EigenValues.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "Model.h"
#include "WaveFunctions.h"

#include <complex>
#include <iostream>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 10;
	const int SIZE_Y = 10;

	//Parameters
	complex<double> mu = 0.0;
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
				if(x == SIZE_X/2){
					model << HoppingAmplitude(-4.,	{x, y, 0, s}, {x, y, 0, s});
					model << HoppingAmplitude(-4.,	{x, y, 1, s}, {x, y, 1, s});
					model << HoppingAmplitude(-4.,	{x, y, 2, s}, {x, y, 2, s});
					model << HoppingAmplitude(-4.,	{x, y, 3, s}, {x, y, 3, s});
				}

				//Add hopping parameters corresponding to t
				model << HoppingAmplitude(-t,		{x, y, 1, s},			{x, y, 0, s}) + HC;
				model << HoppingAmplitude(-t,		{x, y, 2, s},			{x, y, 1, s}) + HC;
				model << HoppingAmplitude(-t,		{x, y, 3, s},			{x, y, 2, s}) + HC;
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, 0, s},	{x, y, 3, s}) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, 0, s},	{x, y, 1, s}) + HC;
					model << HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, 3, s},	{x, y, 2, s}) + HC;
				}
			}
		}
	}

	//Construct model
	model.construct();
	model.constructCOO();

	//Create Geometry
	model.createGeometry(3, 0);
	Geometry *geometry = model.getGeometry();
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				double X = x*3.;
				double Y = y*2.*sqrt(3.)/2.;
				double Z = 0.;
				geometry->setCoordinates(
					{x,	y,	0,	s},
					{
						X + 0.,
						Y + 0.,
						Z
					}
				);
				geometry->setCoordinates(
					{x,	y,	1,	s},
					{
						X + 1/2.,
						Y + sqrt(3.)/2.,
						Z
					}
				);
				geometry->setCoordinates(
					{x,	y,	2,	s},
					{
						X + 3/2.,
						Y + sqrt(3.)/2.,
						Z
					}
				);
				geometry->setCoordinates(
					{x,	y,	3,	s},
					{
						X + 2.,
						Y + 0.,
						Z
					}
				);
			}
		}
	}

	//Setup and run Solver::Diagonalizer
	Solver::Diagonalizer dSolver;
	dSolver.setModel(model);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	FileWriter::writeModel(model);

	//Create PropertyExtractor
	PropertyExtractor::Diagonalizer pe(dSolver);

	//Calculate Density
	Property::Density density = pe.calculateDensity({
		{IDX_ALL,	IDX_ALL,	IDX_ALL,	IDX_SUM_ALL}
	});
	FileWriter::writeDensity(density);

	//Calculate spin up wave function for all sites and state 64, 67, and
	//73
	Property::WaveFunctions waveFunctionsUp = pe.calculateWaveFunctions(
		{
			{IDX_ALL,	IDX_ALL,	IDX_ALL,	0}
		},
		{64, 67, 73}
	);
	FileWriter::writeWaveFunctions(waveFunctionsUp, "WaveFunctionsUp");

	//Calculate spin down wave function for all sites and state 64, 67, and
	//73
	Property::WaveFunctions waveFunctionsDown = pe.calculateWaveFunctions(
		{
			{IDX_ALL,	IDX_ALL,	IDX_ALL,	1}
		},
		{64, 67, 73}
	);
	FileWriter::writeWaveFunctions(waveFunctionsDown, "WaveFunctionsDown");

	return 0;
}
