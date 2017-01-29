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
 *  @brief Self-consistent superconductivity using the Chebyshev method
 *
 *  Self-consistent calculation of the superconducting order-parameter for a 2D
 *  tight-binding model with t = 1, mu = -1, and V_sc = 2. Lattice with edges
 *  and a size of 40x40 sites. The calculation is implemented such that only a
 *  restricted environment of the lattice point at which the order parameter is
 *  calculated at is included in the local model. This allows for a larger
 *  system to be calculated as te computational time scale as O(N) rather than
 *  O(N^2) as a function of the number of lattice sites once the radious of the
 *  environment is fixed.
 *
 *  @author Kristofer Björnson
 */

#include "ChebyshevSolver.h"
#include "CPropertyExtractor.h"
#include "FileWriter.h"
#include "Model.h"
#include "Timer.h"

#include <chrono>
#include <complex>
#include <iostream>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

//Lattice size
const int SIZE_X = 20;
const int SIZE_Y = 20;

//Order parameter. The two buffers are alternatively swaped by setting dCounter
// = 0 or 1. One buffer contains the order parameter used in the previous
//calculation, while the other is used to store the newly obtained order
//parameter that will be used in the next calculation.
complex<double> D[2][SIZE_X][SIZE_Y];
int dCounter = 0;

//ChebyshevSolver parameters, SCALE_FACTOR scales the energy spectrum to lie
//lie within -1 < E < 1, while NUM_COEFFICIENTS and ENERGY_RESOLUTION are the
//number of Chebyshev coefficients used in the expansion of the Green's
//function, and the number of points use to evaluate the Green's function,
//respectively.
const double SCALE_FACTOR = 10.;
const int NUM_COEFFICIENTS = 5000;
const int ENERGY_RESOLUTION = 10000;

//Superconducting pair potential, convergence limit, max iterations, initial
//guess, weight factor with which the old order parameter is mixed with the
//newly calculated one, Debye frequency used to specify the integration
//limit used to calculate the superconducting order parameter, and radius used
//to restrict the size of the environment included in the local model used when
//calculating the order parameter at a site.
const double V_sc = 2.;
const double CONVERGENCE_LIMIT = 0.0001;
const int MAX_ITERATIONS = 50;
const complex<double> D_INITIAL_GUESS = 0.3;
const double SC_WEIGHT_FACTOR = 0.5;
const double DEBYE_FREQUENCY = 10.;
const double SC_MODEL_RADIUS = 15.;

//Callback function responsible for determining the value of the order
//parameter D_{to,from}c_{to}c_{from} where to and from are indices of the form
//(x, y, spin).
complex<double> fD(Index to, Index from){
	//Obtain indices
	int x = from.at(0);
	int y = from.at(1);
	int s = from.at(2);

	//Return appropriate amplitude
	switch(s){
		case 0:
			return conj(D[dCounter][x][y]);
		case 1:
			return -conj(D[dCounter][x][y]);
		case 2:
			return -D[dCounter][x][y];
		case 3:
			return D[dCounter][x][y];
		default://Never happens
			return 0;
	}
}

//Function responsible for setting up the local model Hamiltonian, only sites
//at most a distance 'radius' away from the coordinates 'center_x' and
//'center_y' are included.
Model* setupModel(int center_x, int center_y, double radius){
	//Parameters
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model *model = new Model();
	model->setTalkative(false);	//Limit the amount of text written to the output at model creation
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			//Exclude sites at a distance further than 'radius'
			//away from the center points
			if((x - center_x)*(x - center_x) + (y - center_y)*(y - center_y) > radius*radius)
				continue;

			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model->addHA(HoppingAmplitude(-mu,	{x, y, s},		{x, y, s}));
				model->addHA(HoppingAmplitude(mu,	{x, y, s+2},	{x, y, s+2}));

				//Add hopping amplitudes corresponding to t.
				//Outer if-statements guards against the
				//addition of hopping amplitudes to sites
				//further away from the center than a distance
				//'radius'.
				if((x+1 - center_x)*(x+1 - center_x) + (y - center_y)*(y - center_y) <= radius*radius){
					if(x+1 < SIZE_X){
						model->addHAAndHC(HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, s},	{x, y, s}));
						model->addHAAndHC(HoppingAmplitude(t,	{(x+1)%SIZE_X, y, s+2},	{x, y, s+2}));
					}
				}
				if((x - center_x)*(x - center_x) + (y+1 - center_y)*(y+1 - center_y) <= radius*radius){
					if(y+1 < SIZE_Y){
						model->addHAAndHC(HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, s},	{x, y, s}));
						model->addHAAndHC(HoppingAmplitude(t,	{x, (y+1)%SIZE_Y, s+2},	{x, y, s+2}));
					}
				}

				//Add hopping amplitudes corresponding to the
				//superconducting order parameter.
				model->addHAAndHC(HoppingAmplitude(fD,	{x, y, 3-s},	{x, y, s}));
			}
		}
	}

	//Construct model
	model->construct();
	model->constructCOO();

	return model;
}

//Self-consistency loop
double scLoop(double radius){
	//Setup CPropertyExtractor using GPU accelerated generation of
	//coefficients and Green's function, and using lookup table. The
	//Green's function is only calculated in an energy interval around E=0
	//with a width twice the Debye frequency.
	ChebyshevSolver cSolver;
	cSolver.setScaleFactor(SCALE_FACTOR);
	CPropertyExtractor pe(
		&cSolver,
		NUM_COEFFICIENTS,
		true,
		false,
		true
	);
	pe.setEnergyWindow(
		-DEBYE_FREQUENCY,
		DEBYE_FREQUENCY,
		ENERGY_RESOLUTION
	);

	//Self-consistency loop
	int counter = 0;
	double maxError = 0.;
	while(counter++ < MAX_ITERATIONS){
		//Time each step (Not essential, but useful because the
		//calculation takes some time). See corresponding call
		//to Timer:tock() at the end of the loop.
		Timer::tick("Self-consistency iteration");

		//Clear the order parameter
		for(int x = 0; x < SIZE_X; x++){
			for(int y = 0; y < SIZE_Y; y++){
				D[(dCounter+1)%2][x][y] = 0.;
			}
		}

		//Calculate D(x, y) = <c_{x, y, \downarrow}c_{x, y, \uparrow}>
		for(int x = 0; x < SIZE_X; x++){
			for(int y = 0; y < SIZE_Y; y++){
				//Setup local model
				Model *model = setupModel(x, y, radius);
				cSolver.setModel(model);

				//Calculate anomalous Green's function
				complex<double> *greensFunction = pe.calculateGreensFunction({x, y, 3}, {x, y, 0});

				//Free memory occupied by local model
				delete model;

				//Calculate order parameter
				for(int n = 0; n < ENERGY_RESOLUTION/2; n++){
					const double dE = 2.*DEBYE_FREQUENCY/(double)ENERGY_RESOLUTION;
					D[(dCounter+1)%2][x][y] -= V_sc*i*greensFunction[n]*dE/M_PI;
				}

				//Free memory used for Green's function
				delete [] greensFunction;

				//Mix old and new order parameter
				D[(dCounter+1)%2][x][y] = (1 - SC_WEIGHT_FACTOR)*D[(dCounter+1)%2][x][y] + SC_WEIGHT_FACTOR*D[dCounter][x][y];
			}
		}

		//Swap order parameter buffers
		dCounter = (dCounter+1)%2;

		//Calculate convergence parameter
		maxError = 0.;
		for(int x = 0; x < SIZE_X; x++){
			for(int y = 0; y < SIZE_Y; y++){
				double error = 0.;
				if(abs(D[dCounter][x][y]) == 0 && abs(D[(dCounter+1)%2][x][y]) == 0)
					error = 0.;
				else
					error = abs(D[dCounter][x][y] - D[(dCounter+1)%2][x][y])/max(abs(D[dCounter][x][y]), abs(D[(dCounter+1)%2][x][y]));

				if(error > maxError)
					maxError = error;
			}
		}

		//Output time since Timer:tick()-call at the beginning of
		// the iteration.
		Timer::tock();

		//Exit the self-consistency loop depending on whether the
		//result has converged or not
		if(maxError < CONVERGENCE_LIMIT)
			break;
	}

	return maxError;
}

//Function responsible for initializing the order parameter
void initD(){
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			D[0][x][y] = D_INITIAL_GUESS;
		}
	}
}

int main(int argc, char **argv){
	//Parameters
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Initialize D
	initD();

	//Run self-consistency loop
	double convergenceParameter = scLoop(SC_MODEL_RADIUS);

	//Calculate abs(D) and arg(D)
	double D_abs[SIZE_X*SIZE_Y];
	double D_arg[SIZE_X*SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			D_abs[x*SIZE_Y + y] = abs(D[dCounter][x][y]);
			D_arg[x*SIZE_Y + y] = arg(D[dCounter][x][y]);
		}
	}

	//Save abs(D) and arg(D) in "D_abs" and "D_arg"
	const int D_RANK = 2;
	int dDims[D_RANK] = {SIZE_X, SIZE_Y};
	stringstream ss;
	ss.str("");
	ss << "D_abs_";
	FileWriter::write(D_abs, D_RANK, dDims, ss.str());
	ss.str("");
	ss << "D_arg_";
	FileWriter::write(D_arg, D_RANK, dDims, ss.str());
	ss.str("");
	ss << "attributes";
	const int NUM_ATTRIBUTES = 1;
	double attributes[NUM_ATTRIBUTES];
	attributes[0] = convergenceParameter;
	string attributeNames[NUM_ATTRIBUTES];
	attributeNames[0] = "convergenceParameter";
	FileWriter::writeAttributes(attributes, attributeNames, NUM_ATTRIBUTES, ss.str());

	return 0;
}
