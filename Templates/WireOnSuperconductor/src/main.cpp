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
 *  @brief Wire on superconductor using self-consistent diagonalization
 *
 *  Self-consistent solution for superconducting order-parameter for a 2D
 *  tight-binding model with t_ss = 1, mu = -1, and V_sc = 2. Lattice with
 *  edges and a size of 36x19 sites. A ferromagnetic wire of length 18 is
 *  attached on top of the superconductor with t_mm = 0.75 inside the wire, and
 *  t_sm = 0.9 between wire and superconductor.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Model.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
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

//Lattice size
const int SIZE_X = 4*5;
const int SIZE_Y = 2*5+1;

//Order parameter. The two buffers are alternatively swaped by setting dCounter
// = 0 or 1. One buffer contains the order parameter used in the previous
//calculation, while the other is used to store the newly obtained order
//parameter that will be used in the next calculation.
Array<complex<double>> Delta({2, SIZE_X, SIZE_Y});
unsigned int deltaCounter = 0;

//Superconducting pair potential, convergence limit, max iterations, and
//initial guess.
const double V_sc = 2.;
const double CONVERGENCE_LIMIT = 0.0001;
const int MAX_ITERATIONS = 50;
const complex<double> DELTA_INITIAL_GUESS = 0.3;

//Self-consistent callback that is to be called each time a diagonalization has
//finished. Calculates the order parameter from the current solution.
class SelfConsistencyCallback :
	public Solver::Diagonalizer::SelfConsistencyCallback
{
	bool selfConsistencyCallback(Solver::Diagonalizer &solver){
		//Clear the order parameter
		for(unsigned int x = 0; x < SIZE_X; x++)
			for(unsigned int y = 0; y < SIZE_Y; y++)
				Delta[{(deltaCounter+1)%2, x, y}] = 0.;

		//Calculate D(x, y) = <c_{x, y, \downarrow}c_{x, y, \uparrow}> = \sum_{E_n<E_F} conj(v_d^{(n)})*u_u^{(n)}
		for(unsigned int x = 0; x < SIZE_X; x++){
			for(unsigned int y = 0; y < SIZE_Y; y++){
				for(
					int n = 0;
					n < solver.getModel().getBasisSize()/2;
					n++
				){
					//Obtain amplitudes at site (x,y) for
					//electron_up and hole_down components.
					complex<double> u_u
						= solver.getAmplitude(
							n,
							{0, x, y, 0, 0}
						);
					complex<double> v_d
						= solver.getAmplitude(
							n,
							{0, x, y, 1, 1}
						);

					Delta[{(deltaCounter+1)%2, x, y}]
						-= V_sc*conj(v_d)*u_u;
				}
			}
		}

		//Swap order parameter buffers
		deltaCounter = (deltaCounter+1)%2;

		//Calculate convergence parameter
		double maxError = 0.;
		for(unsigned int x = 0; x < SIZE_X; x++){
			for(unsigned int y = 0; y < SIZE_Y; y++){
				double error = 0.;
				if(abs(Delta[{deltaCounter, x, y}]) == 0
					&& abs(
						Delta[{
							(deltaCounter+1)%2,
							x,
							y
						}]
					) == 0
				){
					error = 0.;
				}
				else{
					error = abs(
						Delta[{deltaCounter, x, y}]
						- Delta[{
							(deltaCounter+1)%2,
							x,
							y
						}]
					)/max(
						abs(
							Delta[{
								deltaCounter,
								x,
								y
							}]
						),
						abs(
							Delta[{
								(deltaCounter+1)%2,
								x,
								y
							}]
						)
					);
				}

				if(error > maxError)
					maxError = error;
			}
		}

		//Return true or false depending on whether the result has converged or not
		if(maxError < CONVERGENCE_LIMIT)
			return true;
		else
			return false;
	}
} selfConsistencyCallback;

//Callback function responsible for determining the value of the order
//parameter D_{to,from}c_{to}c_{from} where to and from are indices of the form
//(x, y, spin).
class DeltaCallback: public HoppingAmplitude::AmplitudeCallback{
	complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		//Obtain indices
		unsigned int x = from[1];
		unsigned int y = from[2];
		unsigned int spin = from[3];
		unsigned int particleHole = from[4];

		//Return appropriate amplitude
		if(spin == 0 && particleHole == 0)
			return conj(Delta[{deltaCounter, x, y}]);
		else if(spin == 1 && particleHole == 0)
			return -conj(Delta[{deltaCounter, x, y}]);
		else if(spin == 0 && particleHole == 1)
			return -Delta[{deltaCounter, x, y}];
		else
			return Delta[{deltaCounter, x, y}];
	}
} deltaCallback;

//Function responsible for initializing the order parameter
void initDelta(){
	for(unsigned int x = 0; x < SIZE_X; x++)
		for(unsigned int y = 0; y < SIZE_Y; y++)
			Delta[{0, x, y}] = DELTA_INITIAL_GUESS;
}

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Parameters
	complex<double> mu = -1.0;
	complex<double> t_ss = 1.0;
	complex<double> t_mm = 0.75;
	complex<double> t_sm = 0.9;
	complex<double> V_z = 0.5;

	//Create model
	Model model;
	//Setup hopping parameters for superconducting layer
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model << HoppingAmplitude(
					-mu,
					{0, x, y, s, 0},
					{0, x, y, s, 0}
				);
				model << HoppingAmplitude(
					mu,
					{0, x, y, s, 1},
					{0, x, y, s, 1}
				);

				//Add hopping parameters corresponding to t_ss
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t_ss,
						{0, (x+1)%SIZE_X, y, s, 0},
						{0, x, y, s, 0}
					) + HC;
					model << HoppingAmplitude(
						t_ss,
						{0, (x+1)%SIZE_X, y, s, 1},
						{0, x, y, s, 1}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t_ss,
						{0, x, (y+1)%SIZE_Y, s, 0},
						{0, x, y, s, 0}
					) + HC;
					model << HoppingAmplitude(
						t_ss,
						{0, x, (y+1)%SIZE_Y, s, 1},
						{0, x, y, s, 1}
					) + HC;
				}
				model << HoppingAmplitude(
					deltaCallback,
					{0, x, y, (s+1)%2, 1},
					{0, x, y, s, 0}
				) + HC;
			}
		}
	}
	//Setup wire and attach it to superconductor
	for(unsigned int x = 0; x < SIZE_X/2; x++){
		for(unsigned int s = 0; s < 2; s++){
			//Add hopping amplitudes corresponding to Zeeman term
			model << HoppingAmplitude(
				-2.*V_z*(s-1/2.),
				{1, x, s, 0},
				{1, x, s, 0}
			);
			model << HoppingAmplitude(
				2.*V_z*(s-1/2.),
				{1, x, s, 1},
				{1, x, s, 1}
			);

			//Add hopping amplitudes corresponding to t_mm
			if(x+1 < SIZE_X/2){
				model << HoppingAmplitude(
					-t_mm,
					{1, x+1, s, 0},
					{1, x, s, 0}
				) + HC;
				model << HoppingAmplitude(
					t_mm,
					{1, x+1, s, 1},
					{1, x, s, 1}
				) + HC;
			}

			//Add hopping amplitudes corresponding to t_ms
			model << HoppingAmplitude(
				-t_sm,
				{0, x+SIZE_X/4, SIZE_Y/2, s, 0},
				{1, x, s, 0}
			) + HC;
			model << HoppingAmplitude(
				t_sm,
				{0, x+SIZE_X/4, SIZE_Y/2, s, 1},
				{1, x, s, 1}
			) + HC;
		}
	}

	//Construct model
	model.construct();

	//Initialize D
	initDelta();

	//Setup and run Solver::Diagonalizer
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.setMaxIterations(MAX_ITERATIONS);
	solver.setSelfConsistencyCallback(selfConsistencyCallback);
	solver.run();

	//Calculate abs(D) and arg(D)
	Array<double> DeltaAbs({SIZE_X, SIZE_Y});
	Array<double> DeltaArg({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			DeltaAbs[{x, y}] = abs(Delta[{deltaCounter, x, y}]);
			DeltaArg[{x, y}] = arg(Delta[{deltaCounter, x, y}]);
		}
	}

	//Plot Delta.
	Plotter plotter;
	plotter.plot(DeltaAbs);
	plotter.save("figures/DeltaAbs.png");
	plotter.clear();
	plotter.plot(DeltaArg);
	plotter.save("figures/DeltaArg.png");

	//Set up the PropertyExtractor.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	const double LOWER_BOUND = -2.;
	const double UPPER_BOUND = 2.;
	const unsigned int RESOLUTION = 1000;
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the Magnetization.
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{0, _a_, _a_, IDX_SPIN, 0}
		});

	//Plot the Magnetization along the z-axis.
	plotter.clear();
	plotter.plot({0, _a_, _a_, IDX_SPIN, 0}, {0, 0, 1}, magnetization);
	plotter.save("figures/Magnetization.png");

	//Calculate the spin-polarized LDOS.
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{0, _a_, SIZE_Y/2, IDX_SPIN, 0}
		});

	//Smooth the spin-polarized LDOS.
	const double SMOOTHING_SIGMA = 0.01;
	const unsigned int SMOOTHING_WINDOW = 51;
	spinPolarizedLDOS = Smooth::gaussian(
		spinPolarizedLDOS,
		SMOOTHING_SIGMA,
		SMOOTHING_WINDOW
	);

	//Plot the spin-polarized LDOS along the z-axis.
	plotter.clear();
	plotter.plot(
		{0, _a_, SIZE_Y/2, IDX_SPIN, 0},
		{0, 0, 1},
		spinPolarizedLDOS
	);
	plotter.save("figures/SpinPolarizedLDOS.png");

	return 0;
}
