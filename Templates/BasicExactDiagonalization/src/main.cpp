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
 *  @brief Basic exact diagonalization example
 *
 *  Solves
 *      H = \sum_{ij}a_{ij}c_{i}^{\dagger}c_{j}
 *      + \sum_{ijkl}b_{ijkl}c_{i}^{\dagger}c_{j}^{\dagger}c_{k}c_{l}
 *  where
 *      a_{ii} = mu
 *      a_{ij} = -t	for i and j nearest neighbor,
 *      a_{ij} = 0	otherwise,
 *
 *      b_{ijji} = U	for i and j oposite spins on the same site,
 *      b_{ijkl} = 0	otherwise.
 *
 *  The ground state is specified to be at half-filling and with equal number
 *  of up and down spins.
 *
 *  @author Kristofer Björnson
 */

#include "DifferenceRule.h"
#include "EDPropertyExtractor.h"
#include "ExactDiagonalizationSolver.h"
#include "FileWriter.h"
#include "Model.h"
#include "SumRule.h"

#include <complex>

using namespace std;
using namespace TBTK;

complex<double> i(0, 1);

int main(int argc, char **argv){
	//Clear folder to prepare for writing new data
	FileWriter::clear();

	//Model Parameters for Hubbard model
	const int SIZE_X = 3;
	const int SIZE_Y = 2;
	complex<double> t = -0.5;
	complex<double> U = 2.;
	complex<double> mu = -U/2.;

	//Create single particle Hamiltonian. (Add a_{ij})
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add mu (i = j = {x, y, s})
				model.addHA(HoppingAmplitude(mu,	{x, y, s},		{x, y, s}));

				//Add -t (i = {x+1, y, s}, j = {x, y, s}.
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude(t,	{(x+1)%SIZE_X, y, s},	{x, y, s}));
				if(y+1 < SIZE_Y)
					model.addHAAndHC(HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, s},	{x, y, s}));
			}
		}
	}

	//Construct single particle Hilbert space.
	model.construct();

	//Create many-body context and add interaction amplitudes.
	model.createManyBodyContext();
	ManyBodyContext *manyBodyContext = model.getManyBodyContext();
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			manyBodyContext->addIA(InteractionAmplitude(
				U,
				{{x, y, 0}, {x, y, 1}},	//Creation operators (i and j)
				{{x, y, 1}, {x, y, 0}}	//Annihilation operators (k and l)
			));
		}
	}

	//Add rules determining what Fock space subspace the ground state
	//belongs to.
	manyBodyContext->addFockStateRule(		//Half-filling
		FockStateRule::SumRule(
			{{IDX_ALL, IDX_ALL, IDX_ALL}},	//Add all states
			SIZE_X*SIZE_Y			//Total number of particles
		)
	);
	manyBodyContext->addFockStateRule(		//Total spin = 0
		FockStateRule::DifferenceRule(
			{{IDX_ALL, IDX_ALL, 0}},	//Add up spins
			{{IDX_ALL, IDX_ALL, 1}},	//Subtract down spins
			0				//spin up - spin down = 0
		)
	);

	//Create exact diagonalization solver. Notation will be simplified.
	ExactDiagonalizationSolver edSolver(&model);

	//Create and initialize EDPropertyExtractor.
	EDPropertyExtractor pe(&edSolver);
	const int ENERGY_RESOLUTION = 1000;
	double LOWER_BOUND = -6.;
	double UPPER_BOUND = 6.;
	pe.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, ENERGY_RESOLUTION);

	//Calculate LDOS and write to file.
	Property::LDOS *ldos = pe.calculateLDOS(
		{IDX_X, 0, IDX_SUM_ALL},
		{SIZE_X, SIZE_Y, 2}
	);
	FileWriter::writeLDOS(ldos);
	delete ldos;
}
