/* Copyright 2019 Kristofer Björnson
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

/** @file GaussianBasisGenerator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/GaussianBasisGenerator.h"
#include "TBTK/TBTKMacros.h"

#include <libint2.hpp>

using namespace std;

namespace TBTK{

vector<GaussianState> GaussianBasisGenerator::generateBasis(){
	TBTKNotYetImplemented("GaussianBasisGenerator::generateBasis()");

	vector<GaussianState> basis;

	libint2::initialize();

	vector<libint2::Atom> atoms;
	atoms.push_back({1, 1, 0, 0});
	atoms.push_back({1, 0, 1, 0});

	libint2::BasisSet basisSet("cc-pVDZ", atoms, true);
	const vector<size_t> &shellToFirstBasisFunction = basisSet.shell2bf();

	unsigned int numSpinOrbitals = 2*basisSet.nbf();

	int spinOrbitalCounter = 0;
	for(auto &shell : basisSet){
		for(auto &contraction : shell.contr){
			for(int n = 0; n < 2*contraction.l + 1; n++){
				for(int s = 0; s < 2; s++){
					basis.push_back(
						GaussianState(
							{
								spinOrbitalCounter,
								s
							},
							{
								shell.O[0],
								shell.O[1],
								shell.O[2]
							},
							2*spinOrbitalCounter
								+ s,
							numSpinOrbitals
						)
					);
				}

				spinOrbitalCounter++;
			}
		}
	}

	libint2::Engine engine(
		libint2::Operator::overlap,
		basisSet.max_nprim(),
		basisSet.max_l(),
		0
	);

	const auto &bufferVector0 = engine.results();

	for(unsigned int shell0 = 0; shell0 < basisSet.size(); shell0++){
		size_t firstBasisFunction0 = shellToFirstBasisFunction[shell0];
		size_t numBasisFunctions0 = basisSet[shell0].size();
		for(
			unsigned int shell1 = 0;
			shell1 < basisSet.size();
			shell1++
		){
			engine.compute(basisSet[shell0], basisSet[shell1]);

			auto intShellSet = bufferVector0[0];
			if(intShellSet == nullptr)
				continue;

			size_t firstBasisFunction1
				= shellToFirstBasisFunction[shell1];
			size_t numBasisFunctions1 = basisSet[shell1].size();

			for(size_t f0 = 0; f0 < numBasisFunctions0; f0++){
				for(
					size_t f1 = 0;
					f1 < numBasisFunctions1;
					f1++
				){
					basis[
						firstBasisFunction0 + f0
					].setOverlap(
						intShellSet[
							f0*numBasisFunctions1
							+ f1
						],
						firstBasisFunction1 + f1
					);
				}
			}
		}
	}

	engine = libint2::Engine(
		libint2::Operator::kinetic,
		basisSet.max_nprim(),
		basisSet.max_l(),
		0
	);

	const auto &bufferVector1 = engine.results();

	for(unsigned int shell0 = 0; shell0 < basisSet.size(); shell0++){
		size_t firstBasisFunction0 = shellToFirstBasisFunction[shell0];
		size_t numBasisFunctions0 = basisSet[shell0].size();
		for(
			unsigned int shell1 = 0;
			shell1 < basisSet.size();
			shell1++
		){
			engine.compute(basisSet[shell0], basisSet[shell1]);

			auto intShellSet = bufferVector1[0];
			if(intShellSet == nullptr)
				continue;

			size_t firstBasisFunction1
				= shellToFirstBasisFunction[shell1];
			size_t numBasisFunctions1 = basisSet[shell1].size();

			for(size_t f0 = 0; f0 < numBasisFunctions0; f0++){
				for(
					size_t f1 = 0;
					f1 < numBasisFunctions1;
					f1++
				){
					basis[
						firstBasisFunction0 + f0
					].setKineticTerm(
						intShellSet[
							f0*numBasisFunctions1
							+ f1
						],
						firstBasisFunction1 + f1
					);
				}
			}
		}
	}

	engine = libint2::Engine(
		libint2::Operator::nuclear,
		basisSet.max_nprim(),
		basisSet.max_l(),
		0
	);
	engine.set_params(libint2::make_point_charges(atoms));

	const auto &bufferVector2 = engine.results();

	for(unsigned int shell0 = 0; shell0 < basisSet.size(); shell0++){
		size_t firstBasisFunction0 = shellToFirstBasisFunction[shell0];
		size_t numBasisFunctions0 = basisSet[shell0].size();
		for(
			unsigned int shell1 = 0;
			shell1 < basisSet.size();
			shell1++
		){
			engine.compute(basisSet[shell0], basisSet[shell1]);

			auto intShellSet = bufferVector2[0];
			if(intShellSet == nullptr)
				continue;

			size_t firstBasisFunction1
				= shellToFirstBasisFunction[shell1];
			size_t numBasisFunctions1 = basisSet[shell1].size();

			for(size_t f0 = 0; f0 < numBasisFunctions0; f0++){
				for(
					size_t f1 = 0;
					f1 < numBasisFunctions1;
					f1++
				){
					basis[
						firstBasisFunction0 + f0
					].setNuclearTerm(
						intShellSet[
							f0*numBasisFunctions1
							+ f1
						],
						firstBasisFunction1 + f1
					);
				}
			}
		}
	}

/*	engine = Engine(
		Operator::coulomb,
		basisSet.max_nprim(),
		basisSet.max_l(),
		0
	);

	const auto &bufferVector3 = engine.results();*/

	libint2::finalize();

	return basis;
}

};	//End of namespace TBTK
