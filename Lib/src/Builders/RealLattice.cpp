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

/** @file RealLattice.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/RealLattice.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

RealLattice::RealLattice(const UnitCell *unitCell){
	this->unitCell = unitCell;
}

RealLattice::~RealLattice(){
}

void RealLattice::addLatticePoint(const Index &latticePoint){
	TBTKAssert(
		latticePoint.getSize() == unitCell->getLatticeVectors().size(),
		"RealLattice::addLaticePoint()",
		"Incompatible lattice dimensions. The lattice point has to have the same dimensionality as the UnitCell.",
		"The UnitCell has " << unitCell->getLatticeVectors().size() << " lattice vectors, but the argument 'latticeVector' has " << to_string(latticePoint.getSize()) << " componentes."
	);

	latticePoints.push_back(latticePoint);
}

StateSet* RealLattice::generateStateSet(){
	StateSet *stateSet = new StateSet();

	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();
	const vector<AbstractState*> states = unitCell->getStates();

	for(unsigned int l = 0; l < latticePoints.size(); l++){
		const Index &latticePoint = latticePoints.at(l);

		for(unsigned int s = 0; s < states.size(); s++){
			AbstractState *state = states.at(s)->clone();

			const vector<double> &coordinates = state->getCoordinates();

			int coordinateDimension = coordinates.size();
			int numLatticeVectors = latticeVectors.size();
			int latticeVectorDimension = latticeVectors.at(0).size();

			TBTKAssert(
				coordinateDimension >= numLatticeVectors,
				"RealLattice::generateStateSet()",
				"Incompatible state and lattice vector"
				<< " dimension. The state has dimension "
				<< coordinateDimension << ", while"
				<< " the number of lattice vectors are "
				<< numLatticeVectors << ".",
				"The State has to have at least the same"
				<< " number of dimensions as the number of"
				<< " lattice vectors."
			);
			TBTKAssert(
				coordinateDimension >= latticeVectorDimension,
				"RealLattice::generateStateSet()",
				"Incompatible state and lattice vector"
				<< " dimension. The state has dimension "
				<< coordinates.size() << ", while"
				<< " the lattice vectors have dimension "
				<< latticeVectorDimension << ".",
				"The State has to have at least the same"
				<< " number of dimensions as the lattice"
				<< " vectors."
			);

			vector<double> position;
			for(int c = 0; c < coordinateDimension; c++)
				position.push_back(coordinates.at(c));

			for(int v = 0; v < numLatticeVectors; v++){
				const vector<double> &latticeVector = latticeVectors.at(v);
				for(int c = 0; c < latticeVectorDimension; c++)
					position.at(c) += latticeVector.at(c)*latticePoint.at(v);
			}

			state->setCoordinates(position);
			state->setContainer(latticePoint);

			stateSet->addState(state);
		}
	}

	return stateSet;
}

};	//End of namespace TBTK
