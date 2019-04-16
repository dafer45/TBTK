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

/** @file GaussianState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/GaussianState.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

GaussianState::GaussianState(
	const Index &index,
	const Vector3d &coordinates,
	unsigned int linearIndex,
	unsigned int basisSize
) : AbstractState(StateID::Gaussian)
{
	setIndex(index);
	setCoordinates(coordinates.getStdVector());
	this->linearIndex = linearIndex;
	this->basisSize = basisSize;
	overlaps.reserve(basisSize);
	kineticTerms.reserve(basisSize);
	nuclearTerms.reserve(basisSize);
	for(unsigned int n = 0; n < basisSize; n++){
		overlaps.push_back(0);
		kineticTerms.push_back(0);
		nuclearTerms.push_back(0);
	}
}

complex<double> GaussianState::getOverlap(const AbstractState &ket) const{
	TBTKNotYetImplemented("GaussianState::getOverlap()");
}

complex<double> GaussianState::getMatrixElement(
	const AbstractState &ket,
	const AbstractOperator &o
) const{
	TBTKNotYetImplemented("GaussianState::getMatrixElement()");
}

};	//End of namespace TBTK
