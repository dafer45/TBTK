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

/** @file Lattice.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Lattice.h"
#include "../include/TBTKMacros.h"

using namespace std;

namespace TBTK{

Lattice::Lattice(UnitCell *unitCell){
	this->unitCell = unitCell;
}

Lattice::~Lattice(){
}

void Lattice::addLatticePoint(const Index &latticePoint){
	TBTKAssert(
		latticePoint.size() == unitCell->getLatticeVectors().size(),
		"Lattice::addLaticePoint()",
		"Incompatible lattice dimensions. The lattice point has to have the same dimensionality as the UnitCell.",
		"The UnitCell has " << unitCell->getLatticeVectors().size() << " lattice vectors, but the argument 'latticeVector' has " << to_string(latticePoint.size()) << " componentes."
	);

	latticePoints.push_back(latticePoint);
}

};	//End of namespace TBTK
