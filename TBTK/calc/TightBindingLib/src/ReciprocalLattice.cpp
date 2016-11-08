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

/** @file ReciprocalLattice.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ReciprocalLattice.h"
#include "../include/TBTKMacros.h"

using namespace std;

namespace TBTK{

ReciprocalLattice::ReciprocalLattice(UnitCell *unitCell){
	TBTKNotYetImplemented("ReciprocalLattice::ReciprocalLattice()");
	this->unitCell = unitCell;

	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();

	switch(latticeVectors.size()){
	case 3:
		TBTKAssert(
			latticeVectors.at(0).size() == 3,
			"ReciprocalLattice::ReciprocalLattice()",
			"Lattice vector dimension not supported.",
			"Only three-dimensional lattice vectors are supported"
			<< " for UnitCells with thee lattice vectors. The"
			<< " supplied UnitCell has " << latticeVectors.at(0).size()
			<< " dimensions."
		);
	default:
		TBTKExit(
			"ReciprocalLattice::ReciprocalLattice()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}
}

ReciprocalLattice::~ReciprocalLattice(){
}

Model* ReciprocalLattice::generateModel(initializer_list<double> momentum){
	TBTKNotYetImplemented("ReciprocalLattice::generateModel()");
	Model *model = new Model();

	return model;
}

};	//End of namespace TBTK
