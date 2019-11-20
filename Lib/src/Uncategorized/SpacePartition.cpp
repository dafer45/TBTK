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

/** @file SpacePartition.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/SpacePartition.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

SpacePartition::SpacePartition(){
}

SpacePartition::SpacePartition(
	const vector<vector<double>> &basisVectors,
	MeshType meshType
){
	this->dimensions = basisVectors.size();
	this->meshType = meshType;

	TBTKAssert(
		dimensions == 1
		|| dimensions == 2
		|| dimensions == 3,
		"SpacePartition::SpacePartition()",
		"Basis dimension not supported.",
		"Only 1-3 basis vectors are supported, but "
		<< basisVectors.size() << " basis vectors supplied."
	);

	for(unsigned int n = 0; n < dimensions; n++){
		TBTKAssert(
			basisVectors.at(n).size() == dimensions,
			"SpacePartition::SpacePartition()",
			"Incompatible dimensions.",
			"The number of basis vectors must agree with the number"
			<< " of components of the basis vectors. The number of"
			<< " basis vectors are '" << dimensions << "',"
			<< " but encountered basis vector with '"
			<< basisVectors.at(n).size() << "' components."
		);

		vector<double> paddedBasisVector;
		for(unsigned int c = 0; c < dimensions; c++)
			paddedBasisVector.push_back(basisVectors.at(n).at(c));
		for(unsigned int c = dimensions; c < 3; c++)
			paddedBasisVector.push_back(0.);

		this->basisVectors.push_back(Vector3d(paddedBasisVector));
	}
	for(unsigned int n = dimensions; n < 3; n++){
		vector<double> vector;
		for(unsigned int c = 0; c < 3; c++){
			if(c == n)
				vector.push_back(1.);
			else
				vector.push_back(0.);
		}

		this->basisVectors.push_back(Vector3d(vector));
	}
}

SpacePartition::~SpacePartition(){
}

};	//End of namespace TBTK
