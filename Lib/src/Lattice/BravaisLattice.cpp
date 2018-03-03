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

/** @file BravaisLattice.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Lattice/BravaisLattice.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Lattice{

BravaisLattice::BravaisLattice(){
}

BravaisLattice::~BravaisLattice(){
}

void BravaisLattice::makePrimitive(){
	TBTKAssert(
		additionalSites.size() == 0,
		"BravaisLattice::makePrimitive()",
		"Conversion to primitive cell not implemented.",
		"Request a proper implmentation from the developer."
	);
}

void BravaisLattice::setLatticeVectors(const vector<vector<double>> &latticeVectors){
	this->latticeVectors.clear();
	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		TBTKAssert(
			latticeVectors.at(n).size() == latticeVectors.size(),
			"BravaisLattice::setLatticeVectors()",
			"Unsupported lattice vector dimension.",
			"The lattice vectors must have the same number of"
			<< " components as the number of lattice vectors. The"
			<< " number of lattice vectors is "
			<< latticeVectors.size() << ", but encountered lattice"
			<< " vector with " << latticeVectors.at(n).size()
			<< " components."
		);

		this->latticeVectors.push_back(vector<double>());
		for(unsigned int c = 0; c < latticeVectors.at(n).size(); c++)
			this->latticeVectors.at(n).push_back(latticeVectors.at(n).at(c));
	}
}

void BravaisLattice::setAdditionalSites(const vector<vector<double>> &additionalSites){
	this->additionalSites.clear();
	for(unsigned int n = 0; n < additionalSites.size(); n++){
		TBTKAssert(
			additionalSites.at(n).size() == latticeVectors.size(),
			"BravaisLattice::setAdditionalSites()",
			"Unsupported site dimension.",
			"The additional site must have the same number of"
			<< " components as the number of lattice vectors. The"
			<< " number of lattice vectors is "
			<< latticeVectors.size() << ", but encountered"
			<< " additional site with "
			<< additionalSites.at(n).size() << " components."
		);

		this->additionalSites.push_back(vector<double>());
		for(unsigned int c = 0; c < additionalSites.at(n).size(); c++)
			this->additionalSites.at(n).push_back(additionalSites.at(n).at(c));
	}
}

};	//End of namespace Lattice
};	//End of namespace TBTK
