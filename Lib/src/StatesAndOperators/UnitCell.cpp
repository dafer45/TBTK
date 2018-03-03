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

/** @file UnitCell.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/UnitCell.h"

using namespace std;

namespace TBTK{

UnitCell::UnitCell(
	initializer_list<initializer_list<double>> latticeVectors,
	bool isOwner
) :
	StateSet(isOwner)
{
	unsigned int numCoordinates = latticeVectors.begin()->size();
	for(unsigned int n = 1; n < latticeVectors.size(); n++){
		TBTKAssert(
			(latticeVectors.begin()+n)->size() == numCoordinates,
			"UnitCell::UnitCell()",
			"Incmopatible coordinate dimensions. The first lattice"
			<< "vector has " << numCoordinates << " coordinates, "
			<< "while lattice vector " << n << " has "
			<< (latticeVectors.begin()+n)->size()
			<< " coordinates.",
			""
		);
	}

	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		this->latticeVectors.push_back(vector<double>());

		const initializer_list<double> *latticeVector = (latticeVectors.begin() + n);
		for(unsigned int c = 0; c < latticeVector->size(); c++)
			this->latticeVectors.at(n).push_back(*(latticeVector->begin()+c));
	}
}

UnitCell::UnitCell(
	const vector<vector<double>> &latticeVectors,
	bool isOwner
) :
	StateSet(isOwner)
{
	unsigned int numCoordinates = latticeVectors.at(0).size();
	for(unsigned int n = 1; n < latticeVectors.size(); n++){
		TBTKAssert(
			latticeVectors.at(n).size() == numCoordinates,
			"UnitCell::UnitCell()",
			"Incmopatible coordinate dimensions. The first lattice"
			<< "vector has " << numCoordinates << " coordinates, "
			<< "while lattice vector " << n << " has "
			<< latticeVectors.at(n).size()
			<< " coordinates.",
			""
		);
	}

	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		this->latticeVectors.push_back(vector<double>());

		const vector<double> &latticeVector = latticeVectors.at(n);
		for(unsigned int c = 0; c < latticeVector.size(); c++)
			this->latticeVectors.at(n).push_back(latticeVector.at(c));
	}
}

UnitCell::~UnitCell(){
}

};	//End of namespace TBTK
