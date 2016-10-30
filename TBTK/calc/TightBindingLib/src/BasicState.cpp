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

/** @file BasicState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/BasicState.h"
#include "../include/TBTKMacros.h"

using namespace std;

namespace TBTK{

BasicState::BasicState(const Index &index, const Index &unitCell) :
	AbstractState(AbstractState::StateID::Basic),
	index(index),
	unitCell(unitCell)
{
}

BasicState::~BasicState(){
}

void BasicState::addOverlap(
	complex<double> overlap,
	const Index &braIndex,
	const Index &braUnitCell
){
	overlaps.push_back(make_tuple(overlap, braIndex, braUnitCell));
}

void BasicState::addMatrixElement(
	complex<double> matrixElement,
	const Index &braIndex,
	const Index &braUnitCell
){
	overlaps.push_back(make_tuple(matrixElement, braIndex, braUnitCell));
}

complex<double> BasicState::getOverlap(const AbstractState &bra) const{
	TBTKAssert(
		bra.getStateID() == AbstractState::Basic,
		"BasicState::getOverlap()",
		"Incompatible states.",
		"The bra state has to be a BasicState."
	);

	for(unsigned int n = 0; n < overlaps.size(); n++)
		if(index.equals(
			get<1>(overlaps.at(n))) &&
			unitCell.equals(get<2>(overlaps.at(n)))
		)
			return get<0>(overlaps.at(n));

	return 0.;
}

complex<double> BasicState::getMatrixElement(
	const AbstractState &bra,
	const AbstractOperator &o
) const{
	TBTKAssert(
		bra.getStateID() == AbstractState::Basic,
		"BasicState::getMatrixElement()",
		"Incompatible states.",
		"The bra state has to be a BasicState."
	);

	for(unsigned int n = 0; n < matrixElements.size(); n++)
		if(
			index.equals(get<1>(matrixElements.at(n))) &&
			unitCell.equals(get<2>(matrixElements.at(n)))
		)
			return get<0>(overlaps.at(n));

	return 0.;
}

};	//End of namespace TBTK
