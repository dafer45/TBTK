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

BasicState::BasicState(const Index &index, const Index &unitCellIndex) :
	AbstractState(AbstractState::StateID::Basic)
{
	setIndex(index);
	setContainer(unitCellIndex);
}

BasicState::~BasicState(){
}

BasicState* BasicState::clone() const{
	BasicState* clonedState = new BasicState(*this);

	return clonedState;
}

void BasicState::addOverlap(
	complex<double> overlap,
	const Index &braIndex,
	const Index &braRelativeUnitCell
){
	overlaps.push_back(make_tuple(overlap, braIndex, braRelativeUnitCell));
}

void BasicState::addMatrixElement(
	complex<double> matrixElement,
	const Index &braIndex,
	const Index &braRelativeUnitCell
){
	matrixElements.push_back(make_tuple(matrixElement, braIndex, braRelativeUnitCell));
}

complex<double> BasicState::getOverlap(const AbstractState &bra) const{
	TBTKAssert(
		bra.getStateID() == AbstractState::Basic,
		"BasicState::getOverlap()",
		"Incompatible states.",
		"The bra state has to be a BasicState."
	);

	for(unsigned int n = 0; n < overlaps.size(); n++){
		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(overlaps.at(n));
		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(overlaps.at(n));

		TBTKAssert(
			braIndex.size() == ketIndex.size(),
			"BasicState::getOverlap()",
			"Incompatible indices for <bra| and |ket>. <bra| has"
			<< " index '" << braIndex.toString() << "', while"
			<< " |ket> has index '"
			<< ketIndex.toString() << "'.",
			""
		);
		TBTKAssert(
			braUnitCell.size() == ketRelativeUnitCell.size(),
			"BasicState::getOverlap()",
			"Incompatible unit cell indices for <bra| and |ket>."
			<< " <bra| has unit cell index '"
			<< braUnitCell.toString() << "', while ket has"
			<< " relative unit cell index '"
			<< ketRelativeUnitCell.toString() << "'.",
			""
		);
		TBTKAssert(
			getContainer().size() == ketRelativeUnitCell.size(),
			"BasicState::getOverlap()",
			"Incompatible unit cell indices for <bra| and |ket>."
			<< " <bra| has unit cell index '"
			<< braUnitCell.toString() << ", while |ket> has unit"
			<< " cell index '" << getContainer().toString()
			<< "'.",
			""
		);

		Index ketAbsoluteUnitCell({});
		for(unsigned int c = 0; c < braUnitCell.size(); c++)
			ketAbsoluteUnitCell.push_back(getContainer().at(c) + ketRelativeUnitCell.at(c));
		if(
			braIndex.equals(ketIndex) &&
			braUnitCell.equals(ketAbsoluteUnitCell)
		)

		return get<0>(overlaps.at(n));
	}

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

	for(unsigned int n = 0; n < matrixElements.size(); n++){
		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(matrixElements.at(n));
		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(matrixElements.at(n));

		TBTKAssert(
			braIndex.size() == ketIndex.size(),
			"BasicState::getMatrixElements()",
			"Incompatible indices for <bra| and |ket>. <bra| has"
			<< " index '" << braIndex.toString() << "', while"
			<< " |ket> has index '"
			<< ketIndex.toString() << "'.",
			""
		);
		TBTKAssert(
			braUnitCell.size() == ketRelativeUnitCell.size(),
			"BasicState::getMatrixElements()",
			"Incompatible unit cell indices for <bra| and |ket>."
			<< " <bra| has unit cell index '"
			<< braUnitCell.toString() << "', while ket has"
			<< " relative unit cell index '"
			<< ketRelativeUnitCell.toString() << "'.",
			""
		);
		TBTKAssert(
			getContainer().size() == ketRelativeUnitCell.size(),
			"BasicState::getMatrixElements()",
			"Incompatible unit cell indices for <bra| and |ket>."
			<< " <bra| has unit cell index '"
			<< braUnitCell.toString() << ", while |ket> has unit"
			<< " cell index '" << getContainer().toString()
			<< "'.",
			""
		);

		Index ketAbsoluteUnitCell({});
		for(unsigned int c = 0; c < braUnitCell.size(); c++)
			ketAbsoluteUnitCell.push_back(getContainer().at(c) + ketRelativeUnitCell.at(c));
		if(
			braIndex.equals(ketIndex) &&
			braUnitCell.equals(ketAbsoluteUnitCell)
		)

		return get<0>(matrixElements.at(n));
	}

	return 0.;
}

};	//End of namespace TBTK
