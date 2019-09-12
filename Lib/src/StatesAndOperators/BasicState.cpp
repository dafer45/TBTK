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

#include "TBTK/BasicState.h"
#include "TBTK/TBTKMacros.h"

#include <algorithm>

using namespace std;

namespace TBTK{

BasicState::BasicState(const Index &index, const Index &unitCellIndex) :
	AbstractState(AbstractState::StateID::Basic)
{
	setIndex(index);
	setContainer(unitCellIndex);

	storage = new Storage;
/*	storage->referenceCounter = 1;
	storage->overlapsIndexTree = nullptr;
	storage->matrixElementsIndexTree = nullptr;*/
}

BasicState::~BasicState(){
	if(storage->release())
		delete storage;
/*	storage->referenceCounter--;
	if(storage->referenceCounter == 0){
		delete storage;
	}*/
}

BasicState* BasicState::clone() const{
	BasicState* clonedState = new BasicState(*this);
	storage->grab();
//	storage->referenceCounter++;

	return clonedState;
}

void BasicState::addOverlap(
	complex<double> overlap,
	const Index &braIndex,
	const Index &braRelativeUnitCell
){
	storage->overlapsIsSorted = false;
	storage->overlaps.push_back(make_tuple(overlap, braIndex, braRelativeUnitCell));
}

void BasicState::addMatrixElement(
	complex<double> matrixElement,
	const Index &braIndex,
	const Index &braRelativeUnitCell
){
	storage->matrixElementsIsSorted = false;
	storage->matrixElements.push_back(make_tuple(matrixElement, braIndex, braRelativeUnitCell));
}

complex<double> BasicState::getOverlap(const AbstractState &bra) const{
	TBTKAssert(
		bra.getStateID() == AbstractState::Basic,
		"BasicState::getOverlap()",
		"Incompatible states.",
		"The bra state has to be a BasicState."
	);

	if(!storage->overlapsIsSorted){
		storage->sortOverlaps();
	}

	int min = 0;
	int max = storage->overlaps.size()-1;
	while(min <= max){
		int m = (min + max)/2;

		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(storage->overlaps.at(m));
		if(ketIndex < braIndex){
			min = m + 1;
			continue;
		}
		if(ketIndex > braIndex){
			max = m - 1;
			continue;
		}

		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(storage->overlaps.at(m));
		Index ketAbsoluteUnitCell;
		for(unsigned int c = 0; c < braUnitCell.getSize(); c++)
			ketAbsoluteUnitCell.pushBack(getContainer().at(c) + ketRelativeUnitCell.at(c));

		if(ketAbsoluteUnitCell < braUnitCell){
			min = m + 1;
			continue;
		}
		if(ketAbsoluteUnitCell > braUnitCell){
			max = m - 1;
			continue;
		}

		return get<0>(storage->overlaps.at(m));
	}

	return 0.;

/*	for(unsigned int n = 0; n < storage->overlaps.size(); n++){
		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(storage->overlaps.at(n));
		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(storage->overlaps.at(n));

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

//		Index ketAbsoluteUnitCell({});
		Index ketAbsoluteUnitCell;
		for(unsigned int c = 0; c < braUnitCell.size(); c++)
			ketAbsoluteUnitCell.push_back(getContainer().at(c) + ketRelativeUnitCell.at(c));
		if(
			braIndex.equals(ketIndex) &&
			braUnitCell.equals(ketAbsoluteUnitCell)
		)

		return get<0>(storage->overlaps.at(n));
	}

	return 0.;*/
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

	if(!storage->matrixElementsIsSorted){
		storage->sortMatrixElements();
	}

	int min = 0;
	int max = storage->matrixElements.size()-1;
	while(min <= max){
		int m = (min + max)/2;

		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(storage->matrixElements.at(m));
		if(ketIndex < braIndex){
			min = m + 1;
			continue;
		}
		if(ketIndex > braIndex){
			max = m - 1;
			continue;
		}

		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(storage->matrixElements.at(m));
		Index ketAbsoluteUnitCell;
		for(unsigned int c = 0; c < braUnitCell.getSize(); c++)
			ketAbsoluteUnitCell.pushBack(getContainer().at(c) + ketRelativeUnitCell.at(c));

		if(ketAbsoluteUnitCell < braUnitCell){
			min = m + 1;
			continue;
		}
		if(ketAbsoluteUnitCell > braUnitCell){
			max = m - 1;
			continue;
		}

		return get<0>(storage->matrixElements.at(m));
	}

	return 0.;

/*	for(unsigned int n = 0; n < storage->matrixElements.size(); n++){
		const Index &braIndex = bra.getIndex();
		const Index &ketIndex = get<1>(storage->matrixElements.at(n));
		const Index &braUnitCell = bra.getContainer();
		const Index &ketRelativeUnitCell = get<2>(storage->matrixElements.at(n));

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

//		Index ketAbsoluteUnitCell({});
		Index ketAbsoluteUnitCell;
		for(unsigned int c = 0; c < braUnitCell.size(); c++)
			ketAbsoluteUnitCell.push_back(getContainer().at(c) + ketRelativeUnitCell.at(c));
		if(
			braIndex.equals(ketIndex) &&
			braUnitCell.equals(ketAbsoluteUnitCell)
		)

		return get<0>(storage->matrixElements.at(n));
	}

	return 0.;*/
}

BasicState::Storage::Storage(){
	referenceCounter = 1;
	overlapsIsSorted = true;
	matrixElementsIsSorted = true;
//	overlapsIndexTree = nullptr;
//	indexedOverlaps = nullptr;
//	matrixElementsIndexTree = nullptr;
//	indexedMatrixElements = nullptr;
}

BasicState::Storage::~Storage(){
/*	if(overlapsIndexTree != nullptr)
		delete overlapsIndexTree;
	if(indexedOverlaps != nullptr)
		delete indexedOverlaps;
	if(matrixElementsIndexTree != nullptr)
		delete matrixElementsIndexTree;
	if(indexedMatrixElements != nullptr)
		delete indexedMatrixElements;*/
}

class SortHelperClass{
public:
	inline bool operator()(const tuple<complex<double>, Index, Index> &overlap1, const tuple<complex<double>, Index, Index> &overlap2){
		if(get<1>(overlap1) < get<1>(overlap2))
			return true;
		if(get<1>(overlap1) > get<1>(overlap2))
			return false;
		if(get<2>(overlap1) < get<2>(overlap2))
			return true;
		return false;
	}
};

void BasicState::Storage::sortOverlaps(){
	sort(overlaps.begin(), overlaps.end(), SortHelperClass());
	overlapsIsSorted = true;
}

void BasicState::Storage::sortMatrixElements(){
	sort(matrixElements.begin(), matrixElements.end(), SortHelperClass());
	matrixElementsIsSorted = true;
}

};	//End of namespace TBTK
