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

/** @file HoppingAmplitudeSet.cpp
 *
 *  @author Kristofer Björnson
 */

#include "HoppingAmplitudeSet.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include "json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

HoppingAmplitudeSet::HoppingAmplitudeSet(){
	isConstructed = false;
	isSorted = false;
	numMatrixElements = -1;

	cooRowIndices = NULL;
	cooColIndices = NULL;
	cooValues = NULL;
}

HoppingAmplitudeSet::HoppingAmplitudeSet(const vector<unsigned int> &capacity){
	isConstructed = false;
	isSorted = false;
	numMatrixElements = -1;

	cooRowIndices = NULL;
	cooColIndices = NULL;
	cooValues = NULL;

	hoppingAmplitudeTree = HoppingAmplitudeTree(capacity);
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	const HoppingAmplitudeSet &hoppingAmplitudeSet
){
	hoppingAmplitudeTree = hoppingAmplitudeSet.hoppingAmplitudeTree;
	isConstructed = hoppingAmplitudeSet.isConstructed;
	isSorted = hoppingAmplitudeSet.isSorted;
	numMatrixElements = hoppingAmplitudeSet.numMatrixElements;

	if(numMatrixElements == -1){
		cooRowIndices = nullptr;
		cooColIndices = nullptr;
		cooValues = nullptr;
	}
	else{
		cooRowIndices = new int[numMatrixElements];
		cooColIndices = new int[numMatrixElements];
		cooValues = new complex<double>[numMatrixElements];
		for(int n = 0; n < numMatrixElements; n++){
			cooRowIndices[n] = hoppingAmplitudeSet.cooRowIndices[n];
			cooColIndices[n] = hoppingAmplitudeSet.cooColIndices[n];
			cooValues[n] = hoppingAmplitudeSet.cooValues[n];
		}
	}
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	HoppingAmplitudeSet &&hoppingAmplitudeSet
){
	hoppingAmplitudeTree = hoppingAmplitudeSet.hoppingAmplitudeTree;
	isConstructed = hoppingAmplitudeSet.isConstructed;
	isSorted = hoppingAmplitudeSet.isSorted;
	numMatrixElements = hoppingAmplitudeSet.numMatrixElements;

	cooRowIndices = hoppingAmplitudeSet.cooRowIndices;
	hoppingAmplitudeSet.cooRowIndices = nullptr;

	cooColIndices = hoppingAmplitudeSet.cooColIndices;
	hoppingAmplitudeSet.cooColIndices = nullptr;

	cooValues = hoppingAmplitudeSet.cooValues;
	hoppingAmplitudeSet.cooValues = nullptr;
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	const string &serialization,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	{
		TBTKAssert(
			validate(serialization, "HoppingAmplitudeSet", mode),
			"HoppingAmplitudeSet::HoppingAmplitudeSet()",
			"Unable to parse string as HoppingAmplitudeSet '"
			<< serialization << "'.",
			""
		);
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		hoppingAmplitudeTree = HoppingAmplitudeTree(elements.at(0), mode);
		stringstream ss;
		ss.str(elements.at(1));
		ss >> isConstructed;
		ss.clear();
		ss.str(elements.at(2));
		ss >> isSorted;
		ss.clear();
		ss.str(elements.at(3));
		ss >> numMatrixElements;
		if(numMatrixElements == -1){
			cooRowIndices = nullptr;
			cooColIndices = nullptr;
			cooValues = nullptr;
		}
		else{
			cooRowIndices = new int[numMatrixElements];
			cooColIndices = new int[numMatrixElements];
			cooValues = new complex<double>[numMatrixElements];
			unsigned int counter = 4;
			for(int n = 0; n < numMatrixElements; n++){
				ss.clear();
				ss.str(elements.at(counter));
				ss >> cooRowIndices[n];
				counter++;
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss.clear();
				ss.str(elements.at(counter));
				ss >> cooColIndices[n];
				counter++;
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss.clear();
				ss.str(elements.at(counter));
				ss >> cooValues[n];
				counter++;
			}
		}

		break;
	}
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			hoppingAmplitudeTree = HoppingAmplitudeTree(
				j.at("hoppingAmplitudeTree").dump(),
				mode
			);
			isConstructed = j.at("isConstructed").get<bool>();
			isSorted = j.at("isSorted").get<bool>();
			numMatrixElements = j.at("numMatrixElements").get<int>();
			if(numMatrixElements == -1){
				cooRowIndices = nullptr;
				cooColIndices = nullptr;
				cooValues = nullptr;
			}
			else{
				json cri = j.at("cooRowIndices");
				json cci = j.at("cooColIndices");
				json cv = j.at("cooValues");
				TBTKAssert(
					distance(
						cri.begin(),
						cri.end()
					) == numMatrixElements,
					"HoppingAmplitudeSet::HoppingAmplitudeSet()",
					"Incompatible array sizes."
					<< " 'numMatrixElements' is "
					<< numMatrixElements << " but"
					<< " cooRowIndices has "
					<< distance(cri.begin(), cri.end())
					<< " elements.",
					""
				);
				TBTKAssert(
					distance(
						cci.begin(),
						cci.end()
					) == numMatrixElements,
					"HoppingAmplitudeSet::HoppingAmplitudeSet()",
					"Incompatible array sizes."
					<< " 'numMatrixElements' is "
					<< numMatrixElements << " but"
					<< " cooColIndices has "
					<< distance(cci.begin(), cci.end())
					<< " elements.",
					""
				);
				TBTKAssert(
					distance(
						cv.begin(),
						cv.end()
					) == numMatrixElements,
					"HoppingAmplitudeSet::HoppingAmplitudeSet()",
					"Incompatible array sizes."
					<< " 'numMatrixElements' is "
					<< numMatrixElements << " but"
					<< " cooValues "
					<< distance(cv.begin(), cv.end())
					<< " elements.",
					""
				);

				cooRowIndices = new int[numMatrixElements];
				cooColIndices = new int[numMatrixElements];
				cooValues = new complex<double>[numMatrixElements];

				unsigned int counter = 0;
				for(json::iterator it = cri.begin(); it < cri.end(); ++it){
					cooRowIndices[counter] = *it;
					counter++;
				}
				counter = 0;
				for(json::iterator it = cci.begin(); it < cci.end(); ++it){
					cooColIndices[counter] = *it;
					counter++;
				}
				counter = 0;
				for(json::iterator it = cv.begin(); it < cv.end(); ++it){
					deserialize(*it, &cooValues[counter], mode);
					counter++;
				}
			}
		}
		catch(json::exception e){
			TBTKExit(
				"HoppingAmplitudeSet::HoppingAmplitudeSet()",
				"Unable to parse string as HoppingAmplitudeSet"
				<< " '" << serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"HoppingAmplitudeSet::HoppingAmplitudeSet()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

HoppingAmplitudeSet::~HoppingAmplitudeSet(){
	if(cooRowIndices != NULL)
		delete [] cooRowIndices;
	if(cooColIndices != NULL)
		delete [] cooColIndices;
	if(cooValues != NULL)
		delete [] cooValues;
}

HoppingAmplitudeSet& HoppingAmplitudeSet::operator=(
	const HoppingAmplitudeSet &rhs
){
	if(this != &rhs){
		hoppingAmplitudeTree = rhs.hoppingAmplitudeTree;
		isConstructed = rhs.isConstructed;
		isSorted = rhs.isSorted;
		numMatrixElements = rhs.numMatrixElements;

		if(numMatrixElements == -1){
			cooRowIndices = nullptr;
			cooColIndices = nullptr;
			cooValues = nullptr;
		}
		else{
			cooRowIndices = new int[numMatrixElements];
			cooColIndices = new int[numMatrixElements];
			cooValues = new complex<double>[numMatrixElements];
			for(int n = 0; n < numMatrixElements; n++){
				cooRowIndices[n] = rhs.cooRowIndices[n];
				cooColIndices[n] = rhs.cooColIndices[n];
				cooValues[n] = rhs.cooValues[n];
			}
		}
	}

	return *this;
}

HoppingAmplitudeSet& HoppingAmplitudeSet::operator=(HoppingAmplitudeSet &&rhs){
	if(this != &rhs){
		hoppingAmplitudeTree = rhs.hoppingAmplitudeTree;
		isConstructed = rhs.isConstructed;
		isSorted = rhs.isSorted;
		numMatrixElements = rhs.numMatrixElements;

		cooRowIndices = rhs.cooRowIndices;
		rhs.cooRowIndices = nullptr;

		cooColIndices = rhs.cooColIndices;
		rhs.cooColIndices = nullptr;

		cooValues = rhs.cooValues;
		rhs.cooValues = nullptr;
	}

	return *this;
}

int HoppingAmplitudeSet::getNumMatrixElements() const{
	TBTKAssert(
		numMatrixElements != -1,
		"HoppingAmplitudeSet::getNumMatrixElements()",
		"COO format not constructed.",
		"Use Model::constructCOO() to construct COO format."
	);

	return numMatrixElements;
}

void HoppingAmplitudeSet::constructCOO(){
	TBTKAssert(
		isSorted,
		"HoppingAmplitudeSet::constructCOO()",
		"Amplitudes not sorted.",
		""
	);
	TBTKAssert(
		numMatrixElements == -1,
		"HoppingAmplitudeSet::constructCOO()",
		"Hamiltonain on COO format already constructed.",
		""
	);

	//Count number of matrix elements
	HoppingAmplitudeSet::Iterator it = getIterator();
	const HoppingAmplitude *ha;
	numMatrixElements = 0;
	int currentCol = -1;
	int currentRow = -1;
	while((ha = it.getHA())){
/*		int col = getBasisIndex(ha->fromIndex);
		int row = getBasisIndex(ha->toIndex);*/
		int col = getBasisIndex(ha->getFromIndex());
		int row = getBasisIndex(ha->getToIndex());
		if(col > currentCol){
			currentCol = col;
			currentRow = -1;
		}
		if(row > currentRow){
			currentRow = row;
			numMatrixElements++;
		}

		it.searchNextHA();
	}

	cooRowIndices = new int[numMatrixElements];
	cooColIndices = new int[numMatrixElements];
	cooValues = new complex<double>[numMatrixElements];

	//Setup matrix on COO format
	it.reset();
	int currentMatrixElement = -1;
	currentCol = -1;
	currentRow = -1;
	while((ha = it.getHA())){
/*		int col = getBasisIndex(ha->fromIndex);
		int row = getBasisIndex(ha->toIndex);*/
		int col = getBasisIndex(ha->getFromIndex());
		int row = getBasisIndex(ha->getToIndex());
		complex<double> amplitude = ha->getAmplitude();

		if(col > currentCol){
			currentCol = col;
			currentRow = -1;
		}
		if(row > currentRow){
			currentRow = row;
			currentMatrixElement++;

			//Note: The sorted HoppingAmplitudeSet is in ordered
			//column major order, while the COO format is in row
			//major order. The Hermitian conjugat eis therefore
			//taken here. (That is, conjugate and intercahnge of
			//rows and columns is intentional)
			cooRowIndices[currentMatrixElement] = col;
			cooColIndices[currentMatrixElement] = row;
			cooValues[currentMatrixElement] = conj(amplitude);
		}
		else{
			cooValues[currentMatrixElement] += conj(amplitude);
		}

		it.searchNextHA();
	}
}

void HoppingAmplitudeSet::destructCOO(){
	numMatrixElements = -1;
	if(cooRowIndices != NULL){
		delete [] cooRowIndices;
		cooRowIndices = NULL;
	}
	if(cooColIndices != NULL){
		delete [] cooColIndices;
		cooColIndices = NULL;
	}
	if(cooValues != NULL){
		delete [] cooValues;
		cooValues = NULL;
	}
}

void HoppingAmplitudeSet::reconstructCOO(){
	if(numMatrixElements != -1){
		destructCOO();
		constructCOO();
	}
}

void HoppingAmplitudeSet::print(){
	hoppingAmplitudeTree.print();
}

HoppingAmplitudeSet::Iterator HoppingAmplitudeSet::getIterator() const{
	return HoppingAmplitudeSet::Iterator(&hoppingAmplitudeTree);
}

HoppingAmplitudeSet::Iterator HoppingAmplitudeSet::getIterator(
	const Index &subspace
) const{
	return HoppingAmplitudeSet::Iterator(hoppingAmplitudeTree.getSubTree(subspace));
}

string HoppingAmplitudeSet::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "HoppingAmplitudeSet(";
		ss << hoppingAmplitudeTree.serialize(mode);
		ss << "," << Serializeable::serialize(isConstructed, mode);
		ss << "," << Serializeable::serialize(isSorted, mode);
		ss << "," << Serializeable::serialize(numMatrixElements, mode);
		if(numMatrixElements != -1){
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializeable::serialize(
					cooRowIndices[n],
					mode
				);
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializeable::serialize(
					cooColIndices[n],
					mode
				);
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializeable::serialize(
					cooValues[n],
					mode
				);
			}
		}
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		json j;
		j["id"] = "HoppingAmplitudeSet";
		j["hoppingAmplitudeTree"] = json::parse(
			hoppingAmplitudeTree.serialize(mode)
		);
		j["isConstructed"] = isConstructed;
		j["isSorted"] = isSorted;
		j["numMatrixElements"] = numMatrixElements;
		if(numMatrixElements != -1){
			for(int n = 0; n < numMatrixElements; n++){
				j["cooRowIndices"].push_back(cooRowIndices[n]);
				j["cooColIndices"].push_back(cooColIndices[n]);
				j["cooValues"].push_back(
					Serializeable::serialize(
						cooValues[n],
						mode
					)
				);
			}
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"HoppingAmplitudeSet::serialize()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

HoppingAmplitudeSet::Iterator::Iterator(const HoppingAmplitudeTree* hoppingAmplitudeTree){
	it = new HoppingAmplitudeTree::Iterator(hoppingAmplitudeTree);
}

HoppingAmplitudeSet::Iterator::~Iterator(){
	delete it;
}

void HoppingAmplitudeSet::Iterator::reset(){
	it->reset();
}

void HoppingAmplitudeSet::Iterator::searchNextHA(){
	it->searchNextHA();
}

const HoppingAmplitude* HoppingAmplitudeSet::Iterator::getHA() const{
	return it->getHA();
}

int HoppingAmplitudeSet::Iterator::getMinBasisIndex() const{
	return it->getMinBasisIndex();
}

int HoppingAmplitudeSet::Iterator::getMaxBasisIndex() const{
	return it->getMaxBasisIndex();
}

int HoppingAmplitudeSet::Iterator::getNumBasisIndices() const{
	return it->getNumBasisIndices();
}

void HoppingAmplitudeSet::tabulate(
	complex<double> **amplitudes,
	int **table,
	int *numHoppingAmplitudes,
	int *maxIndexSize
) const{
	Iterator it = getIterator();
	const HoppingAmplitude *ha;
	(*numHoppingAmplitudes) = 0;
	(*maxIndexSize) = 0;
	while((ha = it.getHA())){
		(*numHoppingAmplitudes)++;

//		int indexSize = ha->fromIndex.size();
		int indexSize = ha->getFromIndex().getSize();
		if(indexSize > *maxIndexSize)
			(*maxIndexSize) = indexSize;

		it.searchNextHA();
	}

	int tableSize = (*numHoppingAmplitudes)*2*(*maxIndexSize);
	(*table) = new int[tableSize];
	for(int n = 0; n < tableSize; n++)
		(*table)[n] = -1;
	(*amplitudes) = new complex<double>[(*numHoppingAmplitudes)];

	it.reset();
	int counter = 0;
	while((ha = it.getHA())){
//		for(unsigned int n = 0; n < ha->fromIndex.size(); n++)
		for(unsigned int n = 0; n < ha->getFromIndex().getSize(); n++)
			(*table)[2*(*maxIndexSize)*counter+n] = ha->getFromIndex().at(n);
//			(*table)[2*(*maxIndexSize)*counter+n] = ha->fromIndex.at(n);
//		for(unsigned int n = 0; n < ha->toIndex.size(); n++)
		for(unsigned int n = 0; n < ha->getToIndex().getSize(); n++)
			(*table)[2*(*maxIndexSize)*counter+n+(*maxIndexSize)] = ha->getToIndex().at(n);
//			(*table)[2*(*maxIndexSize)*counter+n+(*maxIndexSize)] = ha->toIndex.at(n);
		(*amplitudes)[counter] = ha->getAmplitude();

		it.searchNextHA();
		counter++;
	}
}

};	//End of namespace TBTK
