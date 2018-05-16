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

#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

HoppingAmplitudeSet::HoppingAmplitudeSet(){
	isConstructed = false;
	isSorted = false;
	numMatrixElements = -1;

	cooRowIndices = NULL;
	cooColIndices = NULL;
	cooValues = NULL;
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	const vector<unsigned int> &capacity
) :
	HoppingAmplitudeTree(capacity)
{
	isConstructed = false;
	isSorted = false;
	numMatrixElements = -1;

	cooRowIndices = NULL;
	cooColIndices = NULL;
	cooValues = NULL;
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	const HoppingAmplitudeSet &hoppingAmplitudeSet
) :
	HoppingAmplitudeTree(hoppingAmplitudeSet)
{
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
) :
	HoppingAmplitudeTree(hoppingAmplitudeSet)
{
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
) :
	HoppingAmplitudeTree(
		extractComponent(
			serialization,
			"HoppingAmplitudeSet",
			"HoppingAmplitudeTree",
			"hoppingAmplitudeTree",
			mode
		),
		mode
	)
{
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
			nlohmann::json j = nlohmann::json::parse(serialization);
			isConstructed = j.at("isConstructed").get<bool>();
			isSorted = j.at("isSorted").get<bool>();
			numMatrixElements = j.at("numMatrixElements").get<int>();
			if(numMatrixElements == -1){
				cooRowIndices = nullptr;
				cooColIndices = nullptr;
				cooValues = nullptr;
			}
			else{
				nlohmann::json cri = j.at("cooRowIndices");
				nlohmann::json cci = j.at("cooColIndices");
				nlohmann::json cv = j.at("cooValues");
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
				for(nlohmann::json::iterator it = cri.begin(); it < cri.end(); ++it){
					cooRowIndices[counter] = *it;
					counter++;
				}
				counter = 0;
				for(nlohmann::json::iterator it = cci.begin(); it < cci.end(); ++it){
					cooColIndices[counter] = *it;
					counter++;
				}
				counter = 0;
				for(nlohmann::json::iterator it = cv.begin(); it < cv.end(); ++it){
					deserialize(*it, &cooValues[counter], mode);
					counter++;
				}
			}
		}
		catch(nlohmann::json::exception e){
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
			"Only Serializable::Mode::Debug is supported yet.",
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
		HoppingAmplitudeTree::operator=(rhs);
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
		HoppingAmplitudeTree::operator=(rhs);
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
	numMatrixElements = 0;
	int currentCol = -1;
	int currentRow = -1;
	for(
		ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		int col = getBasisIndex((*iterator).getFromIndex());
		int row = getBasisIndex((*iterator).getToIndex());
		if(col > currentCol){
			currentCol = col;
			currentRow = -1;
		}
		if(row > currentRow){
			currentRow = row;
			numMatrixElements++;
		}
	}

	cooRowIndices = new int[numMatrixElements];
	cooColIndices = new int[numMatrixElements];
	cooValues = new complex<double>[numMatrixElements];

	//Setup matrix on COO format
	int currentMatrixElement = -1;
	currentCol = -1;
	currentRow = -1;
	for(
		ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		int col = getBasisIndex((*iterator).getFromIndex());
		int row = getBasisIndex((*iterator).getToIndex());
		complex<double> amplitude = (*iterator).getAmplitude();

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
	HoppingAmplitudeTree::print();
}

string HoppingAmplitudeSet::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "HoppingAmplitudeSet(";
		ss << HoppingAmplitudeTree::serialize(mode);
		ss << "," << Serializable::serialize(isConstructed, mode);
		ss << "," << Serializable::serialize(isSorted, mode);
		ss << "," << Serializable::serialize(numMatrixElements, mode);
		if(numMatrixElements != -1){
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializable::serialize(
					cooRowIndices[n],
					mode
				);
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializable::serialize(
					cooColIndices[n],
					mode
				);
			}
			for(int n = 0; n < numMatrixElements; n++){
				ss << "," << Serializable::serialize(
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
		nlohmann::json j;
		j["id"] = "HoppingAmplitudeSet";
		j["hoppingAmplitudeTree"] = nlohmann::json::parse(
			HoppingAmplitudeTree::serialize(mode)
		);
		j["isConstructed"] = isConstructed;
		j["isSorted"] = isSorted;
		j["numMatrixElements"] = numMatrixElements;
		if(numMatrixElements != -1){
			for(int n = 0; n < numMatrixElements; n++){
				j["cooRowIndices"].push_back(cooRowIndices[n]);
				j["cooColIndices"].push_back(cooColIndices[n]);
				j["cooValues"].push_back(
					Serializable::serialize(
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
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

void HoppingAmplitudeSet::tabulate(
	complex<double> **amplitudes,
	int **table,
	int *numHoppingAmplitudes,
	int *maxIndexSize
) const{
	(*numHoppingAmplitudes) = 0;
	(*maxIndexSize) = 0;
	for(
		ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		(*numHoppingAmplitudes)++;

		int indexSize = (*iterator).getFromIndex().getSize();
		if(indexSize > *maxIndexSize)
			(*maxIndexSize) = indexSize;
	}

	int tableSize = (*numHoppingAmplitudes)*2*(*maxIndexSize);
	(*table) = new int[tableSize];
	for(int n = 0; n < tableSize; n++)
		(*table)[n] = -1;
	(*amplitudes) = new complex<double>[(*numHoppingAmplitudes)];

	int counter = 0;
	for(
		ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		for(unsigned int n = 0; n < (*iterator).getFromIndex().getSize(); n++)
			(*table)[2*(*maxIndexSize)*counter+n] = (*iterator).getFromIndex().at(n);
		for(unsigned int n = 0; n < (*iterator).getToIndex().getSize(); n++)
			(*table)[2*(*maxIndexSize)*counter+n+(*maxIndexSize)] = (*iterator).getToIndex().at(n);
		(*amplitudes)[counter] = (*iterator).getAmplitude();

		counter++;
	}
}

};	//End of namespace TBTK
