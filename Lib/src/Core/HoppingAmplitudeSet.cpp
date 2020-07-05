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

namespace TBTK{

HoppingAmplitudeSet::HoppingAmplitudeSet(){
	isConstructed = false;
}

HoppingAmplitudeSet::HoppingAmplitudeSet(
	const vector<unsigned int> &capacity
) :
	HoppingAmplitudeTree(capacity)
{
	isConstructed = false;
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

		break;
	}
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			isConstructed = j.at("isConstructed").get<bool>();
		}
		catch(nlohmann::json::exception &e){
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
}

IndexTree HoppingAmplitudeSet::getIndexTree() const{
	IndexTree indexTree;
	for(
		ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		indexTree.add((*iterator).getFromIndex());
	}
	indexTree.generateLinearMap();

	return indexTree;
}

IndexTree HoppingAmplitudeSet::getIndexTree(const Index &subspace) const{
	IndexTree indexTree;
	for(
		ConstIterator iterator = cbegin(subspace);
		iterator != cend(subspace);
		++iterator
	){
		indexTree.add((*iterator).getFromIndex());
	}
	indexTree.generateLinearMap();

	return indexTree;
}

string HoppingAmplitudeSet::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "HoppingAmplitudeSet(";
		ss << HoppingAmplitudeTree::serialize(mode);
		ss << "," << Serializable::serialize(isConstructed, mode);
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
