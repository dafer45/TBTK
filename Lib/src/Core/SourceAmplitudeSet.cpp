/* Copyright 2018 Kristofer Björnson
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

/** @file SourceAmplitudeSet.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/SourceAmplitudeSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

SourceAmplitudeSet::SourceAmplitudeSet(){
}

SourceAmplitudeSet::SourceAmplitudeSet(
	const string &serialization,
	Mode mode
){
	switch(mode){
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			sourceAmplitudeTree = IndexedDataTree<
				std::vector<SourceAmplitude>
			>(
				j.at("sourceAmplitudeTree").dump(),
				mode
			);
		}
		catch(json::exception e){
			TBTKExit(
				"SourceAmplitudeSet::SourceAmplitudeSet()",
				"Unable to parse string as SourceAmplitudeSet"
				<< " '" << serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"SourceAmplitudeSet::SourceAmplitudeSet()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

SourceAmplitudeSet::~SourceAmplitudeSet(){
}

SourceAmplitudeSet::Iterator SourceAmplitudeSet::begin(){
	return SourceAmplitudeSet::Iterator(sourceAmplitudeTree);
}

SourceAmplitudeSet::ConstIterator SourceAmplitudeSet::begin() const{
	return SourceAmplitudeSet::ConstIterator(sourceAmplitudeTree);
}

SourceAmplitudeSet::ConstIterator SourceAmplitudeSet::cbegin() const{
	return SourceAmplitudeSet::ConstIterator(sourceAmplitudeTree);
}

SourceAmplitudeSet::Iterator SourceAmplitudeSet::end(){
	return SourceAmplitudeSet::Iterator(sourceAmplitudeTree, true);
}

SourceAmplitudeSet::ConstIterator SourceAmplitudeSet::end() const{
	return SourceAmplitudeSet::ConstIterator(sourceAmplitudeTree, true);
}

SourceAmplitudeSet::ConstIterator SourceAmplitudeSet::cend() const{
	return SourceAmplitudeSet::ConstIterator(sourceAmplitudeTree, true);
}

string SourceAmplitudeSet::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		json j;
		j["id"] = "SourceAmplitudeSet";
		j["sourceAmplitudeTree"] = json::parse(
			sourceAmplitudeTree.serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"HoppingAmplitudeSet::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

/*SourceAmplitudeSet::Iterator::Iterator(
	IndexedDataTree<std::vector<SourceAmplitude>> &sourceAmplitudeTree,
	bool end
) :
	currentSourceAmplitude(0),
	iterator(end ? sourceAmplitudeTree.end() : sourceAmplitudeTree.begin()),
	iteratorEnd(sourceAmplitudeTree.end())
{
}

void SourceAmplitudeSet::Iterator::operator++(){
	if(iterator != iteratorEnd){
		std::vector<SourceAmplitude> &sourceAmplitudes = *iterator;
		if(currentSourceAmplitude+1 == sourceAmplitudes.size()){
			currentSourceAmplitude = 0;
			++iterator;
		}
		else{
			currentSourceAmplitude++;
		}
	}
}

SourceAmplitude& SourceAmplitudeSet::Iterator::operator*(){
	return (*iterator)[currentSourceAmplitude];
}

bool SourceAmplitudeSet::Iterator::operator==(const Iterator &rhs){
	if(
		iterator == rhs.iterator
		&& currentSourceAmplitude == rhs.currentSourceAmplitude
	){
		return true;
	}
	else{
		return false;
	}
}

bool SourceAmplitudeSet::Iterator::operator!=(const Iterator &rhs){
	if(
		iterator != rhs.iterator
		|| currentSourceAmplitude != rhs.currentSourceAmplitude
	){
		return true;
	}
	else{
		return false;
	}
}*/

};	//End of namespace TBTK
