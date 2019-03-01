/* Copyright 2019 Kristofer Björnson
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

/** @file OverlapAmplitudeSet.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/OverlapAmplitudeSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

OverlapAmplitudeSet::OverlapAmplitudeSet(){
}

OverlapAmplitudeSet::OverlapAmplitudeSet(
	const string &serialization,
	Mode mode
){
	switch(mode){
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			overlapAmplitudeTree
				= IndexedDataTree<OverlapAmplitude>(
					j.at("overlapAmplitudeTree").dump(),
					mode
				);
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"OverlapAmplitudeSet::OverlapAmplitudeSet()",
				"Unable to parse string as OverlapAmplitudeSet"
				<< " '" << serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"OverlapAmplitudeSet::OverlapAmplitudeSet()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

OverlapAmplitudeSet::~OverlapAmplitudeSet(){
}

OverlapAmplitudeSet::Iterator OverlapAmplitudeSet::begin(){
	return OverlapAmplitudeSet::Iterator(overlapAmplitudeTree);
}

OverlapAmplitudeSet::ConstIterator OverlapAmplitudeSet::begin() const{
	return OverlapAmplitudeSet::ConstIterator(overlapAmplitudeTree);
}

OverlapAmplitudeSet::ConstIterator OverlapAmplitudeSet::cbegin() const{
	return OverlapAmplitudeSet::ConstIterator(overlapAmplitudeTree);
}

OverlapAmplitudeSet::Iterator OverlapAmplitudeSet::end(){
	return OverlapAmplitudeSet::Iterator(overlapAmplitudeTree, true);
}

OverlapAmplitudeSet::ConstIterator OverlapAmplitudeSet::end() const{
	return OverlapAmplitudeSet::ConstIterator(overlapAmplitudeTree, true);
}

OverlapAmplitudeSet::ConstIterator OverlapAmplitudeSet::cend() const{
	return OverlapAmplitudeSet::ConstIterator(overlapAmplitudeTree, true);
}

string OverlapAmplitudeSet::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "OverlapAmplitudeSet";
		j["overlapAmplitudeTree"] = nlohmann::json::parse(
			overlapAmplitudeTree.serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"OverlapAmplitudeSet::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
