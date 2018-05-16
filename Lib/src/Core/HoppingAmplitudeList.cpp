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

/** @file HoppingAmplitudeList.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HoppingAmplitudeList.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

HoppingAmplitudeList::HoppingAmplitudeList(){
}

HoppingAmplitudeList::HoppingAmplitudeList(
	const string &serialization,
	Serializable::Mode mode
){
	TBTKAssert(
		Serializable::validate(
			serialization,
			"HoppingAmplitudeList",
			mode
		),
		"HoppingAmplitudeList::HoppingAmplitudeList()",
		"Unable to parse string as HoppingAmplitudeList '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			nlohmann::json has = j.at(
				"hoppingAmplitudes"
			);
			for(
				nlohmann::json::iterator it = has.begin();
				it != has.end();
				++it
			){
				hoppingAmplitudes.push_back(
					HoppingAmplitude(it->dump(), mode)
				);
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"HoppingAmplitudeList::HoppingAmplitudeList()",
				"Unable to parse string as HoppingAmplitudeList '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"HoppingAmplitudeList::HoppingAmplitudeList()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string HoppingAmplitudeList::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "HoppingAmplitudeList";
		for(unsigned int n = 0; n < hoppingAmplitudes.size(); n++){
			j["hoppingAmplitudes"].push_back(
				nlohmann::json::parse(
					hoppingAmplitudes[n].serialize(
						Serializable::Mode::JSON
					)
				)
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"HoppingAmplitudeList::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
