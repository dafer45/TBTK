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

/** @file BoundaryCondition.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/BoundaryCondition.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

BoundaryCondition::BoundaryCondition(
	const string &serialization,
	Serializable::Mode mode
){
	TBTKAssert(
		Serializable::validate(
			serialization,
			"BoundaryCondition",
			mode
		),
		"BoundaryCondition::BoundaryCondition()",
		"Unable to parse string as BoundaryCondition '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			hoppingAmplitudeList = HoppingAmplitudeList(
				j["hoppingAmplitudeList"].dump(),
				mode
			);
			sourceAmplitude = SourceAmplitude(
				j["sourceAmplitude"].dump(),
				mode
			);
			eliminationIndex = Index(
				j["eliminationIndex"].dump(),
				mode
			);
		}
		catch(json::exception e){
			TBTKExit(
				"BoundaryCondition::BoundaryCondition()",
				"Unable to parse string as BoundaryCondition '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"BoundaryCondition::BoundaryCondition()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string BoundaryCondition::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		json j;
		j["id"] = "BoundaryCondition";
		j["hoppingAmplitudeList"] = json::parse(hoppingAmplitudeList.serialize(mode));
		j["sourceAmplitude"] = json::parse(sourceAmplitude.serialize(mode));
		j["eliminationIndex"] = json::parse(eliminationIndex.serialize(mode));

		return j.dump();
	}
	default:
		TBTKExit(
			"BoundaryCondition::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
