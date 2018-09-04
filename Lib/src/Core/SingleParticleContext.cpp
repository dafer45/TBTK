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

/** @file SingleParticleContext.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/SingleParticleContext.h"

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{

SingleParticleContext::SingleParticleContext(){
	statistics = Statistics::FermiDirac;
}

SingleParticleContext::SingleParticleContext(
	const vector<unsigned int> &capacity
) :
	hoppingAmplitudeSet(capacity)
{
	statistics = Statistics::FermiDirac;
}

SingleParticleContext::SingleParticleContext(
	const string &serialization,
	Mode mode
)
{
	TBTKAssert(
		validate(serialization, "SingleParticleContext", mode),
		"SingleParticleContext::SingleParticleContext()",
		"Unable to parse string as SingleParticleContext '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			deserialize(
				j.at("statistics").get<string>(),
				&statistics,
				mode
			);
			hoppingAmplitudeSet = HoppingAmplitudeSet(
				j.at("hoppingAmplitudeSet").dump(),
				mode
			);
			geometry = Geometry(
				j.at("geometry").dump(),
				mode
			);
			sourceAmplitudeSet = SourceAmplitudeSet(
				j.at("sourceAmplitudeSet").dump(),
				mode
			);
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"SingleParticleContext::SingleParticleContext()",
				"Unable to parse string as"
				<< " SingleParticleContext '" << serialization
				<< "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"SingleParticleContext::SingleParticleContext()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

string SingleParticleContext::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "SingleParticleContext";
		j["statistics"] = Serializable::serialize(statistics, mode);
		j["hoppingAmplitudeSet"] = nlohmann::json::parse(
			hoppingAmplitudeSet.serialize(mode)
		);
		j["geometry"] = nlohmann::json::parse(
			geometry.serialize(mode)
		);
		j["sourceAmplitudeSet"] = nlohmann::json::parse(
			sourceAmplitudeSet.serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"SingleParticleContext::serialize()",
			"Only Serializable::Mode::Debugis supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
