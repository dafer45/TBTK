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
using namespace nlohmann;

namespace TBTK{

SingleParticleContext::SingleParticleContext(){
	statistics = Statistics::FermiDirac;
	geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const vector<unsigned int> &capacity
) :
	HoppingAmplitudeSet(capacity)
{
	statistics = Statistics::FermiDirac;
	geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const SingleParticleContext &singleParticleContext
) :
	HoppingAmplitudeSet(singleParticleContext)
{
	statistics = singleParticleContext.statistics;
	if(singleParticleContext.geometry == nullptr){
		geometry = nullptr;
	}
	else{
		geometry = new Geometry(
			*singleParticleContext.geometry
		);
	}
}

SingleParticleContext::SingleParticleContext(
	SingleParticleContext &&singleParticleContext
) :
	HoppingAmplitudeSet(singleParticleContext)
{
	statistics = singleParticleContext.statistics;

	geometry = singleParticleContext.geometry;
	singleParticleContext.geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const string &serialization,
	Mode mode
) :
	HoppingAmplitudeSet(
		extractComponent(
			serialization,
			"SingleParticleContext",
			"HoppingAmplitudeSet",
			"hoppingAmplitudeSet",
			mode
		),
		mode
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
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		deserialize(elements.at(0), &statistics, mode);
		if(elements.at(2).compare("null") == 0)
			geometry = nullptr;
		else
			geometry = new Geometry(elements.at(2), mode, *this);

		break;
	}
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			deserialize(
				j.at("statistics").get<string>(),
				&statistics,
				mode
			);
			try{
				geometry = new Geometry(
					j.at("geometry").dump(),
					mode,
					*this
				);
			}
			catch(json::exception e){
				geometry = nullptr;
			}
		}
		catch(json::exception e){
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

SingleParticleContext::~SingleParticleContext(){
	if(geometry != nullptr)
		delete geometry;
}

SingleParticleContext& SingleParticleContext::operator=(
	const SingleParticleContext &rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;

		HoppingAmplitudeSet::operator=(rhs);

		if(rhs.geometry == nullptr)
			geometry = nullptr;
		else
			geometry = new Geometry(*rhs.geometry);
	}

	return *this;
}

SingleParticleContext& SingleParticleContext::operator=(
	SingleParticleContext &&rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;

		HoppingAmplitudeSet::operator=(rhs);

		geometry = rhs.geometry;
		rhs.geometry = nullptr;
	}

	return *this;
}

void SingleParticleContext::createGeometry(int dimensions, int numSpecifiers){
	TBTKAssert(
		getIsConstructed(),
		"SingleParticleContext::createGeometry()",
		"Hilbert space basis has not been constructed yet.",
		""
	);

	geometry = new Geometry(
		dimensions,
		numSpecifiers,
		this
	);
}

string SingleParticleContext::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "SingleParticleContext(";
		ss << Serializable::serialize(statistics, mode);
		ss << "," << HoppingAmplitudeSet::serialize(mode);
		if(geometry == nullptr)
			ss << "," << "null";
		else
			ss << "," << geometry->serialize(mode);
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		json j;
		j["id"] = "SingleParticleContext";
		j["statistics"] = Serializable::serialize(statistics, mode);
		j["hoppingAmplitudeSet"] = json::parse(
			HoppingAmplitudeSet::serialize(mode)
		);
		if(geometry != nullptr)
			j["geometry"] = json::parse(geometry->serialize(mode));

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
