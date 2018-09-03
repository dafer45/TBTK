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
//using namespace nlohmann;

namespace TBTK{

SingleParticleContext::SingleParticleContext(){
	statistics = Statistics::FermiDirac;
//	geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const vector<unsigned int> &capacity
) :
	hoppingAmplitudeSet(capacity)
{
	statistics = Statistics::FermiDirac;
//	geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const SingleParticleContext &singleParticleContext
) :
	hoppingAmplitudeSet(singleParticleContext.getHoppingAmplitudeSet())
{
	statistics = singleParticleContext.statistics;
	geometry = singleParticleContext.geometry;
/*	if(singleParticleContext.geometry == nullptr){
		geometry = nullptr;
	}
	else{
		geometry = new Geometry(
			*singleParticleContext.geometry
		);
	}*/
}

SingleParticleContext::SingleParticleContext(
	SingleParticleContext &&singleParticleContext
) :
	hoppingAmplitudeSet(singleParticleContext.getHoppingAmplitudeSet())
{
	statistics = singleParticleContext.statistics;

	geometry = std::move(singleParticleContext.geometry);
//	singleParticleContext.geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const string &serialization,
	Mode mode
)/* :
	HoppingAmplitudeSet(
		extractComponent(
			serialization,
			"SingleParticleContext",
			"HoppingAmplitudeSet",
			"hoppingAmplitudeSet",
			mode
		),
		mode
	)*/
{
	TBTKAssert(
		validate(serialization, "SingleParticleContext", mode),
		"SingleParticleContext::SingleParticleContext()",
		"Unable to parse string as SingleParticleContext '"
		<< serialization << "'.",
		""
	);

	switch(mode){
/*	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		deserialize(elements.at(0), &statistics, mode);
		hoppingAmplitudeSet = HoppingAmplitudeSet(elements.at(1), mode);
		if(elements.at(2).compare("null") == 0)
			geometry = nullptr;
		else
			geometry = new Geometry(elements.at(2), mode);
//			geometry = new Geometry(elements.at(2), mode, hoppingAmplitudeSet);

		break;
	}*/
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
/*			try{
				geometry = new Geometry(
					j.at("geometry").dump(),
					mode
				);
			}
			catch(nlohmann::json::exception e){
				geometry = nullptr;
			}*/
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

SingleParticleContext::~SingleParticleContext(){
//	if(geometry != nullptr)
//		delete geometry;
}

SingleParticleContext& SingleParticleContext::operator=(
	const SingleParticleContext &rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;

		hoppingAmplitudeSet = rhs.getHoppingAmplitudeSet();

		geometry = rhs.geometry;
/*		if(rhs.geometry == nullptr)
			geometry = nullptr;
		else
			geometry = new Geometry(*rhs.geometry);*/
	}

	return *this;
}

SingleParticleContext& SingleParticleContext::operator=(
	SingleParticleContext &&rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;

		hoppingAmplitudeSet = rhs.getHoppingAmplitudeSet();

		geometry = std::move(rhs.geometry);
//		rhs.geometry = nullptr;
	}

	return *this;
}

/*void SingleParticleContext::createGeometry(int dimensions, int numSpecifiers){
	TBTKAssert(
		hoppingAmplitudeSet.getIsConstructed(),
		"SingleParticleContext::createGeometry()",
		"Hilbert space basis has not been constructed yet.",
		""
	);

	geometry = new Geometry(
	);
}*/

string SingleParticleContext::serialize(Mode mode) const{
	switch(mode){
/*	case Mode::Debug:
	{
		stringstream ss;
		ss << "SingleParticleContext(";
		ss << Serializable::serialize(statistics, mode);
		ss << "," << hoppingAmplitudeSet.serialize(mode);
		if(geometry == nullptr)
			ss << "," << "null";
		else
			ss << "," << geometry->serialize(mode);
		ss << ")";

		return ss.str();
	}*/
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "SingleParticleContext";
		j["statistics"] = Serializable::serialize(statistics, mode);
		j["hoppingAmplitudeSet"] = nlohmann::json::parse(
			hoppingAmplitudeSet.serialize(mode)
		);
/*		if(geometry != nullptr)
			j["geometry"] = nlohmann::json::parse(geometry->serialize(mode));*/
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
