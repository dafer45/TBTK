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

/** @file Model.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Geometry.h"
#include "Model.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <fstream>
#include <math.h>
#include <string>

#include "json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

Model::Model(){
	temperature = 0.;
	chemicalPotential = 0.;

	singleParticleContext = new SingleParticleContext();
	manyBodyContext = NULL;

	isTalkative = true;
}

Model::Model(const Model &model){
	temperature = model.temperature;
	chemicalPotential = model.chemicalPotential;

	singleParticleContext = new SingleParticleContext(
		*model.singleParticleContext
	);
	if(model.manyBodyContext == nullptr){
		manyBodyContext = nullptr;
	}
	else{
		manyBodyContext = new ManyBodyContext(
			*model.manyBodyContext
		);
	}

	isTalkative = model.isTalkative;
}

Model::Model(Model &&model){
	temperature = model.temperature;
	chemicalPotential = model.chemicalPotential;

	singleParticleContext = model.singleParticleContext;
	model.singleParticleContext = nullptr;
	manyBodyContext = model.manyBodyContext;
	model.manyBodyContext = nullptr;

	isTalkative = model.isTalkative;
}

Model::Model(const string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "Model", mode),
		"Model::Model()",
		"Unable to parse string as Model '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		deserialize(elements.at(0), &temperature, mode);
		deserialize(elements.at(1), &chemicalPotential, mode);
		singleParticleContext = new SingleParticleContext(elements.at(2), mode);
		deserialize(elements.at(3), &isTalkative, mode);

		manyBodyContext = nullptr;

		break;
	}
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			temperature = j.at("temperature").get<double>();
			chemicalPotential = j.at(
				"chemicalPotential"
			).get<double>();
			singleParticleContext = new SingleParticleContext(
				j.at("singleParticleContext").dump(),
				mode
			);
			isTalkative = j.at("isTalkative").get<bool>();
		}
		catch(json::exception e){
			TBTKExit(
				"Model::Model()",
				"Unable to parse string as Model '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Model::Model()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

Model::~Model(){
	if(singleParticleContext != nullptr)
		delete singleParticleContext;
	if(manyBodyContext != nullptr)
		delete manyBodyContext;
}

Model& Model::operator=(const Model &rhs){
	if(this != &rhs){
		temperature = rhs.temperature;
		chemicalPotential = rhs.chemicalPotential;

		singleParticleContext = new SingleParticleContext(
			*rhs.singleParticleContext
		);
		if(rhs.manyBodyContext == nullptr){
			manyBodyContext = nullptr;
		}
		else{
			manyBodyContext = new ManyBodyContext(
				*rhs.manyBodyContext
			);
		}

		isTalkative = rhs.isTalkative;
	}

	return *this;
}

Model& Model::operator=(Model &&rhs){
	if(this != &rhs){
		temperature = rhs.temperature;
		chemicalPotential = rhs.chemicalPotential;

		singleParticleContext = rhs.singleParticleContext;
		rhs.singleParticleContext = nullptr;
		manyBodyContext = rhs.manyBodyContext;
		rhs.manyBodyContext = nullptr;

		isTalkative = rhs.isTalkative;
	}

	return *this;
}

void Model::construct(){
	if(isTalkative)
		Streams::out << "Constructing system\n";

	singleParticleContext->construct();

	int basisSize = getBasisSize();

	if(isTalkative)
		Streams::out << "\tBasis size: " << basisSize << "\n";
}

string Model::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "Model(";
		ss << Serializeable::serialize(temperature, mode);
		ss << "," << Serializeable::serialize(chemicalPotential, mode);
		ss << "," << singleParticleContext->serialize(mode);
		ss << "," << Serializeable::serialize(isTalkative, mode);
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		json j;
		j["id"] = "Model";
		j["temperature"] = temperature;
		j["chemicalPotential"] = chemicalPotential;
		j["singleParticleContext"] = json::parse(
			singleParticleContext->serialize(mode)
		);
		j["isTalkative"] = isTalkative;

		return j.dump();
	}
	default:
		TBTKExit(
			"Model::serialize()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
