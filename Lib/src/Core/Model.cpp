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

#include "TBTK/AbstractHoppingAmplitudeFilter.h"
#include "TBTK/Geometry.h"
#include "TBTK/Model.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <fstream>
#include <math.h>
#include <string>

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

Model::Model() : Communicator(true){
	temperature = 0.;
	chemicalPotential = 0.;

	singleParticleContext = new SingleParticleContext();
	manyBodyContext = NULL;
	indexFilter = nullptr;
	hoppingAmplitudeFilter = nullptr;
}

Model::Model(const vector<unsigned int> &capacity) : Communicator(true){
	temperature = 0.;
	chemicalPotential = 0.;

	singleParticleContext = new SingleParticleContext(capacity);
	manyBodyContext = NULL;
	indexFilter = nullptr;
	hoppingAmplitudeFilter = nullptr;
}

Model::Model(const Model &model) : Communicator(model){
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

	if(model.indexFilter == nullptr)
		indexFilter = nullptr;
	else
		indexFilter = model.indexFilter->clone();

	if(model.hoppingAmplitudeFilter == nullptr)
		hoppingAmplitudeFilter = nullptr;
	else
		hoppingAmplitudeFilter = model.hoppingAmplitudeFilter->clone();
}

Model::Model(Model &&model) : Communicator(std::move(model)){
	temperature = model.temperature;
	chemicalPotential = model.chemicalPotential;

	singleParticleContext = model.singleParticleContext;
	model.singleParticleContext = nullptr;
	manyBodyContext = model.manyBodyContext;
	model.manyBodyContext = nullptr;

	indexFilter = model.indexFilter;
	model.indexFilter = nullptr;

	hoppingAmplitudeFilter = model.hoppingAmplitudeFilter;
	model.hoppingAmplitudeFilter = nullptr;
}

Model::Model(const string &serialization, Mode mode) : Communicator(true){
	TBTKAssert(
		validate(serialization, "Model", mode),
		"Model::Model()",
		"Unable to parse string as Model '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		deserialize(elements.at(0), &temperature, mode);
		deserialize(elements.at(1), &chemicalPotential, mode);
		singleParticleContext = new SingleParticleContext(
			elements.at(2),
			mode
		);

		manyBodyContext = nullptr;

		indexFilter = nullptr;
		hoppingAmplitudeFilter = nullptr;

		break;
	}
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			temperature = j.at("temperature").get<double>();
			chemicalPotential = j.at(
				"chemicalPotential"
			).get<double>();
			singleParticleContext = new SingleParticleContext(
				j.at("singleParticleContext").dump(),
				mode
			);

			manyBodyContext = nullptr;

			indexFilter = nullptr;
			hoppingAmplitudeFilter = nullptr;
		}
		catch(nlohmann::json::exception e){
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
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

Model::~Model(){
	if(singleParticleContext != nullptr)
		delete singleParticleContext;
	if(manyBodyContext != nullptr)
		delete manyBodyContext;
	if(indexFilter != nullptr)
		delete indexFilter;
	if(hoppingAmplitudeFilter != nullptr)
		delete hoppingAmplitudeFilter;
}

Model& Model::operator=(const Model &rhs){
	if(this != &rhs){
		temperature = rhs.temperature;
		chemicalPotential = rhs.chemicalPotential;

		if(singleParticleContext != nullptr)
			delete singleParticleContext;
		singleParticleContext = new SingleParticleContext(
			*rhs.singleParticleContext
		);

		if(manyBodyContext != nullptr)
			delete manyBodyContext;
		if(rhs.manyBodyContext == nullptr){
			manyBodyContext = nullptr;
		}
		else{
			manyBodyContext = new ManyBodyContext(
				*rhs.manyBodyContext
			);
		}

		if(indexFilter != nullptr)
			delete indexFilter;
		if(rhs.indexFilter == nullptr){
			indexFilter = nullptr;
		}
		else{
			indexFilter = rhs.indexFilter->clone();
		}

		if(hoppingAmplitudeFilter != nullptr)
			delete hoppingAmplitudeFilter;
		if(rhs.hoppingAmplitudeFilter == nullptr){
			hoppingAmplitudeFilter = nullptr;
		}
		else{
			hoppingAmplitudeFilter
				= rhs.hoppingAmplitudeFilter->clone();
		}
	}

	return *this;
}

Model& Model::operator=(Model &&rhs){
	if(this != &rhs){
		temperature = rhs.temperature;
		chemicalPotential = rhs.chemicalPotential;

		if(singleParticleContext != nullptr)
			delete singleParticleContext;
		singleParticleContext = rhs.singleParticleContext;
		rhs.singleParticleContext = nullptr;

		if(manyBodyContext != nullptr)
			delete manyBodyContext;
		manyBodyContext = rhs.manyBodyContext;
		rhs.manyBodyContext = nullptr;

		if(indexFilter != nullptr)
			delete indexFilter;
		indexFilter = rhs.indexFilter;
		rhs.indexFilter = nullptr;

		if(hoppingAmplitudeFilter != nullptr)
			delete hoppingAmplitudeFilter;
		hoppingAmplitudeFilter = rhs.hoppingAmplitudeFilter;
		rhs.hoppingAmplitudeFilter = nullptr;
	}

	return *this;
}

void Model::addModel(const Model &model, const Index &index){
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= model.getHoppingAmplitudeSet().cbegin();
		iterator != model.getHoppingAmplitudeSet().cend();
		++iterator
	){
		add(
			HoppingAmplitude(
				(*iterator).getAmplitude(),
				Index(index, (*iterator).getToIndex()),
				Index(index, (*iterator).getFromIndex())
			)
		);
	}
}

void Model::construct(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Constructing system\n";

	singleParticleContext->getHoppingAmplitudeSet().construct();

	int basisSize = getBasisSize();

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\tBasis size: " << basisSize << "\n";
}

string Model::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "Model(";
		ss << Serializable::serialize(temperature, mode);
		ss << "," << Serializable::serialize(chemicalPotential, mode);
		ss << "," << singleParticleContext->serialize(mode);
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Model";
		j["temperature"] = temperature;
		j["chemicalPotential"] = chemicalPotential;
		j["singleParticleContext"] = nlohmann::json::parse(
			singleParticleContext->serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Model::serialize()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
