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
#include "TBTK/UnitHandler.h"

#include <fstream>
#include <cmath>
#include <sstream>
#include <string>

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{

Model::Model() : Communicator(false){
	temperature = 0.;
	chemicalPotential = 0.;

	manyParticleContext = NULL;
	indexFilter = nullptr;
	hoppingAmplitudeFilter = nullptr;
}

Model::Model(
	const vector<unsigned int> &capacity
) :
	Communicator(true),
	singleParticleContext(capacity)
{
	temperature = 0.;
	chemicalPotential = 0.;

	manyParticleContext = NULL;
	indexFilter = nullptr;
	hoppingAmplitudeFilter = nullptr;
}

Model::Model(const Model &model) : Communicator(model){
	temperature = model.temperature;
	chemicalPotential = model.chemicalPotential;

	singleParticleContext = model.singleParticleContext;
	if(model.manyParticleContext == nullptr){
		manyParticleContext = nullptr;
	}
	else{
		manyParticleContext = new ManyParticleContext(
			*model.manyParticleContext
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

	singleParticleContext = std::move(model.singleParticleContext);
	manyParticleContext = model.manyParticleContext;
	model.manyParticleContext = nullptr;

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
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			temperature = j.at("temperature").get<double>();
			chemicalPotential = j.at(
				"chemicalPotential"
			).get<double>();
			singleParticleContext = SingleParticleContext(
				j.at("singleParticleContext").dump(),
				mode
			);

			manyParticleContext = nullptr;

			indexFilter = nullptr;
			hoppingAmplitudeFilter = nullptr;
		}
		catch(nlohmann::json::exception &e){
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
	if(manyParticleContext != nullptr)
		delete manyParticleContext;
	if(indexFilter != nullptr)
		delete indexFilter;
	if(hoppingAmplitudeFilter != nullptr)
		delete hoppingAmplitudeFilter;
}

Model& Model::operator=(const Model &rhs){
	if(this != &rhs){
		temperature = rhs.temperature;
		chemicalPotential = rhs.chemicalPotential;

		singleParticleContext = rhs.singleParticleContext;

		if(manyParticleContext != nullptr)
			delete manyParticleContext;
		if(rhs.manyParticleContext == nullptr){
			manyParticleContext = nullptr;
		}
		else{
			manyParticleContext = new ManyParticleContext(
				*rhs.manyParticleContext
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

		singleParticleContext = std::move(rhs.singleParticleContext);

		if(manyParticleContext != nullptr)
			delete manyParticleContext;
		manyParticleContext = rhs.manyParticleContext;
		rhs.manyParticleContext = nullptr;

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

	singleParticleContext.getHoppingAmplitudeSet().construct();

	int basisSize = getBasisSize();

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\tBasis size: " << basisSize << "\n";
}

ostream& operator<<(ostream &stream, const Model &model){
	stream << model.toString();

	return stream;
}

string Model::toString() const{
	stringstream stream;
	stream << "Model\n";
	stream << "\tTemperature: "
		<< UnitHandler::convertTemperatureNaturalToBase(temperature)
		<< UnitHandler::getTemperatureUnitString() << " ("
		<< temperature << " b.u.)\n";
	stream << "\tChemical potential: "
		<< UnitHandler::convertEnergyNaturalToBase(chemicalPotential)
		<< UnitHandler::getEnergyUnitString() << " ("
		<< chemicalPotential << " b.u.)\n";
	switch(singleParticleContext.getStatistics()){
	case Statistics::FermiDirac:
		stream << "\tStatistics: Fermi-Dirac\n";
		break;
	case Statistics::BoseEinstein:
		stream << "\tStatistics: Bose-Einstein\n";
		break;
	default:
		TBTKExit(
			"Model::operator<<()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}
	int basisSize = getBasisSize();
	if(basisSize == -1){
		stream << "\tBasis size: Not yet constructed.";
	}
	else{
		stream << "\tBasis size: " << basisSize;
	}

	return stream.str();
}

string Model::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Model";
		j["temperature"] = temperature;
		j["chemicalPotential"] = chemicalPotential;
		j["singleParticleContext"] = nlohmann::json::parse(
			singleParticleContext.serialize(mode)
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
