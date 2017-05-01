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

#include "SingleParticleContext.h"

using namespace std;

namespace TBTK{

SingleParticleContext::SingleParticleContext(){
	statistics = Statistics::FermiDirac;
	hoppingAmplitudeSet = new HoppingAmplitudeSet();
	geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const SingleParticleContext &singleParticleContext
){
	statistics = singleParticleContext.statistics;
	hoppingAmplitudeSet = new HoppingAmplitudeSet(
		*singleParticleContext.hoppingAmplitudeSet
	);
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
){
	statistics = singleParticleContext.statistics;

	hoppingAmplitudeSet = singleParticleContext.hoppingAmplitudeSet;
	singleParticleContext.hoppingAmplitudeSet = nullptr;

	geometry = singleParticleContext.geometry;
	singleParticleContext.geometry = nullptr;
}

SingleParticleContext::SingleParticleContext(
	const string &serialization,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	{
		TBTKAssert(
			validate(serialization, "SingleParticleContext", mode),
			"SingleParticleContext::SingleParticleContext()",
			"Unable to parse string as SingleParticleContext '"
			<< serialization << "'.",
			""
		);
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		deserialize(elements.at(0), &statistics, mode);
		hoppingAmplitudeSet = new HoppingAmplitudeSet(elements.at(1), mode);
		if(elements.at(2).compare("null") == 0)
			geometry = nullptr;
		else
			geometry = new Geometry(elements.at(2), mode, *hoppingAmplitudeSet);

		break;
	}
	default:
		TBTKExit(
			"SingleParticleContext::SingleParticleContext()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

SingleParticleContext::~SingleParticleContext(){
	if(hoppingAmplitudeSet != nullptr)
		delete hoppingAmplitudeSet;
	if(geometry != nullptr)
		delete geometry;
}

SingleParticleContext& SingleParticleContext::operator=(
	const SingleParticleContext &rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;

		hoppingAmplitudeSet = new HoppingAmplitudeSet(
			*rhs.hoppingAmplitudeSet
		);

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

		hoppingAmplitudeSet = rhs.hoppingAmplitudeSet;
		rhs.hoppingAmplitudeSet = nullptr;

		geometry = rhs.geometry;
		rhs.geometry = nullptr;
	}

	return *this;
}

void SingleParticleContext::construct(){
	hoppingAmplitudeSet->construct();
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
		hoppingAmplitudeSet
	);
}

string SingleParticleContext::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "SingleParticleContext(";
		ss << Serializeable::serialize(statistics, mode);
		ss << "," << hoppingAmplitudeSet->serialize(mode);
		if(geometry == nullptr)
			ss << "," << "null";
		else
			ss << "," << geometry->serialize(mode);
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		stringstream ss;
		ss << "{";
		ss << "id:'SingleParticleContext'";
		ss << "," << "statistics:" << Serializeable::serialize(
			statistics,
			mode
		);
		ss << "," << "hoppingAmplitudeSet:"
			<< hoppingAmplitudeSet->serialize(mode);
		if(geometry != nullptr)
			ss << "," << "geometry:" << geometry->serialize(mode);
		ss << "}";

		return ss.str();
	}
	default:
		TBTKExit(
			"SingleParticleContext::serialize()",
			"Only Serializeable::Mode::Debugis supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
