/* Copyright 2017 Kristofer Björnson
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

/** @file GreensFunction.cpp
 *
 *  @author Kristofer Björnson
 */

#include "GreensFunction.h"
#include "TBTKMacros.h"

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace Property{

double GreensFunction::EPSILON = DEFAULT_EPSILON;

GreensFunction::GreensFunction(
	Type type,
	Format format,
	double lowerBound,
	double upperBound,
	unsigned int resolution
){
	TBTKAssert(
		format == Format::Array,
		"GreensFunction::GreensFunction()",
		"The given constructor can only be called for format == Format::Array.",
		"Change format or Green's function constructor."
	);

	TBTKAssert(
		type == Type::Retarded || type == Type::Advanced || type == Type::Principal || type == Type::NonPrincipal,
		"GreensFunction::GreensFunction()",
		"Unssuported format and type combination.",
		"Format::Array only support Type::Retarded, Type::Advanced, Type::Principal, and Type::NonPrincipal."
	);

	this->format = format;
	this->type = type;

	storage.arrayFormat.lowerBound = lowerBound;
	storage.arrayFormat.upperBound = upperBound;
	storage.arrayFormat.resolution = resolution;
	storage.arrayFormat.data = new complex<double>[resolution];
	for(unsigned int n = 0; n < resolution; n++)
		storage.arrayFormat.data[n] = 0.;
}

GreensFunction::GreensFunction(
	Type type,
	Format format,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
){
	TBTKAssert(
		format == Format::Array,
		"GreensFunction::GreensFunction()",
		"The given constructor can only be called for format == Format::Array.",
		"Change format or Green's function constructor."
	);

	TBTKAssert(
		type == Type::Retarded || type == Type::Advanced || type == Type::Principal || type == Type::NonPrincipal,
		"GreensFunction::GreensFunction()",
		"Unssuported format and type combination.",
		"Format::Array only support Type::Retarded, Type::Advanced, Type::Principal, and Type::NonPrincipal."
	);

	this->format = format;
	this->type = type;

	storage.arrayFormat.lowerBound = lowerBound;
	storage.arrayFormat.upperBound = upperBound;
	storage.arrayFormat.resolution = resolution;
	storage.arrayFormat.data = new complex<double>[resolution];
	for(unsigned int n = 0; n < resolution; n++)
		storage.arrayFormat.data[n] = data[n];
}

GreensFunction::GreensFunction(
	Type type,
	Format format,
	unsigned int numPoles
){
	TBTKAssert(
		format == Format::Poles,
		"GreensFunction::GreensFunction()",
		"The given constructor can only be called for format == Format::Poles.",
		"Change format or Green's function constructor."
	);

	TBTKAssert(
		type == Type::Retarded || type == Type::Advanced || type == Type::FreePole,
		"GreensFunction::GreensFunction()",
		"Unssuported format and type combination.",
		"Format::Poles only support Type::Retarded, Type::Advanced, Type::FreePole."
	);

	this->format = format;
	this->type = type;

	storage.poleFormat.numPoles = numPoles;
	storage.poleFormat.positions = new complex<double>[numPoles];
	storage.poleFormat.amplitudes = new complex<double>[numPoles];
	for(unsigned int n = 0; n < numPoles; n++){
		storage.poleFormat.positions[n] = 0.;
		storage.poleFormat.amplitudes[n] = 0.;
	}
}

GreensFunction::GreensFunction(
	Type type,
	Format format,
	unsigned int numPoles,
	complex<double> *positions,
	complex<double> *amplitudes
){
	TBTKAssert(
		format == Format::Poles,
		"GreensFunction::GreensFunction()",
		"The given constructor can only be called for format == Format::Poles.",
		"Change format or Green's function constructor."
	);

	TBTKAssert(
		type == Type::Retarded || type == Type::Advanced || type == Type::FreePole,
		"GreensFunction::GreensFunction()",
		"Unssuported format and type combination.",
		"Format::Poles only support Type::Retarded, Type::Advanced, Type::FreePole."
	);

	this->format = format;
	this->type = type;

	storage.poleFormat.numPoles = numPoles;
	storage.poleFormat.positions = new complex<double>[numPoles];
	storage.poleFormat.amplitudes = new complex<double>[numPoles];
	for(unsigned int n = 0; n < numPoles; n++){
		storage.poleFormat.positions[n] = positions[n];
		storage.poleFormat.amplitudes[n] = amplitudes[n];
	}
}

GreensFunction::GreensFunction(const GreensFunction &greensFunction){
	type = greensFunction.type;
	format = greensFunction.format;
	switch(format){
	case Format::Array:
		storage.arrayFormat.lowerBound = greensFunction.storage.arrayFormat.lowerBound;
		storage.arrayFormat.upperBound = greensFunction.storage.arrayFormat.upperBound;
		storage.arrayFormat.resolution = greensFunction.storage.arrayFormat.resolution;
		storage.arrayFormat.data = new complex<double>[storage.arrayFormat.resolution];
		for(unsigned int n = 0; n < storage.arrayFormat.resolution; n++)
			storage.arrayFormat.data[n] = greensFunction.storage.arrayFormat.data[n];
		break;
	case Format::Poles:
		storage.poleFormat.numPoles = greensFunction.storage.poleFormat.numPoles;
		storage.poleFormat.positions = new complex<double>[storage.poleFormat.numPoles];
		storage.poleFormat.amplitudes = new complex<double>[storage.poleFormat.numPoles];
		for(unsigned int n = 0; n < storage.poleFormat.numPoles; n++){
			storage.poleFormat.positions[n] = greensFunction.storage.poleFormat.positions[n];
			storage.poleFormat.amplitudes[n] = greensFunction.storage.poleFormat.amplitudes[n];
		}
		break;
	default:
		TBTKExit(
			"GreensFunction::operator()",
			"Unknown Green's function format.",
			"This should never happen, contact the developer."
		);
	}
}

GreensFunction::GreensFunction(GreensFunction &&greensFunction){
	type = greensFunction.type;
	format = greensFunction.format;
	switch(format){
	case Format::Array:
		storage.arrayFormat.lowerBound = greensFunction.storage.arrayFormat.lowerBound;
		storage.arrayFormat.upperBound = greensFunction.storage.arrayFormat.upperBound;
		storage.arrayFormat.resolution = greensFunction.storage.arrayFormat.resolution;
		storage.arrayFormat.data = greensFunction.storage.arrayFormat.data;
		greensFunction.storage.arrayFormat.data = nullptr;
		break;
	case Format::Poles:
		storage.poleFormat.numPoles = greensFunction.storage.poleFormat.numPoles;
		storage.poleFormat.positions = greensFunction.storage.poleFormat.positions;
		greensFunction.storage.poleFormat.positions = nullptr;
		storage.poleFormat.amplitudes = greensFunction.storage.poleFormat.amplitudes;
		greensFunction.storage.poleFormat.amplitudes = nullptr;
		break;
	default:
		TBTKExit(
			"GreensFunction::operator()",
			"Unknown Green's function format.",
			"This should never happen, contact the developer."
		);
	}
}

GreensFunction::~GreensFunction(){
	if(format == Format::Array){
		if(storage.arrayFormat.data != nullptr)
			delete [] storage.arrayFormat.data;
	}
	if(format == Format::Poles){
		if(storage.poleFormat.positions != nullptr)
			delete [] storage.poleFormat.positions;
		if(storage.poleFormat.amplitudes != nullptr)
			delete [] storage.poleFormat.amplitudes;
	}
}

complex<double> GreensFunction::operator()(double E) const{
	switch(format){
	case Format::Array:
	{
		int e = (int)((E - storage.arrayFormat.lowerBound)/(storage.arrayFormat.upperBound - storage.arrayFormat.lowerBound)*(double)storage.arrayFormat.resolution);
		TBTKAssert(
			e >= 0 && e < (int)storage.arrayFormat.resolution,
			"GreensFunction::operator()",
			"Out of bound access for Green's function of format Format::Array.",
			"Use Format::Poles or only access values inside the bounds."
		);

		return storage.arrayFormat.data[e];
	}
	case Format::Poles:
	{
		double epsilon;
		switch(type){
		case Type::Retarded:
			epsilon = EPSILON;
			break;
		case Type::Advanced:
			epsilon = -EPSILON;
			break;
		case Type::FreePole:
			epsilon = 0.;
			break;
		default:
			TBTKExit(
				"GreensFunction::operator()",
				"Unsupported Green's function format and type combination.",
				"This should never happen, contact the developer."
			);
		}

		complex<double> value = 0.;
		for(unsigned int n = 0; n < storage.poleFormat.numPoles; n++)
			value += storage.poleFormat.amplitudes[n]/(E - storage.poleFormat.positions[n] + i*epsilon);

		return value;
	}
	default:
		TBTKExit(
			"GreensFunction::operator()",
			"Unknown Green's function format.",
			"This should never happen, contact the developer."
		);
	}
}

complex<double> GreensFunction::operator()(complex<double> E) const{
	switch(format){
	case Format::Array:
	{
		TBTKExit(
			"GreensFunction::operator()",
			"Unsupported Green's function format.",
			"Only a Green's function with the format Format::Poles can be evaluated at a complex energy."
		);
	}
	case Format::Poles:
	{
		double epsilon;
		switch(type){
		case Type::Retarded:
			epsilon = EPSILON;
			break;
		case Type::Advanced:
			epsilon = -EPSILON;
			break;
		case Type::FreePole:
			epsilon = 0.;
			break;
		default:
			TBTKExit(
				"GreensFunction::operator()",
				"Unsupported Green's function format and type combination.",
				"This should never happen, contact the developer."
			);
		}

		complex<double> value = 0.;
		for(unsigned int n = 0; n < storage.poleFormat.numPoles; n++)
			value += storage.poleFormat.amplitudes[n]/(E - storage.poleFormat.positions[n] + i*epsilon);

		return value;
	}
	default:
		TBTKExit(
			"GreensFunction::operator()",
			"Unknown Green's function format.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
