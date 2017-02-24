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

namespace TBTK{
namespace Property{

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

GreensFunction::~GreensFunction(){
	if(format == Format::Array)
		delete [] storage.arrayFormat.data;
	if(format == Format::Poles){
		delete [] storage.poleFormat.positions;
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
		complex<double> value = 0.;
		for(unsigned int n = 0; n < storage.poleFormat.numPoles; n++)
			value += storage.poleFormat.amplitudes[n]/(E - storage.poleFormat.positions[n]);

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
