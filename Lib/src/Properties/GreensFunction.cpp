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

GreensFunction::GreensFunction(
	Type type,
	double lowerBound,
	double upperBound,
	unsigned int resolution
){
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
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
){
	this->type = type;

	storage.arrayFormat.lowerBound = lowerBound;
	storage.arrayFormat.upperBound = upperBound;
	storage.arrayFormat.resolution = resolution;
	storage.arrayFormat.data = new complex<double>[resolution];
	for(unsigned int n = 0; n < resolution; n++)
		storage.arrayFormat.data[n] = data[n];
}

GreensFunction::GreensFunction(const GreensFunction &greensFunction){
	type = greensFunction.type;
	storage.arrayFormat.lowerBound = greensFunction.storage.arrayFormat.lowerBound;
	storage.arrayFormat.upperBound = greensFunction.storage.arrayFormat.upperBound;
	storage.arrayFormat.resolution = greensFunction.storage.arrayFormat.resolution;
	storage.arrayFormat.data = new complex<double>[storage.arrayFormat.resolution];
	for(unsigned int n = 0; n < storage.arrayFormat.resolution; n++)
		storage.arrayFormat.data[n] = greensFunction.storage.arrayFormat.data[n];
}

GreensFunction::GreensFunction(GreensFunction &&greensFunction){
	type = greensFunction.type;
	storage.arrayFormat.lowerBound = greensFunction.storage.arrayFormat.lowerBound;
	storage.arrayFormat.upperBound = greensFunction.storage.arrayFormat.upperBound;
	storage.arrayFormat.resolution = greensFunction.storage.arrayFormat.resolution;
	storage.arrayFormat.data = greensFunction.storage.arrayFormat.data;
	greensFunction.storage.arrayFormat.data = nullptr;
}

GreensFunction::~GreensFunction(){
	if(storage.arrayFormat.data != nullptr)
		delete [] storage.arrayFormat.data;
}

complex<double> GreensFunction::operator()(double E) const{
	int e = (int)((E - storage.arrayFormat.lowerBound)/(storage.arrayFormat.upperBound - storage.arrayFormat.lowerBound)*(double)storage.arrayFormat.resolution);
	TBTKAssert(
		e >= 0 && e < (int)storage.arrayFormat.resolution,
		"GreensFunction::operator()",
		"Out of bound access for Green's function of format Format::Array.",
		"Use Format::Poles or only access values inside the bounds."
	);

	return storage.arrayFormat.data[e];
}

};	//End of namespace Property
};	//End of namespace TBTK
