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

#include "TBTK/Property/GreensFunction.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Property{

GreensFunction::GreensFunction() : AbstractProperty(){
}

GreensFunction::GreensFunction(
	const IndexTree &indexTree,
	Type type,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	AbstractProperty(indexTree, resolution)
{
	this->type = type;

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
//	this->data = new complex<double>[resolution];
//	for(unsigned int n = 0; n < resolution; n++)
//		this->data[n] = 0.;
}

GreensFunction::GreensFunction(
	const IndexTree &indexTree,
	Type type,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
) :
	AbstractProperty(indexTree, resolution, data)
{
	this->type = type;

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
//	this->data = new complex<double>[resolution];
//	for(unsigned int n = 0; n < resolution; n++)
//		this->data[n] = data[n];
}

/*GreensFunction::GreensFunction(
	const GreensFunction &greensFunction
) :
	AbstractProperty(greensFunction)
{
	type = greensFunction.type;
	lowerBound = greensFunction.lowerBound;
	upperBound = greensFunction.upperBound;
	resolution = greensFunction.resolution;
//	data = new complex<double>[resolution];
//	for(unsigned int n = 0; n < resolution; n++)
//		data[n] = greensFunction.data[n];
}

GreensFunction::GreensFunction(
	GreensFunction &&greensFunction
) :
	AbstractProperty(std::move(greensFunction))
{
	type = greensFunction.type;
	lowerBound = greensFunction.lowerBound;
	upperBound = greensFunction.upperBound;
	resolution = greensFunction.resolution;
//	data = greensFunction.data;
//	greensFunction.data = nullptr;
}

GreensFunction::~GreensFunction(){
//	if(data != nullptr)
//		delete [] data;
}*/

};	//End of namespace Property
};	//End of namespace TBTK
