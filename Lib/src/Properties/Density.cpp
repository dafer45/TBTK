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

/** @file Density.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Density.h"

namespace TBTK{
namespace Property{

Density::Density(int dimensions, const int *ranges){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 1;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Density::Density(int dimensions, const int *ranges, const double *data){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 1;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	this->data = new double[size];
	for(int n = 0; n < size; n++)
		this->data[n] = data[n];
}

Density::Density(const Density &density){
	dimensions = density.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = density.ranges[n];

	size = density.size;

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = density.data[n];
}

Density::Density(Density &&density){
	dimensions = density.dimensions;
	ranges = density.ranges;
	density.ranges = nullptr;

	size = density.size;

	data = density.data;
	density.data = nullptr;
}

Density::~Density(){
	if(ranges != nullptr)
		delete [] ranges;
	if(data != nullptr)
		delete [] data;
}

Density& Density::operator=(const Density &rhs){
	dimensions = rhs.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = rhs.ranges[n];

	size = rhs.size;

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = rhs.data[n];

	return *this;
}

Density& Density::operator=(Density &&rhs){
	if(this != &rhs){
		dimensions = rhs.dimensions;
		ranges = rhs.ranges;
		rhs.ranges = nullptr;

		size = rhs.size;

		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
