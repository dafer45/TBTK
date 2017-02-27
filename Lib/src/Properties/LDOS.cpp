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

/** @file LDOS.cpp
 *
 *  @author Kristofer Björnson
 */

#include "LDOS.h"

namespace TBTK{
namespace Property{

LDOS::LDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution
){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	size = resolution;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

LDOS::LDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	size = resolution;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	this->data = new double[size];
	for(int n = 0; n < size; n++)
		this->data[n] = data[n];
}

LDOS::LDOS(const LDOS &ldos){
	dimensions = ldos.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = ldos.ranges[n];

	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;

	size = ldos.size;

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = ldos.data[n];
}

LDOS::LDOS(LDOS &&ldos){
	dimensions = ldos.dimensions;
	ranges = ldos.ranges;
	ldos.ranges = nullptr;

	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;

	size = ldos.size;

	data = ldos.data;
	ldos.data = nullptr;
}

LDOS::~LDOS(){
	if(ranges != nullptr)
		delete [] ranges;
	if(data != nullptr)
		delete [] data;
}

LDOS& LDOS::operator=(const LDOS &rhs){
	dimensions = rhs.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = rhs.ranges[n];

	lowerBound = rhs.lowerBound;
	upperBound = rhs.upperBound;
	resolution = rhs.resolution;

	size = rhs.size;

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = rhs.data[n];

	return *this;
}

LDOS& LDOS::operator=(LDOS &&rhs){
	if(this != &rhs){
		dimensions = rhs.dimensions;
		ranges = rhs.ranges;
		rhs.ranges = nullptr;

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;

		size = rhs.size;

		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
