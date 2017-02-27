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

/** @file Magnetization.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Magnetization.h"

using namespace std;

namespace TBTK{
namespace Property{

Magnetization::Magnetization(int dimensions, const int* ranges){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 4;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Magnetization::Magnetization(
	int dimensions,
	const int* ranges,
	const complex<double> *data
){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 4;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	this->data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		this->data[n] = data[n];
}

Magnetization::Magnetization(const Magnetization &magnetization){
	dimensions = magnetization.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = magnetization.ranges[n];

	size = magnetization.size;

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = magnetization.data[n];
}

Magnetization::Magnetization(Magnetization &&magnetization){
	dimensions = magnetization.dimensions;
	ranges = magnetization.ranges;
	magnetization.ranges = nullptr;

	size = magnetization.size;

	data = magnetization.data;
	magnetization.data = nullptr;
}

Magnetization::~Magnetization(){
	if(ranges != nullptr)
		delete [] ranges;
	if(data != nullptr)
		delete [] data;
}

Magnetization& Magnetization::operator=(const Magnetization &rhs){
	dimensions = rhs.dimensions;
	ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		ranges[n] = rhs.ranges[n];

	size = rhs.size;

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = rhs.data[n];

	return *this;
}

Magnetization& Magnetization::operator=(Magnetization &&rhs){
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
