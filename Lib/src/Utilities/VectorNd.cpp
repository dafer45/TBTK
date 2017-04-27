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

/** @file VectorNd.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTKMacros.h"
#include "VectorNd.h"

using namespace std;

namespace TBTK{

VectorNd::VectorNd(unsigned int size){
	this->size = size;
	data = new double[size];
}

VectorNd::VectorNd(initializer_list<double> components){
	size = components.size();
	data = new double[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = *(components.begin() + n);
}

VectorNd::VectorNd(const vector<double> &components){
	size = components.size();
	data = new double[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = components.at(n);
}

VectorNd::VectorNd(const VectorNd &vectorNd){
	size = vectorNd.size;
	data = new double[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = vectorNd.data[n];
}

VectorNd::VectorNd(VectorNd &&vectorNd){
	size = vectorNd.size;
	data = vectorNd.data;
	vectorNd.data = nullptr;
}

VectorNd::~VectorNd(){
	if(data != nullptr)
		delete [] data;
}

VectorNd& VectorNd::operator=(const VectorNd &rhs){
	if(this != &rhs){
		size = rhs.size;
		data = new double[size];
		for(unsigned int n = 0; n < size; n++)
			data[n] = rhs.data[n];
	}

	return *this;
}

VectorNd& VectorNd::operator=(VectorNd &&rhs){
	if(this != &rhs){
		size = rhs.size;
		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

};	//End of namespace TBTK
