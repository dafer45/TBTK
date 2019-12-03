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

#include "TBTK/TBTKMacros.h"
#include "TBTK/VectorNd.h"

using namespace std;

namespace TBTK{

VectorNd::VectorNd(){
}

VectorNd::VectorNd(unsigned int size) : data(size){
}

VectorNd::VectorNd(
	initializer_list<double> components
) :
	data(components.size())
{
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] = *(components.begin() + n);
}

VectorNd::VectorNd(const vector<double> &components) : data(components.size()){
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] = components.at(n);
}

};	//End of namespace TBTK
