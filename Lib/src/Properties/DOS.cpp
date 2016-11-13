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

/** @file DOS.cpp
 *
 *  @author Kristofer Björnson
 */

#include "DOS.h"

namespace TBTK{
namespace Property{

DOS::DOS(double lowerBound, double upperBound, int resolution){
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
	data = new double[resolution];
	for(int n = 0; n < resolution; n++)
		data[n] = 0.;
}

DOS::DOS(double lowerBound, double upperBound, int resolution, const double *data){
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
	this->data = new double[resolution];
	for(int n = 0; n < resolution; n++)
		this->data[n] = data[n];
}

DOS::~DOS(){
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
