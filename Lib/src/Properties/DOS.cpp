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
#include "Streams.h"

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

DOS::DOS(const DOS &dos){
	lowerBound = dos.lowerBound;
	upperBound = dos.upperBound;
	resolution = dos.resolution;
	data = new double[resolution];
	for(int n = 0; n < resolution; n++)
		data[n] = dos.data[n];
}

DOS::DOS(DOS &&dos){
	lowerBound = dos.lowerBound;
	upperBound = dos.upperBound;
	resolution = dos.resolution;
	data = dos.data;
	dos.data = nullptr;
}

DOS::~DOS(){
	if(data != nullptr)
		delete [] data;
}

DOS& DOS::operator=(const DOS &rhs){
	lowerBound = rhs.lowerBound;
	upperBound = rhs.upperBound;
	resolution = rhs.resolution;
	data = new double[resolution];
	for(int n = 0; n < resolution; n++)
		data[n] = rhs.data[n];

	return *this;
}

DOS& DOS::operator=(DOS &&rhs){
	if(this != &rhs){
		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
