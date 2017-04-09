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
) :
	AbstractProperty(dimensions, ranges, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	AbstractProperty(dimensions, ranges, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution
) :
	AbstractProperty(indexTree, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	AbstractProperty(indexTree, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const LDOS &ldos
) :
	AbstractProperty(ldos)
{
	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;
}

LDOS::LDOS(
	LDOS &&ldos
) :
	AbstractProperty(std::move(ldos))
{
	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;
}

LDOS::~LDOS(){
}

LDOS& LDOS::operator=(const LDOS &rhs){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

LDOS& LDOS::operator=(LDOS &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
