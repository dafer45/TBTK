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
) : indexDescriptor(IndexDescriptor::Format::Ranges)
{
	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	setSize(resolution*indexDescriptor.getSize());

	double *data = getDataRW();
	for(unsigned int n = 0; n < getSize(); n++)
		data[n] = 0.;
}

LDOS::LDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) : indexDescriptor(IndexDescriptor::Format::Ranges){
	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	setSize(resolution*indexDescriptor.getSize());

	double *thisData = getDataRW();
	for(unsigned int n = 0; n < getSize(); n++)
		thisData[n] = data[n];
}

LDOS::LDOS(
	const LDOS &ldos
) :
	AbstractProperty(ldos),
	indexDescriptor(ldos.indexDescriptor)
{
	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;
}

LDOS::LDOS(
	LDOS &&ldos
) :
	AbstractProperty(std::move(ldos)),
	indexDescriptor(std::move(ldos.indexDescriptor))
{
	lowerBound = ldos.lowerBound;
	upperBound = ldos.upperBound;
	resolution = ldos.resolution;
}

LDOS::~LDOS(){
}

LDOS& LDOS::operator=(const LDOS &rhs){
	AbstractProperty::operator=(rhs);
	indexDescriptor = rhs.indexDescriptor;

	lowerBound = rhs.lowerBound;
	upperBound = rhs.upperBound;
	resolution = rhs.resolution;

	return *this;
}

LDOS& LDOS::operator=(LDOS &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));
		indexDescriptor = std::move(rhs.indexDescriptor);

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
