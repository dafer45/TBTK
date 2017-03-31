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

/** @file SpinPolarizedLDOS.h
 *
 *  @author Kristofer Björnson
 */

#include "SpinPolarizedLDOS.h"

using namespace std;

namespace TBTK{
namespace Property{

SpinPolarizedLDOS::SpinPolarizedLDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution
) :
	indexDescriptor(IndexDescriptor::Format::Ranges)
{
	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	setSize(4*resolution*indexDescriptor.getSize());

	complex<double> *data = getDataRW();
	for(unsigned int n = 0; n < getSize(); n++)
		data[n] = 0.;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const complex<double> *data
) : indexDescriptor(IndexDescriptor::Format::Ranges)
{
	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	setSize(4*resolution*indexDescriptor.getSize());

	complex<double> *thisData = getDataRW();
	for(unsigned int n = 0; n < getSize(); n++)
		thisData[n] = data[n];
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	const SpinPolarizedLDOS &spinPolarizedLDOS
) :
	AbstractProperty(spinPolarizedLDOS),
	indexDescriptor(spinPolarizedLDOS.indexDescriptor)
{
	lowerBound = spinPolarizedLDOS.lowerBound;
	upperBound = spinPolarizedLDOS.upperBound;
	resolution = spinPolarizedLDOS.resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	SpinPolarizedLDOS &&spinPolarizedLDOS
) :
	AbstractProperty(std::move(spinPolarizedLDOS)),
	indexDescriptor(std::move(spinPolarizedLDOS.indexDescriptor))
{
	lowerBound = spinPolarizedLDOS.lowerBound;
	upperBound = spinPolarizedLDOS.upperBound;
	resolution = spinPolarizedLDOS.resolution;
}

SpinPolarizedLDOS::~SpinPolarizedLDOS(){
}

SpinPolarizedLDOS& SpinPolarizedLDOS::operator=(const SpinPolarizedLDOS &rhs){
	AbstractProperty::operator=(rhs);
	indexDescriptor = rhs.indexDescriptor;

	lowerBound = rhs.lowerBound;
	upperBound = rhs.upperBound;
	resolution = rhs.resolution;

	return *this;
}

SpinPolarizedLDOS& SpinPolarizedLDOS::operator=(SpinPolarizedLDOS &&rhs){
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
