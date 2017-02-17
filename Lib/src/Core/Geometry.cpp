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

/** @Geometry.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Geometry.h"
#include "Streams.h"
#include "TBTKMacros.h"

using namespace std;

namespace TBTK{

Geometry::Geometry(int dimensions, int numSpecifiers, const HoppingAmplitudeSet *hoppingAmplitudeSet){
	this->dimensions = dimensions;
	this->numSpecifiers = numSpecifiers;
	this->hoppingAmplitudeSet = hoppingAmplitudeSet;

	coordinates = new double[dimensions*hoppingAmplitudeSet->getBasisSize()];
	if(numSpecifiers != 0)
		specifiers = new int[numSpecifiers*hoppingAmplitudeSet->getBasisSize()];
	else
		specifiers = NULL;
}

Geometry::~Geometry(){
	delete [] coordinates;
	if(specifiers != NULL)
		delete [] specifiers;
}

void Geometry::setCoordinates(
	const Index &index,
	std::initializer_list<double> coordinates,
	std::initializer_list<int> specifiers
){
	int basisIndex = hoppingAmplitudeSet->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	const Index &index,
	const std::vector<double> &coordinates,
	const std::vector<int> &specifiers
){
	int basisIndex = hoppingAmplitudeSet->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	int basisIndex,
	std::initializer_list<double> coordinates,
	std::initializer_list<int> specifiers
){
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	int basisIndex,
	const std::vector<double> &coordinates,
	const std::vector<int> &specifiers
){
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::translate(initializer_list<double> translation){
	if(translation.size() != dimensions){
		TBTKExit(
			"Geometry::translate()",
			"The number of dimensions of the translation vector (" << translation.size() << ") does not match the dimension of the geometry (" << dimensions << ").",
			""
		);
	}

	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		for(unsigned int c = 0; c < dimensions; c++){
			coordinates[n*dimensions + c] += *(translation.begin() + c);
		}
	}
}

double Geometry::getDistance(const Index &index1, const Index &index2) const{
	int basisIndex1 = hoppingAmplitudeSet->getBasisIndex(index1);
	int basisIndex2 = hoppingAmplitudeSet->getBasisIndex(index2);

	double distanceSquared = 0.;
	for(unsigned int n = 0; n < dimensions; n++){
		double difference = coordinates[dimensions*basisIndex1 + n] - coordinates[dimensions*basisIndex2 + n];
		distanceSquared += difference*difference;
	}

	return sqrt(distanceSquared);
}

};	//End of namespace TBTK
