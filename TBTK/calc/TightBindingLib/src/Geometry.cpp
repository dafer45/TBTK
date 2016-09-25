/** @Geometry.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Geometry.h"
#include <iostream>

using namespace std;

namespace TBTK{

Geometry::Geometry(int dimensions, int numSpecifiers, Model *parentModel){
	this->dimensions = dimensions;
	this->numSpecifiers = numSpecifiers;
	this->parentModel = parentModel;

	coordinates = new double[dimensions*parentModel->getBasisSize()];
	if(numSpecifiers != 0)
		specifiers = new int[numSpecifiers*parentModel->getBasisSize()];
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
	int basisIndex = parentModel->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		cout << "Error in Geometry::setCoordinates: Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.\n";
		exit(1);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		cout << "Error in Geometry::addPoint: Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.\n";
		exit(1);
	}
}

void Geometry::setCoordinates(
	const Index &index,
	const std::vector<double> &coordinates,
	const std::vector<int> &specifiers
){
	int basisIndex = parentModel->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		cout << "Error in Geometry::setCoordinates: Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.\n";
		exit(1);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		cout << "Error in Geometry::addPoint: Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.\n";
		exit(1);
	}
}

void Geometry::translate(initializer_list<double> translation){
	if(translation.size() != dimensions){
		cout << "Error in Geometry::translate: The number of dimensions of the translation vector (" << translation.size() << ") does not match the dimension of the geometry (" << dimensions << ").\n";
		exit(1);
	}

	for(int n = 0; n < parentModel->getBasisSize(); n++){
		for(unsigned int c = 0; c < dimensions; c++){
			coordinates[n*dimensions + c] += *(translation.begin() + c);
		}
	}
}

double Geometry::getDistance(const Index &index1, const Index &index2) const{
	int basisIndex1 = parentModel->getBasisIndex(index1);
	int basisIndex2 = parentModel->getBasisIndex(index2);

	double distanceSquared = 0.;
	for(unsigned int n = 0; n < dimensions; n++){
		double difference = coordinates[dimensions*basisIndex1 + n] - coordinates[dimensions*basisIndex2 + n];
		distanceSquared += difference*difference;
	}

	return sqrt(distanceSquared);
}

};	//End of namespace TBTK
