/** @file Ldos.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Ldos.h"

namespace TBTK{
namespace Property{

Ldos::Ldos(int dimensions, const int *ranges, double lowerBound, double upperBound, int resolution){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	size = resolution;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Ldos::~Ldos(){
	delete [] ranges;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
