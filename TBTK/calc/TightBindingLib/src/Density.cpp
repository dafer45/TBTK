/** @file Density.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Density.h"

namespace TBTK{
namespace Property{

Density::Density(int dimensions, const int *ranges){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 1;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Density::~Density(){
	delete [] this->ranges;
	delete [] this->data;
}

};	//End of namespace Property
};	//End of namespace TBTK
