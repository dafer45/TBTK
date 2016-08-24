/** @file Ldos.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Ldos.h"

namespace TBTK{
namespace Property{

Ldos::Ldos(int rank, const int *dims, double lowerLimit, double upperLimit, int resolution){
	this->rank = rank;
	this->dims = new int[rank];
	for(int n = 0; n < rank; n++)
		this->dims[n] = dims[n];

	this->lowerLimit = lowerLimit;
	this->upperLimit = upperLimit;
	this->resolution = resolution;

	size = resolution;
	for(int n = 0; n < rank; n++)
		size *= dims[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Ldos::~Ldos(){
	delete [] dims;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
