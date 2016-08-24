/** @file Density.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Density.h"

namespace TBTK{
namespace Property{

Density::Density(int rank, const int *dims){
	this->rank = rank;
	this->dims = new int[rank];
	for(int n = 0; n < rank; n++)
		this->dims[n] = dims[n];

	size = 1;
	for(int n = 0; n < rank; n++)
		size *= dims[n];

	data = new double[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Density::~Density(){
	delete [] this->dims;
	delete [] this->data;
}

};	//End of namespace Property
};	//End of namespace TBTK
