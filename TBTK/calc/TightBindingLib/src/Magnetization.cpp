/** @file Magnetization.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Magnetization.h"

using namespace std;

namespace TBTK{
namespace Property{

Magnetization::Magnetization(int rank, const int* dims){
	this->rank = rank;
	this->dims = new int[rank];
	for(int n = 0; n < rank; n++)
		this->dims[n] = dims[n];

	size = 4;
	for(int n = 0; n < rank; n++)
		size *= dims[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Magnetization::~Magnetization(){
	delete [] dims;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
