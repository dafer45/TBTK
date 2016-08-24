/** @file SpinPolarizedLdos.h
 *
 *  @author Kristofer Björnson
 */

#include "../include/SpinPolarizedLdos.h"

using namespace std;

namespace TBTK{
namespace Property{

SpinPolarizedLdos::SpinPolarizedLdos(int rank, const int *dims, double lowerLimit, double upperLimit, int resolution){
	this->rank = rank;
	this->dims = new int[rank];
	for(int n = 0; n < rank; n++)
		this->dims[n] = dims[n];

	this->lowerLimit = lowerLimit;
	this->upperLimit = upperLimit;
	this->resolution = resolution;

	size = 4*resolution;
	for(int n = 0; n < rank; n++)
		size *= dims[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

SpinPolarizedLdos::~SpinPolarizedLdos(){
	delete [] dims;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
