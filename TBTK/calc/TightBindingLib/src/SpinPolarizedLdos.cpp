/** @file SpinPolarizedLdos.h
 *
 *  @author Kristofer Björnson
 */

#include "../include/SpinPolarizedLdos.h"

using namespace std;

namespace TBTK{
namespace Property{

SpinPolarizedLdos::SpinPolarizedLdos(int dimensions, const int *ranges, double lowerLimit, double upperLimit, int resolution){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	this->lowerLimit = lowerLimit;
	this->upperLimit = upperLimit;
	this->resolution = resolution;

	size = 4*resolution;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

SpinPolarizedLdos::~SpinPolarizedLdos(){
	delete [] ranges;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
