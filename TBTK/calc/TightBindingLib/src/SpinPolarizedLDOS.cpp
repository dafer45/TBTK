/** @file SpinPolarizedLDOS.h
 *
 *  @author Kristofer Björnson
 */

#include "../include/SpinPolarizedLDOS.h"

using namespace std;

namespace TBTK{
namespace Property{

SpinPolarizedLDOS::SpinPolarizedLDOS(int dimensions, const int *ranges, double lowerBound, double upperBound, int resolution){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;

	size = 4*resolution;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

SpinPolarizedLDOS::~SpinPolarizedLDOS(){
	delete [] ranges;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
