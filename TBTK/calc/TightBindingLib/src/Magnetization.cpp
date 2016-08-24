/** @file Magnetization.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Magnetization.h"

using namespace std;

namespace TBTK{
namespace Property{

Magnetization::Magnetization(int dimensions, const int* ranges){
	this->dimensions = dimensions;
	this->ranges = new int[dimensions];
	for(int n = 0; n < dimensions; n++)
		this->ranges[n] = ranges[n];

	size = 4;
	for(int n = 0; n < dimensions; n++)
		size *= ranges[n];

	data = new complex<double>[size];
	for(int n = 0; n < size; n++)
		data[n] = 0.;
}

Magnetization::~Magnetization(){
	delete [] ranges;
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
