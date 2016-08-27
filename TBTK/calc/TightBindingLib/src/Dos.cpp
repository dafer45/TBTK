/** @file Dos.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Dos.h"

namespace TBTK{
namespace Property{

Dos::Dos(double lowerBound, double upperBound, int resolution){
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
	data = new double[resolution];
}

Dos::~Dos(){
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
