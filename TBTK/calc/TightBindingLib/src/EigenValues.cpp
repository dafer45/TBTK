/** @file EigenValues.cpp
 *
 *  @author Kristofer Björnson
*/

#include "../include/EigenValues.h"

namespace TBTK{
namespace Property{

EigenValues::EigenValues(int size){
	this->size = size;
	data = new double[size];
}

EigenValues::~EigenValues(){
	delete [] data;
}

};	//End of namespace Property
};	//End of namespace TBTK
