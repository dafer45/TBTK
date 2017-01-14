#include "ExtensiveBitRegister.h"
#include "Streams.h"

namespace TBTK{

ExtensiveBitRegister::ExtensiveBitRegister(unsigned int numBits){
	size = numBits/(8*sizeof(unsigned int));
	values = new unsigned int[size];
}

ExtensiveBitRegister::ExtensiveBitRegister(const ExtensiveBitRegister &extensiveBitRegister){
	size = extensiveBitRegister.size;
	values = new unsigned int[size];
	for(unsigned int n = 0; n < size; n++)
		values[n] = extensiveBitRegister.values[n];
}

ExtensiveBitRegister::~ExtensiveBitRegister(){
	delete [] values;
}

};	//End of namespace TBTK
