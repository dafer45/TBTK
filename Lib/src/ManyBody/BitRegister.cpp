#include "BitRegister.h"
#include "Streams.h"

namespace TBTK{

BitRegister::BitRegister(unsigned int numBits){
}

BitRegister::BitRegister(const BitRegister &bitRegister){
	values = bitRegister.values;
}

BitRegister::~BitRegister(){
}

};	//End of namespace TBTK
