/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file ExtensiveBitRegister.cpp
 *
 *  @author Kristofer Björnson
 */


#include "TBTK/ExtensiveBitRegister.h"
#include "TBTK/Streams.h"

namespace TBTK{

ExtensiveBitRegister::ExtensiveBitRegister(){
	values = nullptr;
}

ExtensiveBitRegister::ExtensiveBitRegister(unsigned int numBits){
	size = (numBits-1)/(8*sizeof(unsigned int))+1;
	values = new unsigned int[size];
}

ExtensiveBitRegister::ExtensiveBitRegister(const ExtensiveBitRegister &extensiveBitRegister){
	size = extensiveBitRegister.size;
	values = new unsigned int[size];
	for(unsigned int n = 0; n < size; n++)
		values[n] = extensiveBitRegister.values[n];
}

ExtensiveBitRegister::~ExtensiveBitRegister(){
	if(values != nullptr)
		delete [] values;
}

};	//End of namespace TBTK
