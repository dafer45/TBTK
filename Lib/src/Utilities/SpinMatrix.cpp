/* Copyright 2017 Kristofer Björnson
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

/** @file SpinMatrix.cpp
 *
 *  @author Kristofer Björnson
 */

#include "SpinMatrix.h"

using namespace std;

namespace TBTK{

SpinMatrix::SpinMatrix(){
}

SpinMatrix::~SpinMatrix(){
}

SpinMatrix& SpinMatrix::operator=(complex<double> value){
	for(unsigned int row = 0; row < getNumRows(); row++){
		for(unsigned int col = 0; col < getNumCols(); col++){
			at(row, col) = value;
		}
	}

	return *this;
}

};	//End of namespace TBTK
