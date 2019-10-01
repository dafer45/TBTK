/* Copyright 2019 Kristofer Björnson
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

/** @file SquareLattice.cpp
 *  @author Kristofer Björnson
 */

#include "TBTK/Models/SquareLattice.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Models{

SquareLattice::SquareLattice(
	const vector<unsigned int> &size,
	const vector<complex<double>> &parameters
){
	TBTKAssert(
		size.size() == 2,
		"Models::SquareLattice::SquareLattice()",
		"'size' must have exactly '2' components, but '" << size.size()
		<< "' found.",
		""
	);
	TBTKAssert(
		parameters.size() == 2,
		"Models::SquareLattice::SquareLattice()",
		"'parameters' must have exactly '2' components, but '"
		<< size.size() << "' found.",
		""
	);
	for(unsigned int x = 0; x < size[0]; x++){
		for(unsigned int y = 0; y < size[1]; y++){
			*this << HoppingAmplitude(
				parameters[0],
				{x, y},
				{x, y}
			);
			if(x+1 < size[0]){
				*this << HoppingAmplitude(
					parameters[1],
					{x+1, y},
					{x, y}
				) + HC;
			}
			if(y+1 < size[1]){
				*this << HoppingAmplitude(
					parameters[1],
					{x, y+1},
					{x, y}
				) + HC;
			}
		}
	}
}

}
};
