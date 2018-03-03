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

/** @file D3HexagonalPrimitive.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Lattice/D3/D3HexagonalPrimitive.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D3{

HexagonalPrimitive::HexagonalPrimitive(
	double side0Length,
	double side2Length
) :
	TriclinicPrimitive(
		side0Length,
		side0Length,
		side2Length,
		2.*M_PI/3.,
		M_PI/2.,
		M_PI/2.
	)
{
}

HexagonalPrimitive::~HexagonalPrimitive(){
}

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK
