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

/** @file CubicPrimitive.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Lattice/D3/CubicPrimitive.h"

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D3{

CubicPrimitive::CubicPrimitive(double side0Length) :
	TetragonalPrimitive(
		side0Length,
		side0Length
	)
{
}

CubicPrimitive::~CubicPrimitive(){
}

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK
