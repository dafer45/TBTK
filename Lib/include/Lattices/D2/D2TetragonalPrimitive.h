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

/** @package TBTKcalc
 *  @file TetragonalPrimitive.h
 *  @brief Tetragonal primitive Bravais lattices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TETRAGONAL_PRIMITIVE
#define COM_DAFER45_TBTK_TETRAGONAL_PRIMITIVE

#include "D2OrthorhombicPrimitive.h"

namespace TBTK{
namespace Lattice{
namespace D2{

/** Tetragonal primitive Bravais lattice.
 *
 *  Dimensions:		2
 *  side1Length:	arbitrary
 *  side2Length:	side1Length
 *  angle12:		pi/2 */
class TetragonalPrimitive : public OrthorhombicPrimitive{
public:
	/** Constructor. */
	TetragonalPrimitive(
		double side1Length
	);

	/** Destructor. */
	~TetragonalPrimitive();
};

};	//End of namespace D2
};	//End of namespace Lattice
};	//End of namespace TBTK

#endif
