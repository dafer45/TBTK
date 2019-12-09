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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file D3HexagonalPrimitive.h
 *  @brief Hexagonal primitive Bravais lattices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_D3_HEXAGONAL_PRIMITIVE
#define COM_DAFER45_TBTK_D3_HEXAGONAL_PRIMITIVE

#include "TBTK/Lattice/D3/TriclinicPrimitive.h"

namespace TBTK{
namespace Lattice{
namespace D3{

/** Hexagonal primitive Bravais lattice.
 *
 *  Dimensions:		3
 *  side0Length:	arbitrary
 *  side1Length:	side0Length
 *  side2Length:	arbitrary
 *  angle01:		2*pi/3
 *  angle02:		pi/2
 *  angle12:		pi/2 */
class HexagonalPrimitive : public TriclinicPrimitive{
public:
	/** Constructor. */
	HexagonalPrimitive(
		double side0Length,
		double side2Length
	);

	/** Destructor. */
	~HexagonalPrimitive();
};

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK

#endif
/// @endcond
