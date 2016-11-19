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
 *  @file HexagonalPrimitive.h
 *  @brief Hexagonal primitive Bravais lattices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HEXAGONAL_PRIMITIVE
#define COM_DAFER45_TBTK_HEXAGONAL_PRIMITIVE

#include "MonoclinicPrimitive.h"

namespace TBTK{
namespace Lattices{

/** Hexagonal primitive Bravais lattice.
 *
 *  Dimensions:		2
 *  side1Length:	arbitrary
 *  side2Length:	side1Length
 *  angle12:		2*pi/3 */
class HexagonalPrimitive : public MonoclinicPrimitive{
public:
	/** Constructor. */
	HexagonalPrimitive(
		double side1Length
	);

	/** Destructor. */
	~HexagonalPrimitive();
};

};	//End of namespace Lattices
};	//End of namespace TBTK

#endif
