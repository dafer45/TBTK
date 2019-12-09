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
 *  @file OrthorhombicFaceCentered.h
 *  @brief Orthorhombic face-centered Bravais lattices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_D3_ORTHORHOMBIC_FACE_CENTERED
#define COM_DAFER45_TBTK_D3_ORTHORHOMBIC_FACE_CENTERED

#include "TBTK/Lattice/OrthorhombicPrimitive.h"

namespace TBTK{
namespace Lattice{
namespace D3{

/** Orthorhombic face-centered Bravais lattice.
 *
 *  Dimensions:		3
 *  side0Length:	arbitrary
 *  side1Length:	arbitrary
 *  side2Length:	arbitrary
 *  angle01:		pi/2
 *  angle02:		pi/2
 *  angle12:		pi/2
 *
 *  Additional sites:
 *  (side0Length/2,	side1Length/2,	0)
 *  (side0Length/2,	0,		side2Length/2)
 *  (0,			side1Length/2,	side2Length/2) */
class OrthorhombicFaceCentered : public OrthorhombicPrimitive{
public:
	/** Constructor. */
	OrthorhombicFaceCentered(
		double side0Length,
		double side1Length,
		double side2Length
	);

	/** Destructor. */
	~OrthorhombicFaceCentered();

	/** Overrides BravaisLattice::makePrimitive(). */
	virtual void makePrimitive();
};

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK

#endif
/// @endcond
