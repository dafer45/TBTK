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
 *  @file WignerSeitzCell.h
 *  @brief Wigner-Seitz cell.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WIGNER_SEITZ_CELL
#define COM_DAFER45_TBTK_WIGNER_SEITZ_CELL

#include "Vector3d.h"
#include "Index.h"

#include <vector>
#include <initializer_list>

namespace TBTK{

/** Wigner-Seitz cell. */
class WignerSeitzCell{
public:
	/** Constructor. */
	WignerSeitzCell(std::initializer_list<std::initializer_list<double>> basisVectors);

	/** Constructor. */
	WignerSeitzCell(const std::vector<std::vector<double>> &basisVectors);

	/** Destructor. */
	~WignerSeitzCell();

	/** Returns the index of the Wigner-Seitz cell corresponding to the
	 *  given coordinate. */
	Index getCellIndex(
		std::initializer_list<double> coordinate
	) const;
private:
	/** Lattice dimension. */
	unsigned dimensions;

	/** Basis vectors stored as three Vector3d. For lower dimensions the
	 *  vectors are padded with zeros to create three-dimensional vectors
	 *  and additional vectors along the extra dimensions are setup to unify
	 *  the calculations for all dimensions between 1-3. */
	std::vector<Vector3d> basisVectors;

	/** Vectors pointing along the normal direction of the two other basis
	 *  vectors, and with reciprocal normal length of the corresponding
	 *  basis vector. A scalar product between a coordinate vector and this
	 *  vector gives the number of the corresponding basis vectors that is
	 *  needed to reach the coordinate. */
	std::vector<Vector3d> reciprocalNormals;
};

};	//End namespace TBTK

#endif
