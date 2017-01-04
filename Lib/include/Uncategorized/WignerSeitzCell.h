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
 *  @brief Wiegner-Seitz cell.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WIGNER_SEITZ_CELL
#define COM_DAFER45_TBTK_WIGNER_SEITZ_CELL

#include "Index.h"
#include "Vector3d.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

/** Wigner-Seitz cell. */
class WignerSeitzCell{
public:
	/** Enum used to indicate whether the mesh should be nodal or interior.
	 *  A nodal mesh is a mesh resulting from dividing a Wigner-Seitz cell
	 *  into equispaced elements and identifying the mesh with the nodal
	 *  grid points. An interior mesh is a mesh that results from a similar
	 *  division but where the mesh is identified with the central point of
	 *  the line/area/volume elements of the grid. Note that interior
	 *  points never fall on the bounding surface of the Wigner-Seitz cell,
	 *  while nodal points does. Such boundary terms are included in the
	 *  nodal mesh. */
	enum class MeshType {Nodal, Interior};

	/** Constructor. */
	WignerSeitzCell(std::initializer_list<std::initializer_list<double>> basisVectors);

	/** Constructor. */
	WignerSeitzCell(const std::vector<std::vector<double>> &basisVectors);

	/** Destructor. */
	~WignerSeitzCell();

	/** Get number of dimensions. */
	unsigned int getNumDimensions() const;

	/** Returns an equispaced mesh covering the Wigner-Seitz cell, using
	 *  numMeshPoints mesh points along the corresponding directions. For
	 *  odd values, the mesh contains the points given by
	 *  "m*basisVector_n/numMeshPoints_n + other vectors", where m is an
	 *  integer, and n is a basis vector index. For even values, the mesh
	 *  contains the points given by
	 *  "(m+1/2)*basisVectors_n/numMeshPoints_n + other vectors". */
	std::vector<std::vector<double>> getMesh(
		std::initializer_list<unsigned int> numMeshPoints,
		MeshType meshType
	) const;
private:
	/** Lattice dimension. */
	unsigned int dimensions;

	/** Basis vectors stored as three Vector3d. For lower dimensions the
	 *  vectors are padded with zeros to create three-dimensional vectors
	 *  and additional vectors along the extra dimensions are setup to
	 *  unify the calculations for all dimensions between 1-3. */
	std::vector<Vector3d> basisVectors;

	/** Constant used to give a margin to the Wigner-Seitz cell. If not
	 *  used, boundary terms could be included or not included basen on
	 *  numerical fluctuations. */
	static constexpr double ROUNDOFF_MARGIN_MULTIPLIER = 1.000001;
};

inline unsigned int WignerSeitzCell::getNumDimensions() const{
	return dimensions;
}

};	//End namespace TBTK

#endif
