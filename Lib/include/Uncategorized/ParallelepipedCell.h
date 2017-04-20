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
 *  @file ParallelepipedCell.h
 *  @brief Parallelepiped cell.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARALLELEPIPED_CELL
#define COM_DAFER45_TBTK_PARALLELEPIPED_CELL

#include "Index.h"
#include "Vector3d.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

/** Parallelepiped cell. */
class ParallelepipedCell{
public:
	/** Constructor. */
	ParallelepipedCell(std::initializer_list<std::initializer_list<double>> basisVectors);

	/** Constructor. */
	ParallelepipedCell(const std::vector<std::vector<double>> &basisVectors);

	/** Destructor. */
	~ParallelepipedCell();

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate. */
	Index getCellIndex(
		std::initializer_list<double> coordinate
	) const;

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate. */
	Index getCellIndex(
		const std::vector<double> &coordinate
	) const;

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate, where the parallelepiped has been subdivided into
	 *  smaller cells specified by numMeshPoints. */
	Index getCellIndex(
		std::initializer_list<double> coordinate,
		std::initializer_list<unsigned int> numMeshPoints
	) const;

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate, where the parallelepiped has been subdivided into
	 *  smaller cells specified by numMeshPoints. */
	Index getCellIndex(
		const std::vector<double> &coordinate,
		std::initializer_list<unsigned int> numMeshPoints
	) const;

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate, where the parallelepiped has been subdivided into
	 *  smaller cells specified by numMeshPoints. */
	Index getCellIndex(
		std::initializer_list<double> coordinate,
		const std::vector<unsigned int> &numMeshPoints
	) const;

	/** Returns the index of the parallelepiped cell corresponding to the
	 *  given coordinate, where the parallelepiped has been subdivided into
	 *  smaller cells specified by numMeshPoints. */
	Index getCellIndex(
		const std::vector<double> &coordinate,
		const std::vector<unsigned int> &numMeshPoints
	) const;

	/** Returns an equispaced mesh covering the parallelepiped cell, using
	 *  numMeshPoints mesh points along the corresponding directions. For
	 *  odd values, the mesh contains the points given by
	 *  "m*basisVector_n/numMeshPoints_n + other vectors", where m is an
	 *  integer, and n is a basis vector index. For even values, the mesh
	 *  contains the points given by
	 *  "(m+1/2)*basisVectors_n/numMeshPoints_n + other vectors". */
	std::vector<std::vector<double>> getMesh(
		std::initializer_list<unsigned int> numMeshPoints
	) const;

	/** Returns an equispaced mesh covering the parallelepiped cell, using
	 *  numMeshPoints mesh points along the corresponding directions. For
	 *  odd values, the mesh contains the points given by
	 *  "m*basisVector_n/numMeshPoints_n + other vectors", where m is an
	 *  integer, and n is a basis vector index. For even values, the mesh
	 *  contains the points given by
	 *  "(m+1/2)*basisVectors_n/numMeshPoints_n + other vectors". */
	std::vector<std::vector<double>> getMesh(
		std::vector<unsigned int> numMeshPoints
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
