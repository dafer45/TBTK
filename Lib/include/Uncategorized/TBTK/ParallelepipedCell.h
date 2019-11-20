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

#include "TBTK/Index.h"
#include "TBTK/SpacePartition.h"
#include "TBTK/Vector3d.h"

#include <vector>

namespace TBTK{

/** Parallelepiped cell. */
class ParallelepipedCell : public SpacePartition{
public:
	/** Constructor. */
	ParallelepipedCell(
		const std::vector<std::vector<double>> &basisVectors,
		MeshType meshType
	);

	/** Destructor. */
	virtual ~ParallelepipedCell();

	/** Implements SpacePartition::getMajorCellIndex(). */
	virtual Index getMajorCellIndex(
		const std::vector<double> &coordinate
	) const;

	/** Implements SpacePartition::getMinorCellIndex(). */
	virtual Index getMinorCellIndex(
		const std::vector<double> &coordinate,
		const std::vector<unsigned int> &numMeshPoints
	) const;

	/** Implements SpacePartition::getMajorMesh(). */
	virtual std::vector<std::vector<double>> getMajorMesh(
		const std::vector<unsigned int> &numMeshPoints
	) const;

	/** Implements SpacePartition::getMinorMesh(). */
	virtual std::vector<std::vector<double>> getMinorMesh(
		const std::vector<unsigned int> &numMeshPoints
	) const;

	/** Implements SpacePartition::getMinorMeshPoint(). */
	virtual std::vector<double> getMinorMeshPoint(
		const std::vector<unsigned int> &meshPoint,
		const std::vector<unsigned int> &numMeshPoints
	) const;
private:
	/** Vectors pointing along the normal direction of the two other basis
	 *  vectors, and with reciprocal normal length of the corresponding
	 *  basis vector. A scalar product between a coordinate vector and this
	 *  vector gives the number of the corresponding basis vectors that is
	 *  needed to reach the coordinate. */
	std::vector<Vector3d> reciprocalNormals;
};

};	//End namespace TBTK

#endif
