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

#include "TBTK/Index.h"
#include "TBTK/ParallelepipedCell.h"
#include "TBTK/Vector3d.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

/** Wigner-Seitz cell. */
class WignerSeitzCell : public ParallelepipedCell{
public:
	/** Constructor. */
	WignerSeitzCell(
		std::initializer_list<std::initializer_list<double>> basisVectors,
		MeshType meshType
	);

	/** Constructor. */
	WignerSeitzCell(
		const std::vector<std::vector<double>> &basisVectors,
		MeshType meshType
	);

	/** Destructor. */
	virtual ~WignerSeitzCell();

	/** Implements SpacePartition::getCellIndex(). */
	virtual Index getMajorCellIndex(
		const std::vector<double> &coordinates
	) const;

	/** Implements SpacePartition::getCellIndex(). */
	virtual Index getMinorCellIndex(
		const std::vector<double> &coordinates,
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

	/** Implements SpacePartition::getMinorMeshPoint*/
	virtual std::vector<double> getMinorMeshPoint(
		const std::vector<unsigned int> &meshPoint,
		const std::vector<unsigned int> &numMeshPoints
	) const;
private:
	/** Constant used to give a margin to the Wigner-Seitz cell. If not
	 *  used, boundary terms could be included or not included basen on
	 *  numerical fluctuations. */
//	static constexpr double ROUNDOFF_MARGIN_MULTIPLIER = 1.000001;

	/** Get cell index. */
	unsigned int getMajorCellIndexInternal(Vector3d coordinates) const;

	/** Returns the coordinate obtained after translating the coordinate
	 *  back to the first Wigner-Seitz cell. */
	Vector3d getFirstCellCoordinates(Vector3d coordinate) const;
};

inline unsigned int WignerSeitzCell::getMajorCellIndexInternal(
	Vector3d coordinates
) const{
	const std::vector<Vector3d> &basisVectors = getBasisVectors();
	double coordinatesNorm = coordinates.norm();

	unsigned int cellIndex = 0;
	switch(getNumDimensions()){
	case 1:
	{
		TBTKExit(
			"WignerSeitzCell::getMajorCellIndexInternal()",
			"Not yet implemented for 1D.",
			""
		);
	}
	case 2:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);
		Vector3d b2 = (b0*b1).unit();

		double minDistanceLine0 = Vector3d::dotProduct(b0, (b1*b2).unit());
		double minDistanceLine1 = Vector3d::dotProduct(b1, (b2*b0).unit());
		double X = abs(2*coordinatesNorm/minDistanceLine0);
		double Y = abs(2*coordinatesNorm/minDistanceLine1);

		for(int x = -X-1; x < X+1; x++){
			for(int y = -Y-1; y < Y+1; y++){
				if(x == 0 && y == 0)
					continue;

				Vector3d latticePoint
					= x*basisVectors.at(0)
					+ y*basisVectors.at(1);

				if(
					Vector3d::dotProduct(
						latticePoint,
						coordinates
					)/Vector3d::dotProduct(
						latticePoint,
						latticePoint
					) > 1/2.
				){
					cellIndex++;
				}
			}
		}

		break;
	}
	case 3:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);
		const Vector3d &b2 = basisVectors.at(2);

		double minDistancePlane0 = Vector3d::dotProduct(b0, (b1*b2).unit());
		double minDistancePlane1 = Vector3d::dotProduct(b1, (b2*b0).unit());
		double minDistancePlane2 = Vector3d::dotProduct(b2, (b0*b1).unit());
		double X = abs(2*coordinatesNorm/minDistancePlane0);
		double Y = abs(2*coordinatesNorm/minDistancePlane1);
		double Z = abs(2*coordinatesNorm/minDistancePlane2);

		for(int x = -X-1; x < X+1; x++){
			for(int y = -Y-1; y < Y+1; y++){
				for(int z = -Z-1; z < Z+1; z++){
					if(x == 0 && y == 0 && z == 0)
						continue;

					Vector3d latticePoint
						= x*basisVectors.at(0)
						+ y*basisVectors.at(1)
						+ z*basisVectors.at(2);

					if(
						Vector3d::dotProduct(
							latticePoint,
							coordinates
						)/Vector3d::dotProduct(
							latticePoint,
							latticePoint
						) > 1/2.
					){
						cellIndex++;
					}
				}
			}
		}

		break;
	}
	default:
		TBTKExit(
			"WignerSeitzCell::getCellIndex()",
			"Only coordinates with 1-3 componenents supported.",
			""
		);
	}

	return cellIndex;
}

inline Vector3d WignerSeitzCell::getFirstCellCoordinates(Vector3d coordinates) const{
	const std::vector<Vector3d> &basisVectors = getBasisVectors();
	Vector3d b[3];
	for(unsigned int n = 0; n < 3; n++)
		b[n] = basisVectors.at(n);

	for(unsigned int n = 0; n < getNumDimensions(); n++){
		double coordinatesProjection = Vector3d::dotProduct(
			coordinates,
			(b[(n+1)%3]*b[(n+2)%3]).unit()
		);
		double bProjection = Vector3d::dotProduct(
			b[n],
			(b[(n+1)%3]*b[(n+2)%3]).unit()
		);

		coordinates = coordinates - ((int)((coordinatesProjection - bProjection)/bProjection + 3/2.))*b[n];
	}

	bool done = false;
	while(!done){
		done = true;
		for(int x = -1; x <= 1; x++){
			for(int y = -1; y <= 1; y++){
				for(int z = -1; z <= 1; z++){
					const Vector3d v = x*b[0] + y*b[1] + z*b[2];

					if((coordinates + v).norm() < coordinates.norm()){
						coordinates = coordinates + v;
						done = false;
					}
				}
			}
		}
	}

	return coordinates;
}

};	//End namespace TBTK

#endif
