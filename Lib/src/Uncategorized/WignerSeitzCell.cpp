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

/** @file WignerSeitzCell.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/WignerSeitzCell.h"

using namespace std;

namespace TBTK{

WignerSeitzCell::WignerSeitzCell(){
}

WignerSeitzCell::WignerSeitzCell(
	const vector<vector<double>> &basisVectors,
	MeshType meshType
) :
	ParallelepipedCell(basisVectors, meshType)
{
}

WignerSeitzCell::~WignerSeitzCell(){
}

Index WignerSeitzCell::getMajorCellIndex(
	const vector<double> &coordinates
) const{
	TBTKAssert(
		coordinates.size() == getNumDimensions(),
		"ParallelepipedCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " dimension of the Wigner-Seitz cell. The Wigner-Seitz cell"
		<< " has " << getNumDimensions() << " dimensions, but"
		<< " coordinate with " << coordinates.size() << " components"
		<< " supplied."
	);

	Vector3d coordinatesVector;
	switch(coordinates.size()){
	case 1:
		coordinatesVector = Vector3d({
			coordinates.at(0),
			0,
			0
		});
		break;
	case 2:
		coordinatesVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			0
		});
		break;
	case 3:
		coordinatesVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			coordinates.at(2)
		});
		break;
	default:
		TBTKExit(
			"WignerSeitzCell::getCellIndex()",
			"Only coordinates with 1-3 components supported.",
			""
		);
	}

	return {(int)getMajorCellIndexInternal(coordinatesVector)};
}

Index WignerSeitzCell::getMinorCellIndex(
	const vector<double> &coordinates,
	const vector<unsigned int> &numMeshPoints
) const{
	TBTKAssert(
		coordinates.size() == getNumDimensions(),
		"ParallelepipedCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " dimension of the Wigner-Seitz cell. The Wigner-Seitz cell"
		<< " has " << getNumDimensions() << " dimensions, but"
		<< " coordinate with " << coordinates.size() << " components"
		<< " supplied."
	);

	Vector3d coordinatesVector;
	switch(coordinates.size()){
	case 1:
		coordinatesVector = Vector3d({
			*(coordinates.begin() + 0),
			0,
			0
		});
		break;
	case 2:
		coordinatesVector = Vector3d({
			*(coordinates.begin() + 0),
			*(coordinates.begin() + 1),
			0
		});
		break;
	case 3:
		coordinatesVector = Vector3d({
			*(coordinates.begin() + 0),
			*(coordinates.begin() + 1),
			*(coordinates.begin() + 2)
		});
		break;
	default:
		TBTKExit(
			"WignerSeitzCell::getCellIndex()",
			"Only coordinates with 1-3 components supported.",
			""
		);
	}

	unsigned int majorCellIndex = getMajorCellIndexInternal(coordinatesVector);
	Vector3d firstCellCoordinate;
	if(majorCellIndex > 0){
		coordinatesVector = getFirstCellCoordinates(coordinatesVector);
	}

	Index cellIndex = ParallelepipedCell::getMinorCellIndex(
		coordinates,
		numMeshPoints
	);
	for(unsigned int n = 0; n < cellIndex.getSize(); n++)
		cellIndex.at(n) = (cellIndex.at(n) + *(numMeshPoints.begin() + n))%(*(numMeshPoints.begin() + n));

	return cellIndex;
}

vector<vector<double>> WignerSeitzCell::getMajorMesh(
	const vector<unsigned int> &numMeshPoints
) const{
	TBTKExit(
		"WignerSeitzCell::getMajorMesh()",
		"Wigner-Seitz cell does not have a major mesh.",
		""
	);
}

vector<vector<double>> WignerSeitzCell::getMinorMesh(
	const vector<unsigned int> &numMeshPoints
) const{
	vector<vector<double>> mesh
		= ParallelepipedCell::getMinorMesh(numMeshPoints);

	switch(getNumDimensions()){
	case 1:
		for(unsigned int n = 0; n < mesh.size(); n++){
			Vector3d firstCellMeshPoint = getFirstCellCoordinates(
				{mesh.at(n).at(0), 0, 0}
			);
			mesh.at(n).at(0) = firstCellMeshPoint.x;
		}

		break;
	case 2:
		for(unsigned int n = 0; n < mesh.size(); n++){
			Vector3d firstCellMeshPoint = getFirstCellCoordinates(
				{mesh.at(n).at(0), mesh.at(n).at(1), 0}
			);
			mesh.at(n).at(0) = firstCellMeshPoint.x;
			mesh.at(n).at(1) = firstCellMeshPoint.y;
		}

		break;
	case 3:
		for(unsigned int n = 0; n < mesh.size(); n++){
			Vector3d firstCellMeshPoint = getFirstCellCoordinates(mesh.at(n));
			mesh.at(n).at(0) = firstCellMeshPoint.x;
			mesh.at(n).at(1) = firstCellMeshPoint.y;
			mesh.at(n).at(2) = firstCellMeshPoint.z;
		}

		break;
	default:
		TBTKExit(
			"Parallelepiped::getMesh()",
			"This should never happed.",
			"Notify the developer about this bug."
		);
	}

	return mesh;
}

vector<double> WignerSeitzCell::getMinorMeshPoint(
	const vector<unsigned int> &meshPoint,
	const vector<unsigned int> &numMeshPoints
) const{
	vector<double> mp = ParallelepipedCell::getMinorMeshPoint(
		meshPoint,
		numMeshPoints
	);

	switch(getNumDimensions()){
	case 1:
	{
		Vector3d firstCellMeshPoint = getFirstCellCoordinates(
			{mp[0], 0, 0}
		);

		return {firstCellMeshPoint.x};
	}
	case 2:
	{
		Vector3d firstCellMeshPoint = getFirstCellCoordinates(
			{mp[0], mp[1], 0}
		);

		return {firstCellMeshPoint.x, firstCellMeshPoint.y};
	}
	case 3:
	{
		Vector3d firstCellMeshPoint = getFirstCellCoordinates(
			mp
		);

		return {
			firstCellMeshPoint.x,
			firstCellMeshPoint.y,
			firstCellMeshPoint.z
		};
	}
	default:
		TBTKExit(
			"WignerSeitzCell::getMinorMeshPoint()",
			"This should never happed.",
			"Notify the developer about this bug."
		);
	}
}

};	//End of namespace TBTK
