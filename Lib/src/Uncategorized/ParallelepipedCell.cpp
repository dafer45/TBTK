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

/** @file ParallelepipedCell.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ParallelepipedCell.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

ParallelepipedCell::ParallelepipedCell(){
}

ParallelepipedCell::ParallelepipedCell(
	const vector<vector<double>> &basisVectors,
	MeshType meshType
) :
	SpacePartition(basisVectors, meshType)
{
	const vector<Vector3d> &bv = getBasisVectors();
	for(unsigned int n = 0; n < 3; n++){
		const Vector3d &v0 = bv.at(n);
		const Vector3d &v1 = bv.at((n+1)%3);
		const Vector3d &v2 = bv.at((n+2)%3);

		Vector3d normal = v1*v2;
		normal = normal/Vector3d::dotProduct(normal, v0);

		reciprocalNormals.push_back(normal);
	}
}

ParallelepipedCell::~ParallelepipedCell(){
}

Index ParallelepipedCell::getMajorCellIndex(
	const vector<double> &coordinates
) const{
	TBTKAssert(
		coordinates.size() == getNumDimensions(),
		"ParallelepipedCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " dimension of the parallelepiped cell. The parallelepiped"
		<< " cell has " << getNumDimensions() << " dimensions, but coordinate"
		<< " with " << coordinates.size() << " components supplied."
	);

	Vector3d coordinateVector;
	switch(coordinates.size()){
	case 1:
		coordinateVector = Vector3d({
			coordinates.at(0),
			0.,
			0.
		});
		break;
	case 2:
		coordinateVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			0.
		});
		break;
	case 3:
		coordinateVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			coordinates.at(2)
		});
		break;
	default:
		TBTKExit(
			"ParallelepipedCell::getCellIndex()",
			"Only coordinates with 1-3 components supported.",
			""
		);
		break;
	}

	Index cellIndex;
	cellIndex.reserve(getNumDimensions());	//Optimization
	for(unsigned int n = 0; n < getNumDimensions(); n++){
		double v = Vector3d::dotProduct(
			coordinateVector,
			reciprocalNormals.at(n)
		);
		if(v > 0)
			cellIndex.pushBack((int)(v + 1/2.));
		else
			cellIndex.pushBack((int)(v - 1/2.));
	}

	return cellIndex;
}

Index ParallelepipedCell::getMinorCellIndex(
	const vector<double> &coordinates,
	const vector<unsigned int> &numMeshPoints
) const{
	TBTKAssert(
		coordinates.size() == getNumDimensions(),
		"ParallelepipedCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " dimension of the parallelepiped cell. The parallelepiped"
		<< " cell has " << getNumDimensions() << " dimensions, but coordinate"
		<< " with " << coordinates.size() << " components supplied."
	);
	TBTKAssert(
		coordinates.size() == numMeshPoints.size(),
		"ParallelepipedCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " number of mesh points. 'coordinate' has "
		<< coordinates.size() << " components, while 'numMeshPoints'"
		<< " have " << numMeshPoints.size() << " components."
	);

	Vector3d coordinateVector;
	switch(coordinates.size()){
	case 1:
		coordinateVector = Vector3d({
			coordinates.at(0),
			0.,
			0.
		});
		break;
	case 2:
		coordinateVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			0.
		});
		break;
	case 3:
		coordinateVector = Vector3d({
			coordinates.at(0),
			coordinates.at(1),
			coordinates.at(2)
		});
		break;
	default:
		TBTKExit(
			"ParallelepipedCell::getCellIndex()",
			"Only coordinates with 1-3 components supported.",
			""
		);
		break;
	}

	Index cellIndex;
	switch(getMeshType()){
	case MeshType::Nodal:
		for(unsigned int n = 0; n < getNumDimensions(); n++){
			double v = Vector3d::dotProduct(
				coordinateVector,
				reciprocalNormals.at(n)
			);
			if(v >= 0)
				cellIndex.pushBack((int)(v*(numMeshPoints.at(n)) + 1/2.));
			else
				cellIndex.pushBack((int)(v*(numMeshPoints.at(n)) - 1/2.));
		}
		break;
	case MeshType::Interior:
		for(unsigned int n = 0; n < getNumDimensions(); n++){
			double v = Vector3d::dotProduct(
				coordinateVector,
				reciprocalNormals.at(n)
			);
			if(v >= 0)
				cellIndex.pushBack((int)(v*(numMeshPoints.at(n))));
			else
				cellIndex.pushBack((int)(v*(numMeshPoints.at(n))-1));
		}
		break;
	default:
		TBTKExit(
			"ParallelepipedCell::getCellIndex()",
			"Only SpacePartition::MeshType::Nodal is supported so"
			<< " far.",
			""
		);
	}

	return cellIndex;
}

vector<vector<double>> ParallelepipedCell::getMajorMesh(
	const std::vector<unsigned int> &numMeshPoints
) const{
	TBTKAssert(
		numMeshPoints.size() == getNumDimensions(),
		"ParallelepipedCell::getMesh()",
		"Incompatible diemsnions.",
		"The argument 'numMeshPoints' must have the same number of"
		<< " components as the dimension of the parallelepiped cell."
		<< " The parallelepiped cell has dimension "
		<< getNumDimensions() << ", while numMeshPoints have "
		<< numMeshPoints.size() << " components."
	);

	const vector<Vector3d> &basisVectors = getBasisVectors();
	vector<vector<double>> mesh;

	switch(numMeshPoints.size()){
	case 1:
	{
		const Vector3d &b0 = basisVectors.at(0);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			mesh.push_back(vector<double>());
			mesh.back().push_back(x*b0.x);
		}

		break;
	}
	case 2:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			for(unsigned int y = 0; y < numMeshPoints.at(1); y++){
				mesh.push_back(vector<double>());
				Vector3d meshPoint = x*b0 + y*b1;
				mesh.back().push_back(meshPoint.x);
				mesh.back().push_back(meshPoint.y);
			}
		}

		break;
	}
	case 3:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);
		const Vector3d &b2 = basisVectors.at(2);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			for(unsigned int y = 0; y < numMeshPoints.at(1); y++){
				for(
					unsigned int z = 0;
					z < numMeshPoints.at(2);
					z++
				){
					mesh.push_back(vector<double>());
					Vector3d meshPoint
						= x*b0 + y*b1 + z*b2;
					mesh.back().push_back(meshPoint.x);
					mesh.back().push_back(meshPoint.y);
					mesh.back().push_back(meshPoint.z);
				}
			}
		}

		break;
	}
	default:
		TBTKExit(
			"ParallelepipedCell::getMesh()",
			"This should never happen.",
			"Notify the developer about this bug."
		);
	}

	return mesh;
}

vector<vector<double>> ParallelepipedCell::getMinorMesh(
	const std::vector<unsigned int> &numMeshPoints
) const{
	TBTKAssert(
		numMeshPoints.size() == getNumDimensions(),
		"ParallelepipedCell::getMesh()",
		"Incompatible dimensions.",
		"The argument 'numMeshPoints' must have the same number of"
		<< " components as the dimension of the parallelepiped cell."
		<< " The parallelepiped cell has dimension "
		<< getNumDimensions() << ", while numMeshPoints have "
		<< numMeshPoints.size() << " components."
	);

	const vector<Vector3d> &basisVectors = getBasisVectors();
	vector<vector<double>> mesh;

	switch(numMeshPoints.size()){
	case 1:
	{
		const Vector3d &b0 = basisVectors.at(0);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			mesh.push_back(vector<double>());
			Vector3d meshPoint = x*b0/numMeshPoints.at(0);
			mesh.back().push_back(meshPoint.x);
		}

		break;
	}
	case 2:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			for(unsigned int y = 0; y < numMeshPoints.at(1); y++){
				mesh.push_back(vector<double>());
				Vector3d meshPoint
					= x*b0/numMeshPoints.at(0)
					+ y*b1/numMeshPoints.at(1);
				mesh.back().push_back(meshPoint.x);
				mesh.back().push_back(meshPoint.y);
			}
		}

		break;
	}
	case 3:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);
		const Vector3d &b2 = basisVectors.at(2);

		for(unsigned int x = 0; x < numMeshPoints.at(0); x++){
			for(unsigned int y = 0; y < numMeshPoints.at(1); y++){
				for(
					unsigned int z = 0;
					z < numMeshPoints.at(2);
					z++
				){
					mesh.push_back(vector<double>());
					Vector3d meshPoint
						= x*b0/numMeshPoints.at(0)
						+ y*b1/numMeshPoints.at(1)
						+ z*b2/numMeshPoints.at(2);
					mesh.back().push_back(meshPoint.x);
					mesh.back().push_back(meshPoint.y);
					mesh.back().push_back(meshPoint.z);
				}
			}
		}

		break;
	}
	default:
		TBTKExit(
			"ParallelepipedCell::getMesh()",
			"This should never happen.",
			"Notify the developer about this bug."
		);
	}

	return mesh;
}

vector<double> ParallelepipedCell::getMinorMeshPoint(
	const std::vector<unsigned int> &meshPoint,
	const std::vector<unsigned int> &numMeshPoints
) const{
	TBTKAssert(
		meshPoint.size() == getNumDimensions(),
		"ParallelepipedCell::getMinorMeshPoint()",
		"Incompatible dimensions.",
		"The argument 'meshPoint' must have the same number of"
		<< " components as the dimension of the parallelepiped cell."
		<< " The parallelepiped cell has dimension "
		<< getNumDimensions() << ", while numMeshPoints have "
		<< meshPoint.size() << " components."
	);
	TBTKAssert(
		numMeshPoints.size() == getNumDimensions(),
		"ParallelepipedCell::getMinorMeshPoint()",
		"Incompatible dimensions.",
		"The argument 'numMeshPoints' must have the same number of"
		<< " components as the dimension of the parallelepiped cell."
		<< " The parallelepiped cell has dimension "
		<< getNumDimensions() << ", while numMeshPoints have "
		<< numMeshPoints.size() << " components."
	);
	for(unsigned int n = 0; n < meshPoint.size(); n++){
		TBTKAssert(
			meshPoint[n] < numMeshPoints[n],
			"ParallelepipedCell::getMinorMeshPoint()",
			"Mesh point out of range.",
			"The mesh point 'meshPoint[" << n << "]="
			<< meshPoint[n] << "' must be less than"
			<< " 'numMeshPoints[" << n << "]=" << numMeshPoints[n]
			<< "'."
		);
	}

	const vector<Vector3d> &basisVectors = getBasisVectors();

	switch(numMeshPoints.size()){
	case 1:
	{
		const Vector3d &b0 = basisVectors.at(0);

		return {(meshPoint[0]*b0/numMeshPoints.at(0)).x};
	}
	case 2:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);

		Vector3d mp
			= meshPoint[0]*b0/numMeshPoints.at(0)
			+ meshPoint[1]*b1/numMeshPoints.at(1);

		return {mp.x, mp.y};
	}
	case 3:
	{
		const Vector3d &b0 = basisVectors.at(0);
		const Vector3d &b1 = basisVectors.at(1);
		const Vector3d &b2 = basisVectors.at(2);

		Vector3d mp
			= meshPoint[0]*b0/numMeshPoints.at(0)
			+ meshPoint[1]*b1/numMeshPoints.at(1)
			+ meshPoint[2]*b2/numMeshPoints.at(2);

		return {mp.x, mp.y, mp.z};
	}
	default:
		TBTKExit(
			"ParallelepipedCell::getMesh()",
			"This should never happen.",
			"Notify the developer about this bug."
		);
	}
}

};	//End of namespace TBTK
