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

#include "WignerSeitzCell.h"
#include "TBTKMacros.h"

using namespace std;

namespace TBTK{

WignerSeitzCell::WignerSeitzCell(initializer_list<initializer_list<double>> basisVectors){
	this->dimensions = basisVectors.size();

	TBTKAssert(
		dimensions == 1
		|| dimensions == 2
		|| dimensions == 3,
		"WignerSeitzCell::WignerSeitzCell()",
		"Basis dimension not supported.",
		"Only 1-3 basis vectors are supported, but "
		<< basisVectors.size() << " basis vectors supplied."
	);

	for(unsigned int n = 0; n < dimensions; n++){
		TBTKAssert(
			(basisVectors.begin() + n)->size() == dimensions,
			"WignerSeitzCell::WignerSeitzCell()",
			"Incompatible dimensions.",
			"The number of basis vectors must agree with the number"
			<< " of components of the basis vectors. The number of"
			<< " basis vectors are '" << dimensions << "',"
			<< " but encountered basis vector with '"
			<< (basisVectors.begin() + n)->size() << "' components."
		);

		vector<double> paddedBasisVector;
		for(unsigned int c = 0; c < dimensions; c++)
			paddedBasisVector.push_back(*((basisVectors.begin() + n)->begin() + c));
		for(unsigned int c = dimensions; c < 3; c++)
			paddedBasisVector.push_back(0.);

		this->basisVectors.push_back(Vector3d(paddedBasisVector));
	}
	for(unsigned int n = dimensions; n < 3; n++){
		vector<double> vector;
		for(unsigned int c = 0; c < 3; c++){
			if(c == n)
				vector.push_back(1.);
			else
				vector.push_back(0.);
		}

		this->basisVectors.push_back(Vector3d(vector));
	}
}

WignerSeitzCell::WignerSeitzCell(const vector<vector<double>> &basisVectors){
	this->dimensions = basisVectors.size();

	TBTKAssert(
		dimensions == 1
		|| dimensions == 2
		|| dimensions == 3,
		"WignerSeitzCell::WignerSeitzCell()",
		"Basis dimension not supported.",
		"Only 1-3 basis vectors are supported, but "
		<< basisVectors.size() << " basis vectors supplied."
	);

	for(unsigned int n = 0; n < dimensions; n++){
		TBTKAssert(
			basisVectors.at(n).size() == dimensions,
			"WignerSeitzCell::WignerSeitzCell()",
			"Incompatible dimensions.",
			"The number of basis vectors must agree with the number"
			<< " of components of the basis vectors. The number of"
			<< " basis vectors are '" << dimensions << "',"
			<< " but encountered basis vector with '"
			<< basisVectors.at(n).size() << "' components."
		);

		vector<double> paddedBasisVector;
		for(unsigned int c = 0; c < dimensions; c++)
			paddedBasisVector.push_back(basisVectors.at(n).at(c));
		for(unsigned int c = dimensions; c < 3; c++)
			paddedBasisVector.push_back(0.);

		this->basisVectors.push_back(Vector3d(paddedBasisVector));
	}
	for(unsigned int n = dimensions; n < 3; n++){
		vector<double> vector;
		for(unsigned int c = 0; c < 3; c++){
			if(c == n)
				vector.push_back(1.);
			else
				vector.push_back(0.);
		}

		this->basisVectors.push_back(Vector3d(vector));
	}
}

WignerSeitzCell::~WignerSeitzCell(){
}

vector<vector<double>> WignerSeitzCell::getMesh(
	initializer_list<unsigned int> numMeshPoints
) const{
	TBTKAssert(
		numMeshPoints.size() == dimensions,
		"ParallelepipedCell::getMesh()",
		"Incompatible diemsnions.",
		"The argument 'numMeshPoints' must have the same number of"
		<< " components as the dimension of the parallelepiped cell."
		<< " The parallelepiped cell has dimension " << dimensions
		<< ", while numMeshPoints have " << numMeshPoints.size()
		<< " components."
	);

	vector<vector<double>> mesh;

	unsigned int nmp[3];
	nmp[0] = *(numMeshPoints.begin() + 0);
	if(numMeshPoints.size() > 1)
		nmp[1] = *(numMeshPoints.begin() + 1);
	else
		nmp[1] = 1;
	if(numMeshPoints.size() > 2)
		nmp[2] = *(numMeshPoints.begin() + 2);
	else
		nmp[2] = 1;

	vector<Vector3d> additionalCorners;
	switch(dimensions){
	case 1:
		break;
	case 2:
		additionalCorners.push_back(basisVectors.at(0) + basisVectors.at(1));
		additionalCorners.push_back(basisVectors.at(0) - basisVectors.at(1));
		break;
	case 3:
		additionalCorners.push_back(basisVectors.at(0) + basisVectors.at(1));
		additionalCorners.push_back(basisVectors.at(0) - basisVectors.at(1));
		additionalCorners.push_back(basisVectors.at(0) + basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(0) - basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(1) + basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(1) - basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(0) + basisVectors.at(1) + basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(0) + basisVectors.at(1) - basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(0) - basisVectors.at(1) + basisVectors.at(2));
		additionalCorners.push_back(basisVectors.at(0) - basisVectors.at(1) - basisVectors.at(2));
		break;
	default:
		TBTKExit(
			"Parallelepiped::getMesh()",
			"This should never happed.",
			"Notify the developer about this bug."
		);
		break;
	}

	for(unsigned int x = 0; x < nmp[0]; x++){
		Vector3d v0;
		if(nmp[0]%2 == 0)
			v0 = ((int)x - (int)(nmp[0]/2) + 1/2.)*basisVectors.at(0)/(nmp[0]-1);
		else
			v0 = ((int)x - (int)(nmp[0]/2))*basisVectors.at(0)/(nmp[0]-1);

		Vector3d perpToB1 = v0.perpendicular(basisVectors.at(1));
		Vector3d shiftY = (perpToB1*(v0.norm())/Vector3d::dotProduct(perpToB1, v0.unit())).perpendicular(basisVectors.at(0));
		if(nmp[0]%2 == 1 && x == nmp[0]/2)
			shiftY = Vector3d({0.,	0.,	0.});

		for(unsigned int y = 0; y < nmp[1]; y++){
			Vector3d perpToB0 = basisVectors.at(1).perpendicular(basisVectors.at(0));
			Vector3d stepY = (perpToB0/Vector3d::dotProduct(basisVectors.at(1).unit(), perpToB0.unit()))/(nmp[1]-1);

			Vector3d v1;
			if(nmp[1]%2 == 0)
				v1 = ((int)y - (int)(nmp[1]/2) + 1/2.)*stepY + shiftY;
			else
				v1 = ((int)y - (int)(nmp[1]/2))*stepY + shiftY;

			Vector3d perpToB2 = (v0+v1).perpendicular(basisVectors.at(2));
			Vector3d shiftZ = (perpToB2*((v0+v1).norm())/Vector3d::dotProduct(perpToB2, (v0+v1).unit())).parallel(basisVectors.at(0)*basisVectors.at(1));

			for(unsigned int z = 0; z < nmp[2]; z++){
				Vector3d perpToB01 = basisVectors.at(2).parallel(basisVectors.at(0)*basisVectors.at(1));
				Vector3d stepZ = (perpToB01/Vector3d::dotProduct(basisVectors.at(2).unit(), perpToB01.unit()))/nmp[2];

				Vector3d v2;
				if(nmp[2]%2 == 0)
					v2 = ((int)z - (int)(nmp[2]/2) + 1/2.)*stepZ + shiftZ*0;
				else
					v2 = ((int)z - (int)(nmp[2]/2))*stepZ + shiftZ*0;

				bool isContainedByAllPlanes = true;
				for(unsigned int n = 0; n < additionalCorners.size(); n++){
					if(abs(Vector3d::dotProduct(v0 + v1 + v2, additionalCorners.at(n).unit())/additionalCorners.at(n).norm()) > 1/2.)
						isContainedByAllPlanes = false;
				}
				if(!isContainedByAllPlanes)
					continue;

				switch(numMeshPoints.size()){
				case 1:
					mesh.push_back({v0.x});
					break;
				case 2:
					mesh.push_back({
						(v0 + v1).x,
						(v0 + v1).y
					});
					break;
				case 3:
					mesh.push_back({
						(v0 + v1 + v2).x,
						(v0 + v1 + v2).y,
						(v0 + v1 + v2).z
					});
					break;
				default:
					TBTKExit(
						"Parallelepiped::getMesh()",
						"This should never happed.",
						"Notify the developer about this bug."
					);
					break;
				}
			}
		}
	}

	return mesh;
}

};	//End of namespace TBTK
