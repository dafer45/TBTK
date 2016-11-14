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

	for(unsigned int n = 0; n < 3; n++){
		const Vector3d &v0 = this->basisVectors.at(n);
		const Vector3d &v1 = this->basisVectors.at((n+1)%3);
		const Vector3d &v2 = this->basisVectors.at((n+2)%3);

		Vector3d normal = v1*v2;
		normal = normal/Vector3d::dotProduct(normal, v0);

		reciprocalNormals.push_back(normal);
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

	for(unsigned int n = 0; n < 3; n++){
		const Vector3d &v0 = basisVectors.at(n);
		const Vector3d &v1 = basisVectors.at((n+1)%3);
		const Vector3d &v2 = basisVectors.at((n+2)%3);

		Vector3d normal = v1*v2;
		normal = normal/Vector3d::dotProduct(normal, v0);

		reciprocalNormals.push_back(normal);
	}
}

WignerSeitzCell::~WignerSeitzCell(){
}

Index WignerSeitzCell::getCellIndex(initializer_list<double> coordinates) const{
	TBTKAssert(
		coordinates.size() == dimensions,
		"WignerSeitzCell::getCellIndex()",
		"Incompatible dimensions.",
		"The number of coordinate components must agree with the"
		<< " dimension of the Wigner-Seitz cell. The Weigner-Seitz"
		<< " cell has " << dimensions << " dimensions, but coordinate"
		<< " with " << coordinates.size() << " components supplied."
	);

	Vector3d coordinateVector;
	switch(coordinates.size()){
	case 1:
		coordinateVector = Vector3d({
			*(coordinates.begin() + 0),
			0.,
			0.
		});
		break;
	case 2:
		coordinateVector = Vector3d({
			*(coordinates.begin() + 0),
			*(coordinates.begin() + 1),
			0.
		});
		break;
	case 3:
		coordinateVector = Vector3d({
			*(coordinates.begin() + 0),
			*(coordinates.begin() + 1),
			*(coordinates.begin() + 2)
		});
		break;
	default:
		TBTKExit(
			"WignerSeitzCell::WignerSeitzCell()",
			"This should never happen.",
			"Notify the developer about this bug."
		);
		break;
	}

	Index cellIndex({});
	for(unsigned int n = 0; n < dimensions; n++){
		double v = Vector3d::dotProduct(
			coordinateVector,
			reciprocalNormals.at(n)
		);
		if(v > 0)
			cellIndex.push_back((int)(v + 1/2.));
		else
			cellIndex.push_back((int)(v - 1/2.));
	}

	return cellIndex;
}

};	//End of namespace TBTK
