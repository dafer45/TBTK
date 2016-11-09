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

/** @file ReciprocalLattice.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ReciprocalLattice.h"
#include "../include/TBTKMacros.h"
#include "../include/Vector3d.h"

using namespace std;

namespace TBTK{

ReciprocalLattice::ReciprocalLattice(UnitCell *unitCell, initializer_list<int> size){
	this->unitCell = unitCell;

	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();

	TBTKAssert(
		size.size() == latticeVectors.size(),
		"ReciprocalLattice::ReciprocalLattice()",
		"Incompatible dimensions. The number of lattice vectors in the"
		<< " UnitCell must be the same as the number of components in"
		<< " size. The number of lattice vectors are "
		<< latticeVectors.size() << ", while the number of components"
		<< " of size are " << size.size() << ".",
		""
	);

	for(unsigned int n = 0; n < size.size(); n++)
		this->size.push_back(*(size.begin() + n));

	//Calculate reciprocal lattice vectors
	switch(latticeVectors.size()){
	case 1:
	{
		//1D real space lattice to 1D reciprocal lattice vectors.

		TBTKAssert(
			latticeVectors.at(0).size() == 1
			|| latticeVectors.at(0).size() == 2
			|| latticeVectors.at(0).size() == 3,
			"ReciprocalLattice::ReciprocalLattice()",
			"Lattice vector dimension not supported.",
			"Only one-, two-, and three-dimensional lattice"
			<< " vectors are supported for UnitCells with one"
			<< " lattice vector. The supplied UnitCell has a"
			<< " lattice vector with "
			<< latticeVectors.at(0).size() << " dimensions."
		);

		//Ensure that the lattice vectors are represented by
		//three-dimensional vectors during the calculation. Will be
		//restored to original dimensionallity at the end of this
		//code block.
		vector<double> paddedLatticeVector;
		unsigned int c = 0;
		for(; c < latticeVectors.at(0).size(); c++)
			paddedLatticeVector.push_back(latticeVectors.at(0).at(c));
		for(; c < 3; c++)
			paddedLatticeVector.push_back(latticeVectors.at(0).at(c));

		//Real space lattice vectors on Vector3d format.
		Vector3d v(paddedLatticeVector);

		//Calculate reciprocal lattice vectors on Vector3d format.
		Vector3d r = 2.*M_PI*v/Vector3d::dotProduct(v, v);

		//Convert reciprocal lattice vectors on Vector3d format back to
		//vector<double> with the same number of components as the
		//original lattice vectors.
		reciprocalLatticeVectors.push_back(vector<double>());
		reciprocalLatticeVectors.at(0).push_back(r.x);
		if(latticeVectors.at(0).size() > 1)
			reciprocalLatticeVectors.at(0).push_back(r.y);
		if(latticeVectors.at(0).size() > 2)
			reciprocalLatticeVectors.at(0).push_back(r.z);
	}
	case 2:
	{
		//2D real space lattice to 2D reciprocal lattice vectors.

		TBTKAssert(
			latticeVectors.at(0).size() == 2
			|| latticeVectors.at(0).size() == 3,
			"ReciprocalLattice::ReciprocalLattice()",
			"Lattice vector dimension not supported.",
			"Only two- and three-dimensional lattice vectors are"
			<< " supported for UnitCells with two lattice"
			<< " vectors. The supplied UnitCell has lattice"
			<< " vectors with " << latticeVectors.at(0).size()
			<< " dimensions."
		);

		//Ensure that the lattice vectors are represented by
		//three-dimensional vectors during the calculation. Will be
		//restored to original dimensionallity at the end of this
		//code block.
		vector<vector<double>> paddedLatticeVectors;
		for(unsigned int n = 0; n < 2; n++){
			paddedLatticeVectors.push_back(vector<double>());

			for(unsigned int c = 0; c < latticeVectors.at(n).size(); c++)
				paddedLatticeVectors.at(n).push_back(latticeVectors.at(n).at(c));
			if(latticeVectors.at(n).size() == 2)
				paddedLatticeVectors.at(n).push_back(0.);
		}

		//Real space lattice vectors on Vector3d format. v[2] is
		//created to simplyfy the math by making the calculation
		//similar to the one for three-dimensional UnitCells.
		Vector3d v[3];
		for(unsigned int n = 0; n < 2; n++)
			v[n] = Vector3d(paddedLatticeVectors.at(n));
		v[2] = v[0]*v[1];

		//Calculate reciprocal lattice vectors on Vector3d format.
		Vector3d r[2];
		for(unsigned int n = 0; n < 2; n++)
			r[n] = 2.*M_PI*v[n+1]*v[(n+2)%3]/Vector3d::dotProduct(v[n], v[n+1]*v[(n+2)%3]);

		//Convert reciprocal lattice vectors on Vector3d format back to
		//vector<double> with the same number of components as the
		//original lattice vectors.
		for(unsigned int n = 0; n < 2; n++){
			reciprocalLatticeVectors.push_back(vector<double>());
			reciprocalLatticeVectors.at(n).push_back(r[n].x);
			reciprocalLatticeVectors.at(n).push_back(r[n].y);
			if(latticeVectors.at(0).size() == 3)
				reciprocalLatticeVectors.at(n).push_back(r[n].z);
		}

		break;
	}
	case 3:
	{
		//3D real space lattice to 3D reciprocal lattice vectors.

		TBTKAssert(
			latticeVectors.at(0).size() == 3,
			"ReciprocalLattice::ReciprocalLattice()",
			"Lattice vector dimension not supported.",
			"Only three-dimensional lattice vectors are supported"
			<< " for UnitCells with three lattice vectors. The"
			<< " supplied UnitCell has lattice vectors with "
			<< latticeVectors.at(0).size() << " dimensions."
		);

		//Real space lattice vectors on Vector3d format. v[2] is
		//created to simplyfy the math by making the calculation
		//similar to the one for three-dimensional UnitCells.
		Vector3d v[3];
		for(unsigned int n = 0; n < 3; n++)
			v[n] = Vector3d(latticeVectors.at(n));

		//Calculate reciprocal lattice vectors on Vector3d format.
		Vector3d r[3];
		for(unsigned int n = 0; n < 3; n++)
			r[n] = 2.*M_PI*v[(n+1)%3]*v[(n+2)%3]/(Vector3d::dotProduct(v[n], v[(n+1)%3]*v[(n+2)%3]));

		//Convert reciprocal lattice vectors on Vector3d format back to
		//vector<double>.
		for(unsigned int n = 0; n < 3; n++)
			reciprocalLatticeVectors.push_back(r[n].getStdVector());

		break;
	}
	default:
		TBTKExit(
			"ReciprocalLattice::ReciprocalLattice()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}

	switch(latticeVectors.size()){
	case 1:
		TBTKNotYetImplemented("ReciprocalLattice::ReciprocalLattice()");
		break;
	case 2:
		TBTKNotYetImplemented("ReciprocalLattice::ReciprocalLattice()");
		break;
	case 3:
		TBTKNotYetImplemented("ReciprocalLattice::ReciprocalLattice()");
		break;
	default:
		TBTKExit(
			"ReciprocalLattice::ReciprocalLattice()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}
}

ReciprocalLattice::~ReciprocalLattice(){
}

Model* ReciprocalLattice::generateModel(initializer_list<double> momentum) const{
	TBTKNotYetImplemented("ReciprocalLattice::generateModel()");
	Model *model = new Model();

	return model;
}

};	//End of namespace TBTK
