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
#include "../include/Lattice.h"

#include <limits>
#include <typeinfo>

using namespace std;

namespace TBTK{

ReciprocalLattice::ReciprocalLattice(
	UnitCell *unitCell/*,
	initializer_list<int> size*/
){
	this->unitCell = unitCell;
	this->realSpaceEnvironment = NULL;
	this->realSpaceEnvironmentStateTree = NULL;
	this->realSpaceReferenceCell = NULL;

	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();

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

		break;
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
	{
		//Ensure that the lattice vectors are represented by
		//three-dimensional vectors during the calculation.
		vector<double> paddedLatticeVector;
		unsigned int c = 0;
		for(; c < latticeVectors.at(0).size(); c++)
			paddedLatticeVector.push_back(latticeVectors.at(0).at(c));
		for(; c < 3; c++)
			paddedLatticeVector.push_back(latticeVectors.at(0).at(c));

		//Real space lattice vectors on Vector3d format.
		Vector3d v(paddedLatticeVector);

		//Maximum distance from the origin to any point contained in
		//the UnitCell.
		double maxDistanceFromOrigin = v.norm();

		//Find maximum extent.
		const vector<AbstractState*> &states = unitCell->getStates();
		double maxExtent = 0.;
		for(unsigned int n = 0; n < states.size(); n++){
			TBTKAssert(
				states.at(n)->getExtent() != numeric_limits<double>::infinity()
				&& states.at(n)->getExtent() != numeric_limits<double>::max(),
				"ReciprocalLattice::ReciprocalLattice()",
				"Encountered state with infinite extent, but"
				<< " only states with finite extent"
				<< " supported.",
				"Use AbstractState::setExtent() to set the"
				<< " extent of each state in the UnitCell."
			);

			if(maxExtent < states.at(n)->getExtent())
				maxExtent = states.at(n)->getExtent();
		}

		//Radius of a sphere centered at the origin, which is large
		//enough that all states that have an extent that overlapps
		//with the states in the unit cell are centered inside the
		//sphere.
		double enclosingRadius = (2.*maxExtent + maxDistanceFromOrigin)*ROUNDOFF_MARGIN_MULTIPLIER;

		//Calculate number of UnitCells needed to cover the enclosing
		//sphere and Index shift required to ensure that all UnitCells
		//have positive indices.
		int realSpaceLatticeHalfSize = (int)(enclosingRadius/v.norm()) + 1;

		//Create a lattice for the real space environment.
		Lattice realSpaceEnvironmentLattice(unitCell);
		for(int n = 0; n < 2*realSpaceLatticeHalfSize+1; n++)
			realSpaceEnvironmentLattice.addLatticePoint({n});

		//Create StateSet for the real space environment.
		realSpaceEnvironment = realSpaceEnvironmentLattice.generateStateSet();

		//Create StateTreeNode for quick access of states in
		//realSpaceEnvironment.
		realSpaceEnvironmentStateTree = new StateTreeNode(*realSpaceEnvironment);

		//Extract states contained in the refrence cell. That is, the
		//cell containg all the states that will be used as kets.
		realSpaceReferenceCell = new StateSet(false);
		const vector<AbstractState*> &referenceStates = realSpaceEnvironment->getStates();
		for(unsigned int n = 0; n < referenceStates.size(); n++)
			if(referenceStates.at(n)->getContainer().equals({realSpaceLatticeHalfSize}))
				realSpaceReferenceCell->addState(referenceStates.at(n));

		break;
	}
	case 2:
		TBTKNotYetImplemented("ReciprocalLattice::ReciprocalLattice()");
		break;
	case 3:
	{
		//Real space lattice vectors on Vector3d format.
		Vector3d v[3];
		for(int n = 0; n < 3; n++)
			v[n] = Vector3d(latticeVectors.at(n));

		//Maximum distance from the origin to any point contained in
		//the UnitCell. Occurs at one of the corners of the
		//parallelepiped spanned by the lattice vectors.
		double maxDistanceFromOrigin = 0.;
		for(int n = 1; n < 8; n++){
			Vector3d w = (n%2)*v[0] + ((n/2)%2)*v[1] + ((n/4)%2)*v[2];
			if(w.norm() > maxDistanceFromOrigin)
				maxDistanceFromOrigin = w.norm();
		}

		//Find maximum extent.
		const vector<AbstractState*> &states = unitCell->getStates();
		double maxExtent = 0.;
		for(unsigned int n = 0; n < states.size(); n++){
			TBTKAssert(
				states.at(n)->getExtent() != numeric_limits<double>::infinity()
				&& states.at(n)->getExtent() != numeric_limits<double>::max(),
				"ReciprocalLattice::ReciprocalLattice()",
				"Encountered state with infinite extent, but"
				<< " only states with finite extent"
				<< " supported.",
				"Use AbstractState::setExtent() to set the"
				<< " extent of each state in the UnitCell."
			);

			if(maxExtent < states.at(n)->getExtent())
				maxExtent = states.at(n)->getExtent();
		}

		//Radius of a sphere centered at the origin, which is large
		//enough that all states that have an extent that overlapps
		//with the states in the unit cell are centered inside the
		//sphere.
		double enclosingRadius = (2.*maxExtent + maxDistanceFromOrigin)*ROUNDOFF_MARGIN_MULTIPLIER;

		//Calculate number of UnitCells needed to cover the enclosing
		//sphere and Index shift required to ensure that all UnitCells
		//have positive indices.
		int realSpaceLatticeHalfSize[3];
		for(int n = 0; n < 3; n++){
			Vector3d normal = v[(n+1)%3]*v[(n+2)%3];
			normal = normal/normal.norm();
			double perpendicularLength = Vector3d::dotProduct(v[n], normal);
			realSpaceLatticeHalfSize[n] = (int)(enclosingRadius/perpendicularLength) + 1;
		}

		//Create a lattice for the real space environment.
		Lattice realSpaceEnvironmentLattice(unitCell);
		for(int x = 0; x < 2*realSpaceLatticeHalfSize[0]+1; x++)
			for(int y = 0; y < 2*realSpaceLatticeHalfSize[1]+1; y++)
				for(int z = 0; z < 2*realSpaceLatticeHalfSize[2]+1; z++)
					realSpaceEnvironmentLattice.addLatticePoint({x, y, z});

		//Create StateSet for the real space environment.
		realSpaceEnvironment = realSpaceEnvironmentLattice.generateStateSet();

		//Create StateTreeNode for quick access of states in
		//realSpaceEnvironment.
		realSpaceEnvironmentStateTree = new StateTreeNode(*realSpaceEnvironment);

		//Extract states contained in the refrence cell. That is, the
		//cell containg all the states that will be used as kets.
		realSpaceReferenceCell = new StateSet(false);
		const vector<AbstractState*> &referenceStates = realSpaceEnvironment->getStates();
		for(unsigned int n = 0; n < referenceStates.size(); n++)
			if(referenceStates.at(n)->getContainer().equals({
				realSpaceLatticeHalfSize[0],
				realSpaceLatticeHalfSize[1],
				realSpaceLatticeHalfSize[2]})
			){
				realSpaceReferenceCell->addState(referenceStates.at(n));
			}

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
}

ReciprocalLattice::~ReciprocalLattice(){
	if(realSpaceEnvironment != NULL)
		delete realSpaceEnvironment;
	if(realSpaceEnvironmentStateTree != NULL)
		delete realSpaceEnvironmentStateTree;
	if(realSpaceReferenceCell != NULL)
		delete realSpaceReferenceCell;
}

Model* ReciprocalLattice::generateModel(initializer_list<double> momentum) const{
	Model *model = new Model();

	TBTKAssert(
		momentum.size() == reciprocalLatticeVectors.at(0).size(),
		"ReciprocalLattice::ReciprocalLattice()",
		"Incompatible dimensions. The number of components of momentum"
		<< " must be the same as the number of components of the"
		<< " lattice vectors in the UnitCell. The the number of"
		<< " components of the momentum are " << momentum.size() << ","
		<< " while the number of components for the latticeVectors"
		<< " are " << reciprocalLatticeVectors.at(0).size() << ".",
		""
	);

	for(unsigned int from = 0; from < realSpaceReferenceCell->getStates().size(); from++){
		//Get reference ket.
		const AbstractState *referenceKet = realSpaceReferenceCell->getStates().at(from);

		for(unsigned int to = 0; to < realSpaceReferenceCell->getStates().size(); to++){
			//Get reference bra and its Index.
			const AbstractState *referenceBra = realSpaceReferenceCell->getStates().at(to);
			Index referenceBraIndex(referenceBra->getIndex());

			//Get all bras that have a possible overlap with the
			//reference ket.
			const vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
				referenceKet->getCoordinates(),
				referenceKet->getExtent()
			);

			//Calculate momentum space amplitude
			complex<double> amplitude = 0.;
			for(unsigned int n = 0; n < bras->size(); n++){
				//Loop over all states that have a possible
				//finite overlap with the reference ket.
				const AbstractState *bra = bras->at(n);
				if(bra->getIndex().equals(referenceBraIndex)){
					//Only states with the same Index as
					//the reference ket contributes to the
					//amplitude.

					static const complex<double> i(0., 1.);
					complex<double> exponent = 0.;
					for(unsigned int c = 0; c < momentum.size(); c++)
						exponent += i*(*(momentum.begin() + c))*(bra->getCoordinates().at(c) - referenceBra->getCoordinates().at(c));
					amplitude += bras->at(n)->getMatrixElement(*referenceKet)*exp(exponent);
				}
			}

			//Add HoppingAmplitude to Hamiltonian, unless the
			//amplitude is exactly zero.
//			if(amplitude != 0.)
				model->addHA(HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex()));
		}
	}

	return model;
}

};	//End of namespace TBTK
