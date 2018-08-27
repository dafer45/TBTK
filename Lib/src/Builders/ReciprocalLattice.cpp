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

#include "TBTK/RealLattice.h"
#include "TBTK/ReciprocalLattice.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Vector3d.h"

#include <limits>
#include <typeinfo>

using namespace std;

namespace TBTK{

ReciprocalLattice::ReciprocalLattice(
	UnitCell *unitCell
){
	this->unitCell = unitCell;
	this->realSpaceEnvironment = NULL;
	this->realSpaceEnvironmentStateTree = NULL;
	this->realSpaceReferenceCell = NULL;

	setupReciprocalLatticeVectors(unitCell);
	setupRealSpaceEnvironment(unitCell);
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
/*	Model *model = new Model();

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
			vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
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

			delete bras;

			//Add HoppingAmplitude to Hamiltonian, unless the
			//amplitude is exactly zero.
//			if(amplitude != 0.)
//				model->addHA(HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex()));
				*model << HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex());
		}
	}

	return model;*/
	vector<double> m;
	for(unsigned int n = 0; n < momentum.size(); n++)
		m.push_back(*(momentum.begin() + n));

	return generateModel(m);
}

Model* ReciprocalLattice::generateModel(vector<double> momentum) const{
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
/*			const vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
				referenceKet->getCoordinates(),
				referenceKet->getExtent()
			);*/
			vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
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
						exponent += i*(momentum.at(c))*(bra->getCoordinates().at(c) - referenceBra->getCoordinates().at(c));
					amplitude += bras->at(n)->getMatrixElement(*referenceKet)*exp(exponent);
				}
			}

			delete bras;

			//Add HoppingAmplitude to Hamiltonian, unless the
			//amplitude is exactly zero.
//			if(amplitude != 0.)
//				model->addHA(HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex()));
				*model << HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex());
		}
	}

	return model;
}

Model* ReciprocalLattice::generateModel(
	const vector<vector<double>> &momentums,
	const vector<Index> &blockIndices
) const{
	TBTKAssert(
		momentums.size() == blockIndices.size(),
		"ReciprocalLattice::ReciprocalLattice()",
		"Incompatible dimensions. The number of momentums must be the"
		<< " same as the number of blockIndices. The the number of"
		<< " momentums are " << momentums.size() << ", while the"
		<< " number of blockIndices are " << blockIndices.size()
		<< ".",
		""
	);

	for(unsigned int n = 0; n < momentums.size(); n++){
		TBTKAssert(
			momentums[n].size() == reciprocalLatticeVectors.at(0).size(),
			"ReciprocalLattice::ReciprocalLattice()",
			"Incompatible dimensions. The number of components of"
			<< " momentums must be the same as the number of"
			<< " components of the lattice vectors in the UnitCell."
			<< " The the number of components of momentums[" << n
			<< "] are " << momentums[n].size() << ", while the"
			<< " number of components for the latticeVectors are "
			<< reciprocalLatticeVectors.at(0).size() << ".",
			""
		);
		TBTKAssert(
			blockIndices[n].getSize() == reciprocalLatticeVectors.at(0).size(),
			"ReciprocalLattice::ReciprocalLattice()",
			"Incompatible dimensions. The number of components of"
			<< " blockIndices must be the same as the number of"
			<< " components of the lattice vectors in the UnitCell."
			<< " The the number of components of blockIndices["
			<< n << "] are " << blockIndices[n].getSize()
			<< ", while the number of components for the"
			<< " latticeVectors are "
			<< reciprocalLatticeVectors.at(0).size()
			<< ".",
			""
		);
	}

	Model *model = new Model();

	for(unsigned int from = 0; from < realSpaceReferenceCell->getStates().size(); from++){
		//Get reference ket.
		const AbstractState *referenceKet = realSpaceReferenceCell->getStates().at(from);

		for(unsigned int to = 0; to < realSpaceReferenceCell->getStates().size(); to++){
			//Get reference bra and its Index.
			const AbstractState *referenceBra = realSpaceReferenceCell->getStates().at(to);
			Index referenceBraIndex(referenceBra->getIndex());

			//Get all bras that have a possible overlap with the
			//reference ket.
/*			const vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
				referenceKet->getCoordinates(),
				referenceKet->getExtent()
			);*/
			vector<const AbstractState*> *bras = realSpaceEnvironmentStateTree->getOverlappingStates(
				referenceKet->getCoordinates(),
				referenceKet->getExtent()
			);

			//Calculate momentum space amplitudes
			vector<complex<double>> amplitudes;
			for(unsigned int n = 0; n < momentums.size(); n++)
				amplitudes.push_back(0.);

			for(unsigned int n = 0; n < bras->size(); n++){
				//Loop over all states that have a possible
				//finite overlap with the reference ket.
				const AbstractState *bra = bras->at(n);
				if(bra->getIndex().equals(referenceBraIndex)){
					//Only states with the same Index as
					//the reference ket contributes to the
					//amplitude.

					//Get matrix element.
					complex<double> matrixElement
						= bras->at(
							n
						)->getMatrixElement(
							*referenceKet
						);

					//Get coordinates.
					const vector<double> &braCoordinates
						= bra->getCoordinates();
					const vector<double> &referenceCoordinates
						= referenceBra->getCoordinates();
					vector<double> coordinatesDifference;
					for(
						unsigned int c = 0;
						c < braCoordinates.size();
						c++
					){
						coordinatesDifference.push_back(
							braCoordinates[c]
							- referenceCoordinates[c]
						);
					}

					static const complex<double> i(0., 1.);
#ifdef TBTK_USE_OPEN_MP
					#pragma omp parallel for
#endif
					for(
						unsigned int m = 0;
						m < momentums.size();
						m++
					){
						const vector<double> &momentum
							= momentums[m];

						double exponent = 0.;
						for(
							unsigned int c = 0;
							c < momentums[m].size();
							c++
						){
							exponent += momentum[c]*coordinatesDifference[c];
						}

						amplitudes[m]
							+= matrixElement*exp(
								i*exponent
							);
					}
				}
			}

			delete bras;

			//Add HoppingAmplitude to Hamiltonian, unless the
			//amplitude is exactly zero.
			for(unsigned int n = 0; n < momentums.size(); n++){
//				if(amplitude != 0.)
//					model->addHA(HoppingAmplitude(amplitude, referenceBraIndex, referenceKet->getIndex()));
					*model << HoppingAmplitude(
						amplitudes[n],
						Index(
							blockIndices[n],
							referenceBraIndex
						),
						Index(
							blockIndices[n],
							referenceKet->getIndex()
						)
					);
			}
		}
	}

	return model;
}

void ReciprocalLattice::setupReciprocalLatticeVectors(const UnitCell *unitCell){
	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();

	TBTKAssert(
		latticeVectors.at(0).size() == latticeVectors.size(),
		"ReciprocalLattice::ReciprocalLattice()",
		"Lattice vector dimension not supported.",
		"The number of lattice vectors must agree with the number of"
		<< " components of each individual lattice vector. The"
		<< " supplied UnitCell has " << latticeVectors.size()
		<< " lattice vectors with " << latticeVectors.at(0).size()
		<< " components each."
	);

	//Ensure that the lattice vectors are represented by three-dimensional
	//vectors during the calculation. Will be restored to original
	//dimensionallity at the end of this function.
	vector<vector<double>> paddedLatticeVectors;
	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		paddedLatticeVectors.push_back(vector<double>());
		unsigned int c = 0;
		for(; c < latticeVectors.at(0).size(); c++)
			paddedLatticeVectors.at(n).push_back(latticeVectors.at(n).at(c));
		for(; c < 3; c++)
			paddedLatticeVectors.at(n).push_back(0);
	}

	//Real space lattice vectors on Vector3d format. If the number of
	//lattice vectors is smaller than 3, v[1] and v[2] are created to
	//simplyfy the math, making the calculation the same as for the
	//three-dimensional UnitCells.
	Vector3d v[3];
	for(unsigned int n = 0; n < latticeVectors.size(); n++)
		v[n] = Vector3d(paddedLatticeVectors.at(n));
	switch(latticeVectors.size()){
	case 1:
		v[1] = Vector3d({0, 1, 0});
		v[2] = Vector3d({0, 0, 1});
		break;
	case 2:
		v[2] = Vector3d({0, 0, 1});
		break;
	case 3:
		break;
	default:
		TBTKExit(
			"ReciprocalLattice::setupReciprocalLatticeVectors()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}

	//Calculate reciprocal lattice vectors on Vector3d format.
	Vector3d r[3];
	for(unsigned int n = 0; n < 3; n++)
		r[n] = 2.*M_PI*v[(n+1)%3]*v[(n+2)%3]/(Vector3d::dotProduct(v[n], v[(n+1)%3]*v[(n+2)%3]));

	//Convert reciprocal lattice vectors on Vector3d format back to
	//vector<double>.
	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		reciprocalLatticeVectors.push_back(vector<double>());
		reciprocalLatticeVectors.at(n).push_back(r[n].x);
		if(latticeVectors.size() > 1)
			reciprocalLatticeVectors.at(n).push_back(r[n].y);
		if(latticeVectors.size() > 2)
			reciprocalLatticeVectors.at(n).push_back(r[n].z);
	}
}

void ReciprocalLattice::setupRealSpaceEnvironment(const UnitCell *unitCell){
	const vector<vector<double>> latticeVectors = unitCell->getLatticeVectors();

	//Ensure that the lattice vectors are represented by three-dimensional
	//vectors during the calculation.
	vector<vector<double>> paddedLatticeVectors;
	for(unsigned int n = 0; n < latticeVectors.size(); n++){
		paddedLatticeVectors.push_back(vector<double>());
		unsigned int c = 0;
		for(; c < latticeVectors.at(0).size(); c++)
			paddedLatticeVectors.at(n).push_back(latticeVectors.at(n).at(c));
		for(; c < 3; c++)
			paddedLatticeVectors.at(n).push_back(0);
	}

	//Real space lattice vectors on Vector3d format. If the number of
	//lattice vectors is smaller than 3, v[1] and v[2] are created to
	//simplyfy the math, making the calculation the same as for the
	//three-dimensional UnitCells.
	Vector3d v[3];
	for(unsigned int n = 0; n < latticeVectors.size(); n++)
		v[n] = Vector3d(paddedLatticeVectors.at(n));
	switch(latticeVectors.size()){
	case 1:
		v[1] = Vector3d({0, 1, 0});
		v[2] = Vector3d({0, 0, 1});
		break;
	case 2:
		v[2] = Vector3d({0, 0, 1});
		break;
	case 3:
		break;
	default:
		TBTKExit(
			"ReciprocalLattice::setupReciprocalLatticeVectors()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}

	//Maximum distance from the origin to any point contained in the
	//UnitCell. Occurs at one of the corners of the parallelepiped spanned
	//by the lattice vectors. (1 << latticeVectors.size()) evaluates to the
	//number of corners 2, 4, and 8 for 1, 2, and 3 lattice vectors,
	//respectively.
	double maxDistanceFromOrigin = 0.;
	for(int n = 1; n < (1 << latticeVectors.size()); n++){
		Vector3d w;
		switch(latticeVectors.size()){
		case 1:
			w = (n%2)*v[0];
			break;
		case 2:
			w = (n%2)*v[0] + ((n/2)%2)*v[1];
			break;
		case 3:
			w = (n%2)*v[0] + ((n/2)%2)*v[1] + ((n/4)%2)*v[2];
			break;
		default:
			TBTKExit(
				"ReciprocalLattice::setupReciprocalLatticeVectors()",
				"Unit cell dimension not supported.",
				"Only UnitCells with 1-3 lattice vectors are"
				<< " supported, but the supplied UnitCell has "
				<< latticeVectors.size() << " lattice vectors."
			);
			break;
		}
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
	RealLattice realSpaceEnvironmentLattice(unitCell);
	switch(latticeVectors.size()){
	case 1:
		for(int x = 0; x < 2*realSpaceLatticeHalfSize[0]+1; x++)
			realSpaceEnvironmentLattice.addLatticePoint({x});
		break;
	case 2:
		for(int x = 0; x < 2*realSpaceLatticeHalfSize[0]+1; x++)
			for(int y = 0; y < 2*realSpaceLatticeHalfSize[1]+1; y++)
				realSpaceEnvironmentLattice.addLatticePoint({x, y});
		break;
	case 3:
		for(int x = 0; x < 2*realSpaceLatticeHalfSize[0]+1; x++)
			for(int y = 0; y < 2*realSpaceLatticeHalfSize[1]+1; y++)
				for(int z = 0; z < 2*realSpaceLatticeHalfSize[2]+1; z++)
					realSpaceEnvironmentLattice.addLatticePoint({x, y, z});
		break;
	default:
		TBTKExit(
			"ReciprocalLattice::setupReciprocalLatticeVectors()",
			"Unit cell dimension not supported.",
			"Only UnitCells with 1-3 lattice vectors are"
			<< " supported, but the supplied UnitCell has "
			<< latticeVectors.size() << " lattice vectors."
		);
		break;
	}

	//Create StateSet for the real space environment.
	realSpaceEnvironment = realSpaceEnvironmentLattice.generateStateSet();

	//Create StateTreeNode for quick access of states in
	//realSpaceEnvironment.
	realSpaceEnvironmentStateTree = new StateTreeNode(*realSpaceEnvironment);

	//Extract states contained in the refrence cell. That is, the
	//cell containg all the states that will be used as kets.
	realSpaceReferenceCell = new StateSet(false);
	const vector<AbstractState*> &referenceStates = realSpaceEnvironment->getStates();
	for(unsigned int n = 0; n < referenceStates.size(); n++){
		switch(latticeVectors.size()){
		case 1:
			if(referenceStates.at(n)->getContainer().equals({
					realSpaceLatticeHalfSize[0]
				})
			){
				realSpaceReferenceCell->addState(referenceStates.at(n));
			}
			break;
		case 2:
			if(referenceStates.at(n)->getContainer().equals({
					realSpaceLatticeHalfSize[0],
					realSpaceLatticeHalfSize[1]
				})
			){
				realSpaceReferenceCell->addState(referenceStates.at(n));
			}
			break;
		case 3:
			if(referenceStates.at(n)->getContainer().equals({
					realSpaceLatticeHalfSize[0],
					realSpaceLatticeHalfSize[1],
					realSpaceLatticeHalfSize[2]
				})
			){
				realSpaceReferenceCell->addState(referenceStates.at(n));
			}
			break;
		default:
			TBTKExit(
				"ReciprocalLattice::setupReciprocalLatticeVectors()",
				"Unit cell dimension not supported.",
				"Only UnitCells with 1-3 lattice vectors are"
				<< " supported, but the supplied UnitCell has "
				<< latticeVectors.size() << " lattice vectors."
			);
			break;
		}
	}
}

};	//End of namespace TBTK
