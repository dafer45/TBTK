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

/** @file ModelFactory.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/ModelFactory.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

Model* ModelFactory::createSquareLattice(
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t
){
	Model *model = new Model();

	TBTKAssert(
		size.size() == periodic.size(),
		"ModelFactory::createSquareLattice()",
		"Argument 'size' and argument 'periodic' have different dimensions.",
		""
	);

	switch(size.size()){
	case 1:
		createSquareLattice1D(model, size, periodic, t);
		break;
	case 2:
		createSquareLattice2D(model, size, periodic, t);
		break;
	case 3:
		createSquareLattice3D(model, size, periodic, t);
		break;
	default:
		TBTKExit(
			"ModelFactory::createSquareLattice()",
			"Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.",
			""
		);
	}

	return model;
}

Model* ModelFactory::createHexagonalLattice(
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t
){
	Model *model = new Model();

	TBTKAssert(
		size.size() == periodic.size(),
		"ModelFactory::createHexagonalLattice()",
		"Argument 'size' and argument 'periodic' have different dimensions.",
		""
	);

	TBTKAssert(
		size.size() == 2,
		"ModelFactory::createHexagonalLattice():",
		"Only 2 dimensions supported, but " << size.size() << " dimensions requested.",
		""
	);

	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	bool periodicX = *periodic.begin();
	bool periodicY = *(periodic.begin() + 1);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int s = 0; s < 2; s++){
				*model << HoppingAmplitude(t, {x, y, 1, s},	{x, y, 0, s}) + HC;
				*model << HoppingAmplitude(t, {x, y, 2, s},	{x, y, 1, s}) + HC;
				*model << HoppingAmplitude(t, {x, y, 3, s},	{x, y, 2, s}) + HC;
				if(periodicX || x+1 < sizeX){
					*model << HoppingAmplitude(t,	{(x+1)%sizeX, y, 0, s},	{x, y, 3, s}) + HC;
				}
				if(periodicY || y+1 < sizeY){
					*model << HoppingAmplitude(t,	{x, (y+1)%sizeY, 0, s},	{x, y, 1, s}) + HC;
					*model << HoppingAmplitude(t,	{x, (y+1)%sizeY, 3, s},	{x, y, 2, s}) + HC;
				}
			}
		}
	}

	return model;
}

Model* ModelFactory::createModel(
	const StateSet &stateSet,
	const AbstractOperator &o
){
	Model *model = new Model();

	const vector<AbstractState*> states = stateSet.getStates();
	for(unsigned int from = 0; from < states.size(); from++){
		AbstractState *ket = states.at(from);
		for(unsigned int to = 0; to < states.size(); to++){
			AbstractState *bra = states.at(to);

			complex<double> amplitude = ket->getMatrixElement(*bra);
			if(amplitude != 0.){
				*model << HoppingAmplitude(
					amplitude,
					Index(
						bra->getContainer(),
						bra->getIndex()
					),
					Index(
						ket->getContainer(),
						ket->getIndex()
					)
				);
			}
		}
	}

	unsigned int numCoordinates = states.at(0)->getCoordinates().size();
	for(unsigned int n = 1; n < states.size(); n++){
		TBTKAssert(
			states.at(n)->getCoordinates().size() == numCoordinates,
			"ModelFactory::createModel()",
			"Incompatible coordinate dimensions. First state has "
				<< numCoordinates
				<< " coordinates, while state " << n << " has "
				<< states.at(n)->getCoordinates().size() << " coordinates.",
			""
		);
	}

	unsigned int numSpecifiers = states.at(0)->getSpecifiers().size();
	for(unsigned int n = 1; n < states.size(); n++){
		TBTKAssert(
			states.at(n)->getSpecifiers().size() == numSpecifiers,
			"ModelFactory::createModel()",
			"Incompatible number of specifiers. First state has "
				<< numSpecifiers
				<< " specifiers, while state " << n << " has "
				<< states.at(n)->getSpecifiers().size() << " specifiers.",
			""
		);
	}

	model->construct();
//	model->createGeometry(numCoordinates, numSpecifiers);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	for(unsigned int n = 0; n < states.size(); n++){
		geometry.setCoordinate(
			Index(
				states.at(n)->getContainer(),
				states.at(n)->getIndex()
			),
			states.at(n)->getCoordinates()/*,
			states.at(n)->getSpecifiers()*/
		);
	}

	//TODO:
	//Add specifiers to a yet missing Specifier class (should be analogous
	//to the Geometry class but for non Geometric data.)

	return model;
}

Model* ModelFactory::createModel(
	const StateSet &stateSet,
	const StateTreeNode &stateTreeNode,
	const AbstractOperator &o
){
	Model *model = new Model();

	const vector<AbstractState*> states = stateSet.getStates();
	for(unsigned int from = 0; from < states.size(); from++){
		AbstractState *ket = states.at(from);
		const vector<const AbstractState*> *bras = stateTreeNode.getOverlappingStates(ket->getCoordinates(), ket->getExtent());

		for(unsigned int to = 0; to < bras->size(); to++){
			const AbstractState *bra = bras->at(to);

			complex<double> amplitude = ket->getMatrixElement(*bra);
			if(amplitude != 0.){
				*model << HoppingAmplitude(
					amplitude,
					Index(
						bra->getContainer(),
						bra->getIndex()
					),
					Index(
						ket->getContainer(),
						ket->getIndex()
					)
				);
			}
		}

		delete bras;
	}

	unsigned int numCoordinates = states.at(0)->getCoordinates().size();
	for(unsigned int n = 1; n < states.size(); n++){
		TBTKAssert(
			states.at(n)->getCoordinates().size() == numCoordinates,
			"ModelFactory::createModel()",
			"Incompatible coordinate dimensions. First state has "
				<< numCoordinates
				<< " coordinates, while state " << n << " has "
				<< states.at(n)->getCoordinates().size() << " coordinates.",
			""
		);
	}

	unsigned int numSpecifiers = states.at(0)->getSpecifiers().size();
	for(unsigned int n = 1; n < states.size(); n++){
		TBTKAssert(
			states.at(n)->getSpecifiers().size() == numSpecifiers,
			"ModelFactory::createModel()",
			"Incompatible number of specifiers. First state has "
				<< numSpecifiers
				<< " specifiers, while state " << n << " has "
				<< states.at(n)->getSpecifiers().size() << " specifiers.",
			""
		);
	}

	model->construct();
//	model->createGeometry(numCoordinates, numSpecifiers);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	for(unsigned int n = 0; n < states.size(); n++){
		geometry.setCoordinate(
			Index(
				states.at(n)->getContainer(),
				states.at(n)->getIndex()
			),
			states.at(n)->getCoordinates()/*,
			states.at(n)->getSpecifiers()*/
		);
	}

	//TODO:
	//Add specifiers to a yet missing Specifier class (should be analogous
	//to the Geometry class but for non Geometric data.)

	return model;
}

Model* ModelFactory::createModel(
	const UnitCell &unitCell,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	const bool *includedCells,
	const AbstractOperator &o
){
	Model *model = new Model();

	TBTKAssert(
		size.size() > 0,
		"ModelFactory::createModel()",
		"The argument 'size' must have at least one component.",
		""
	);
	TBTKAssert(
		unitCell.getLatticeVectors().size() == size.size(),
		"ModelFactory::createModel()",
		"The number of lattice vectors in the unit cell must agree with the number of components in the argument 'size'.",
		""
	);
	TBTKAssert(
		unitCell.getLatticeVectors().at(0).size() >= size.size(),
		"ModelFactory::createModel()",
		"The lattice vectors must at least have the same dimension as the number of components in the argument 'size'",
		""
	);
	TBTKAssert(
		size.size() >= periodic.size(),
		"ModelFactory::createModel()",
		"The arguments 'size' and 'periodic' must have the same number of components.",
		""
	);

	switch(size.size()){
	case 1:
		createModel1D(unitCell, size, periodic, includedCells, o);
		break;
	case 2:
		createModel2D(unitCell, size, periodic, includedCells, o);
		break;
	case 3:
		createModel3D(unitCell, size, periodic, includedCells, o);
		break;
	default:
		TBTKExit(
			"ModelFactory::createModel()",
			"Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.",
			""
		);
	}

	return model;
}

void ModelFactory::addSquareGeometry(
	Model *model,
	std::initializer_list<int> size
){
	switch(size.size()){
	case 1:
		addSquareGeometry1D(model, size);
		break;
	case 2:
		addSquareGeometry2D(model, size);
		break;
	case 3:
		addSquareGeometry3D(model, size);
		break;
	default:
		TBTKExit(
			"ModelFactory::addSquareGeometry()",
			"Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.",
			""
		);
	}
}

void ModelFactory::addHexagonalGeometry(
	Model *model,
	std::initializer_list<int> size
){
	if(size.size() != 2){
		TBTKExit(
			"ModelFactory::addSquareGeometry()",
			"Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.",
			""
		);
	}

//	model->createGeometry(3, 0);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int s = 0; s < 2; s++){
				geometry.setCoordinate({x, y, 0, s},	{3.*x + 0.,	sqrt(3.)*y + 0.,		0.});
				geometry.setCoordinate({x, y, 1, s},	{3.*x + 1/2.,	sqrt(3.)*y + sqrt(3.)/2.,	0.});
				geometry.setCoordinate({x, y, 2, s},	{3.*x + 3/2.,	sqrt(3.)*y + sqrt(3.)/2.,	0.});
				geometry.setCoordinate({x, y, 3, s},	{3.*x + 2.,	sqrt(3.)*y + 0.,		0.});
			}
		}
	}
}

Model* ModelFactory::merge(
	initializer_list<Model*> models
){
	Model *model = new Model();
	for(unsigned int n = 0; n < models.size(); n++){
		Model *m = *(models.begin() + n);
		for(
			HoppingAmplitudeSet::ConstIterator iterator
				= m->getHoppingAmplitudeSet().begin();
			iterator != m->getHoppingAmplitudeSet().end();
			++iterator
		){
			complex<double> amplitude = (*iterator).getAmplitude();
//			Index from = ha->fromIndex;
//			Index to = ha->fromIndex;
			Index from = (*iterator).getFromIndex();
			Index to = (*iterator).getToIndex();

			vector<Subindex> newFrom({(Subindex)n});
			for(unsigned int c = 0; c < from.getSize(); c++)
				newFrom.push_back(from.at(c));

			vector<Subindex> newTo({(Subindex)n});
			for(unsigned int c = 0; c < to.getSize(); c++)
				newTo.push_back(to.at(c));

			*model << HoppingAmplitude(amplitude, newTo, newFrom);
		}
	}

	model->construct();

	bool geometryExists = true;
	for(unsigned int n = 0; n < models.size(); n++){
		Model *m = *(models.begin() + n);
/*		if(m->getGeometry() == NULL){
			geometryExists = false;
			Streams::out << "Warning in ModelFactory::merge: Geometric data connot be merged because model " << n << " lacks geometric data.\n";
			break;
		}*/
		try{
			for(int n = 0; n < m->getBasisSize(); n++){
				m->getGeometry().getCoordinate(
					m->getHoppingAmplitudeSet(
					).getPhysicalIndex(n)
				);
			}
		}
		catch(ElementNotFoundException &e){
			geometryExists = false;
			Streams::out << "Warning in ModelFactory::merge:"
			<< " Geometric data connot be merged because model "
			<< n << " has incomplete geometric data.\n";
			break;
		}

		if(m->getGeometry().getDimensions() != 3){
			geometryExists = false;
			Streams::out << "Warning in ModelFactory::merge:"
			<< " Geometric data connot be merged because model "
			<< n << " has geometric of dimension "
			<< m->getGeometry().getDimensions() << ".\n";
			break;
		}

/*		if(m->getGeometry()->getNumSpecifiers() != 0){
			Streams::out << "Warning in ModelFactory::merge: Specifiers ignored in model " << n << ".\n";
		}*/
	}

	if(geometryExists){
//		model->createGeometry(3, 0);
//		Geometry *geometry = model->getGeometry();
		Geometry &geometry = model->getGeometry();

		for(unsigned int n = 0; n < models.size(); n++){
			Model *m = *(models.begin() + n);
//			Geometry *g = m->getGeometry();
			Geometry &g = m->getGeometry();
			for(
				HoppingAmplitudeSet::ConstIterator iterator
					= m->getHoppingAmplitudeSet().begin();
				iterator != m->getHoppingAmplitudeSet().end();
				++iterator
			){
//				Index from = ha->fromIndex;
				Index from = (*iterator).getFromIndex();

				vector<Subindex> newFrom({(Subindex)n});
				for(unsigned int c = 0; c < from.getSize(); c++)
					newFrom.push_back(from.at(c));

//				int basisIndex = m->getBasisIndex(ha->fromIndex);
				int basisIndex = m->getBasisIndex((*iterator).getFromIndex());
//				const double *coordinates = g->getCoordinates(basisIndex);
				const vector<double>& coordinates
					= g.getCoordinate(
						model->getHoppingAmplitudeSet(
						).getPhysicalIndex(basisIndex)
					);

				geometry.setCoordinate(
					newFrom,
					{
						coordinates[0],
						coordinates[1],
						coordinates[2]
					}
				);
			}
		}
	}

	return model;
}

void ModelFactory::createSquareLattice1D(
	Model *model,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t
){
	int sizeX = *size.begin();
	bool periodicX = *periodic.begin();
	for(int x = 0; x < sizeX; x++){
		for(int s = 0; s < 2; s++){
			if(periodicX || x+1 < sizeX)
				*model << HoppingAmplitude(t,	{(x+1)%sizeX, s},	{x, s}) + HC;
		}
	}
}

void ModelFactory::createSquareLattice2D(
	Model *model,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t
){
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	bool periodicX = *periodic.begin();
	bool periodicY = *(periodic.begin() + 1);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int s = 0; s < 2; s++){
				if(periodicX || x+1 < sizeX)
					*model << HoppingAmplitude(t,	{(x+1)%sizeX, y, s},	{x, y, s}) + HC;
				if(periodicY || y+1 < sizeY)
					*model << HoppingAmplitude(t,	{x, (y+1)%sizeY, s},	{x, y, s}) + HC;
			}
		}
	}
}

void ModelFactory::createSquareLattice3D(
	Model *model,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t
){
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	int sizeZ = *(size.begin() + 2);
	bool periodicX = *periodic.begin();
	bool periodicY = *(periodic.begin() + 1);
	bool periodicZ = *(periodic.begin() + 2);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int z = 0; z < sizeZ; z++){
				for(int s = 0; s < 2; s++){
					if(periodicX || x+1 < sizeX)
						*model << HoppingAmplitude(t,	{(x+1)%sizeX, y, z, s},	{x, y, z, s}) + HC;
					if(periodicY || y+1 < sizeY)
						*model << HoppingAmplitude(t,	{x, (y+1)%sizeY, z, s},	{x, y, z, s}) + HC;
					if(periodicZ || z+1 < sizeZ)
						*model << HoppingAmplitude(t,	{x, y, (z+1)&sizeZ, s},	{x, y, z, s}) + HC;
				}
			}
		}
	}
}

Model* ModelFactory::createModel1D(
	const UnitCell &unitCell,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	const bool *includedCells,
	const AbstractOperator &o
){
	TBTKNotYetImplemented("ModelFactory::createModel1D()");
}

Model* ModelFactory::createModel2D(
	const UnitCell &unitCell,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	const bool *includedCells,
	const AbstractOperator &o
){
	TBTKNotYetImplemented("ModelFactory::createModel2D()");
}

Model* ModelFactory::createModel3D(
	const UnitCell &unitCell,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	const bool *includedCells,
	const AbstractOperator &o
){
	TBTKNotYetImplemented("ModelFactory::createModel3D()");
}

void ModelFactory::addSquareGeometry1D(
	Model *model,
	initializer_list<int> size
){
//	model->createGeometry(3, 0);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	int sizeX = *size.begin();
	for(int x = 0; x < sizeX; x++)
		for(int s = 0; s < 2; s++)
			geometry.setCoordinate({x, s},	{1.*x, 0., 0.});
}

void ModelFactory::addSquareGeometry2D(
	Model *model,
	initializer_list<int> size
){
//	model->createGeometry(3, 0);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int s = 0; s < 2; s++){
				geometry.setCoordinate(
					{x, y, s},
					{1.*x, 1.*y, 0.}
				);
			}
		}
	}
}

void ModelFactory::addSquareGeometry3D(
	Model *model,
	initializer_list<int> size
){
//	model->createGeometry(3, 0);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	int sizeZ = *(size.begin() + 2);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int z = 0; z < sizeZ; z++){
				for(int s = 0; s < 2; s++){
					geometry.setCoordinate(
						{x, y, z, s},
						{1.*x, 1.*y, 1.*z}
					);
				}
			}
		}
	}
}

};	//End of namespace TBTK
