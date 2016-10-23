/** @file ModelFactory.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ModelFactory.h"
#include "../include/AmplitudeSet.h"
#include "../include/Streams.h"
#include "../include/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Util{

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
				model->addHAAndHC(HoppingAmplitude(t, {x, y, 1, s},	{x, y, 0, s}));
				model->addHAAndHC(HoppingAmplitude(t, {x, y, 2, s},	{x, y, 1, s}));
				model->addHAAndHC(HoppingAmplitude(t, {x, y, 3, s},	{x, y, 2, s}));
				if(periodicX || x+1 < sizeX)
					model->addHAAndHC(HoppingAmplitude(t,	{(x+1)%sizeX, y, 0, s},	{x, y, 3, s}));
				if(periodicY || y+1 < sizeY){
					model->addHAAndHC(HoppingAmplitude(t,	{x, (y+1)%sizeY, 0, s},	{x, y, 1, s}));
					model->addHAAndHC(HoppingAmplitude(t,	{x, (y+1)%sizeY, 3, s},	{x, y, 2, s}));
				}
			}
		}
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

	model->createGeometry(3, 0);
	Geometry *geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	for(int x = 0; x < sizeX; x++){
		for(int y = 0; y < sizeY; y++){
			for(int s = 0; s < 2; s++){
				geometry->setCoordinates({x, y, 0, s},	{3.*x + 0.,	sqrt(3.)*y + 0.,		0.});
				geometry->setCoordinates({x, y, 1, s},	{3.*x + 1/2.,	sqrt(3.)*y + sqrt(3.)/2.,	0.});
				geometry->setCoordinates({x, y, 2, s},	{3.*x + 3/2.,	sqrt(3.)*y + sqrt(3.)/2.,	0.});
				geometry->setCoordinates({x, y, 3, s},	{3.*x + 2.,	sqrt(3.)*y + 0.,		0.});
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
		AmplitudeSet::Iterator it = m->getAmplitudeSet()->getIterator();
		HoppingAmplitude *ha;
		while((ha = it.getHA())){
			complex<double> amplitude = ha->getAmplitude();
			Index from = ha->fromIndex;
			Index to = ha->fromIndex;

			vector<int> newFrom({(int)n});
			for(unsigned int c = 0; c < from.size(); c++)
				newFrom.push_back(from.at(c));

			vector<int> newTo({(int)n});
			for(unsigned int c = 0; c < to.size(); c++)
				newTo.push_back(to.at(c));

			model->addHA(HoppingAmplitude(amplitude, newTo, newFrom));

			it.searchNextHA();
		}
	}

	model->construct();

	bool geometryExists = true;
	for(unsigned int n = 0; n < models.size(); n++){
		Model *m = *(models.begin() + n);
		if(m->getGeometry() == NULL){
			geometryExists = false;
			Util::Streams::out << "Warning in ModelFactory::merge: Geometric data connot be merged because model " << n << " lacks geometric data.\n";
			break;
		}

		if(m->getGeometry()->getDimensions() != 3){
			geometryExists = false;
			Util::Streams::out << "Warning in ModelFactory::merge: Geometric data connot be merged because model " << n << " has geometric of dimension " << m->getGeometry()->getDimensions() << ".\n";
			break;
		}

		if(m->getGeometry()->getNumSpecifiers() != 0){
			Util::Streams::out << "Warning in ModelFactory::merge: Specifiers ignored in model " << n << ".\n";
		}
	}

	if(geometryExists){
		model->createGeometry(3, 0);
		Geometry *geometry = model->getGeometry();

		for(unsigned int n = 0; n < models.size(); n++){
			Model *m = *(models.begin() + n);
			Geometry *g = m->getGeometry();
			AmplitudeSet::Iterator it = m->getAmplitudeSet()->getIterator();
			HoppingAmplitude *ha;
			while((ha = it.getHA())){
				Index from = ha->fromIndex;

				vector<int> newFrom({(int)n});
				for(unsigned int c = 0; c < from.size(); c++)
					newFrom.push_back(from.at(c));

				int basisIndex = m->getBasisIndex(ha->fromIndex);
				const double *coordinates = g->getCoordinates(basisIndex);

				geometry->setCoordinates(newFrom, {coordinates[0], coordinates[1], coordinates[2]});

				it.searchNextHA();
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
				model->addHAAndHC(HoppingAmplitude(t,	{(x+1)%sizeX, s},	{x, s}));
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
					model->addHAAndHC(HoppingAmplitude(t,	{(x+1)%sizeX, y, s},	{x, y, s}));
				if(periodicY || y+1 < sizeY)
					model->addHAAndHC(HoppingAmplitude(t,	{x, (y+1)%sizeY, s},	{x, y, s}));
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
						model->addHAAndHC(HoppingAmplitude(t,	{(x+1)%sizeX, y, z, s},	{x, y, z, s}));
					if(periodicY || y+1 < sizeY)
						model->addHAAndHC(HoppingAmplitude(t,	{x, (y+1)%sizeY, z, s},	{x, y, z, s}));
					if(periodicZ || z+1 < sizeZ)
						model->addHAAndHC(HoppingAmplitude(t,	{x, y, (z+1)&sizeZ, s},	{x, y, z, s}));
				}
			}
		}
	}
}

void ModelFactory::addSquareGeometry1D(
	Model *model,
	initializer_list<int> size
){
	model->createGeometry(3, 0);
	Geometry *geometry = model->getGeometry();
	int sizeX = *size.begin();
	for(int x = 0; x < sizeX; x++)
		for(int s = 0; s < 2; s++)
			geometry->setCoordinates({x, s},	{1.*x, 0., 0.});
}

void ModelFactory::addSquareGeometry2D(
	Model *model,
	initializer_list<int> size
){
	model->createGeometry(3, 0);
	Geometry *geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	for(int x = 0; x < sizeX; x++)
		for(int y = 0; y < sizeY; y++)
			for(int s = 0; s < 2; s++)
				geometry->setCoordinates({x, y, s},	{1.*x, 1.*y, 0.});
}

void ModelFactory::addSquareGeometry3D(
	Model *model,
	initializer_list<int> size
){
	model->createGeometry(3, 0);
	Geometry *geometry = model->getGeometry();
	int sizeX = *size.begin();
	int sizeY = *(size.begin() + 1);
	int sizeZ = *(size.begin() + 2);
	for(int x = 0; x < sizeX; x++)
		for(int y = 0; y < sizeY; y++)
			for(int z = 0; z < sizeZ; z++)
				for(int s = 0; s < 2; s++)
					geometry->setCoordinates({x, y, z, s},	{1.*x, 1.*y, 1.*z});
}

};	//End of namespace Util
};	//End of namespace TBTK
