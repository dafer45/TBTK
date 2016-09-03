/** @file ModelFactory.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ModelFactory.h"

using namespace std;

namespace TBTK{
namespace Util{

Model* ModelFactory::createSquareLattice(
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t)
{
	Model *model = new Model();

	if(size.size() != periodic.size()){
		cout << "Error in ModelFactory::createSquareLattice: size and periodic have different dimensions.\n";
		exit(1);
	}

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
		cout << "Error in ModelFactory::createSquareLattice: Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.\n";
		exit(1);
	}

	return model;
}

Model* ModelFactory::createHexagonalLattice(
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t)
{
	Model *model = new Model();

	if(size.size() != periodic.size()){
		cout << "Error in ModelFactory::createHexagonalLattice: size and periodic have different dimensions.\n";
		exit(1);
	}

	if(size.size() != 2){
		cout << "Error in ModelFactory::createHexagonalLattice: Only 2 dimensions supported, but " << size.size() << " dimensions requested.\n";
		exit(1);
	}

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
	std::initializer_list<int> size)
{
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
		cout << "Error in ModelFactory::addSquareGeometry: Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.\n";
		exit(1);
	}
}

void ModelFactory::addHexagonalGeometry(
	Model *model,
	std::initializer_list<int> size)
{
	if(size.size() != 2){
		cout << "Error in ModelFactory::addSquareGeometry: Only 1-3 dimensions supported, but " << size.size() << " dimensions requested.\n";
		exit(1);
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

void ModelFactory::createSquareLattice1D(
	Model *model,
	initializer_list<int> size,
	initializer_list<bool> periodic,
	complex<double> t)
{
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
	complex<double> t)
{
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
	complex<double> t)
{
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
	initializer_list<int> size)
{
	model->createGeometry(3, 0);
	Geometry *geometry = model->getGeometry();
	int sizeX = *size.begin();
	for(int x = 0; x < sizeX; x++)
		for(int s = 0; s < 2; s++)
			geometry->setCoordinates({x, s},	{1.*x, 0., 0.});
}

void ModelFactory::addSquareGeometry2D(
	Model *model,
	initializer_list<int> size)
{
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
	initializer_list<int> size)
{
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
