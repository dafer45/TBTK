/** @package TBTKcalc
 *  @file Geometry.h
 *  @brief Contains geometric inforamtion about a model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GEOMETRY
#define COM_DAFER45_TBTK_GEOMETRY

#include "Model.h"
#include <initializer_list>

namespace TBTK{

class Geometry{
public:
	/** Constructor. */
	Geometry(int dimensions, int numSpecifiers, Model *parentModel);

	/** Destructor. */
	~Geometry();

	/** Add a coordinate (and specifiers) for an index. */
	void setCoordinates(const Index &index, std::initializer_list<double> coordinates, std::initializer_list<int> specifiers = {});

	/** Get dimensions. */
	int getDimensions();

	/** Get number of specifiers. */
	int getNumSpecifiers();

	/** Get Coordinates using a physical index. */
	const double* getCoordinates(const Index &index);

	/** Get coordinates using a Hilbert space index. */
	const double* getCoordinates(int index);

	/** Get specifier using a physical index. */
	int getSpecifier(const Index &index, int specifier);

	/** Get specifier using a Hilbert space index. */
	int getSpecifier(int index, int specifier);

	/** Get distance between positions corresponding to index1 and index2,
	 *  using physical indices. */
	double getDistance(const Index &index1, const Index &index2);

	/** Get distance between positions corresponding to index1 and index2,
	 *  using physical indices. */
	double getDistance(int index1, int index2);
private:
	/** Number of spatial dimensions. */
	int dimensions;

	/** Number of specifiers. */
	int numSpecifiers;

	/** Spatial dimensions. */
	double *coordinates;

	/** Specifiers such as orbital number, spin-species, etc. */
	int *specifiers;

	/** The Model that the geometry corresponds to. */
	Model *parentModel;
};

inline int Geometry::getDimensions(){
	return dimensions;
}

inline int Geometry::getNumSpecifiers(){
	return numSpecifiers;
}

inline const double* Geometry::getCoordinates(const Index &index){
	return &(coordinates[dimensions*parentModel->getBasisIndex(index)]);
}

inline const double* Geometry::getCoordinates(int index){
	return &(coordinates[dimensions*index]);
}

inline int Geometry::getSpecifier(const Index &index, int specifier){
	return specifiers[numSpecifiers*parentModel->getBasisIndex(index) + specifier];
}

inline int Geometry::getSpecifier(int index, int specifier){
	return specifiers[numSpecifiers*index + specifier];
}

inline double Geometry::getDistance(int index1, int index2){
	double distanceSquared = 0.;
	for(int n = 0; n < dimensions; n++){
		double difference = coordinates[index1] - coordinates[index2];
		distanceSquared += difference*difference;
	}

	return sqrt(distanceSquared);
}

};	//End of namespace TBTK

#endif
