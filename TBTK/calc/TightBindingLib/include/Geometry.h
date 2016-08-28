/** @package TBTKcalc
 *  @file Geometry.h
 *  @brief Contains geometric inforamtion about a model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GEOMETRY
#define COM_DAFER45_TBTK_GEOMETRY

#include "Model.h"

namespace TBTK{

class Geometry{
public:
	/** Constructor. */
	Geometry(int dimensions, int numSpecifiers, Model *parentModel);

	/** Destructor. */
	~Geometry();

	/** Add a coordinate (and specifiers) for an index. */
	void setCoordinates(const Index &index, const double *coordinates, const int *specifiers = NULL);

	/** Get dimensions. */
	int getDimensions();

	/** Get number of specifiers. */
	int getNumSpecifiers();

	/** Get Coordinates. */
	const double* getCoordinates(const Index &index);

	/** Get specifier. */
	int getSpecifier(const Index &index, int specifier);

	/** Get distance between positions corresponding to index1 and index2.
	 */
	double getDistance(const Index &index1, const Index &index2);
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

inline int Geometry::getSpecifier(const Index &index, int specifier){
	return specifiers[numSpecifiers*parentModel->getBasisIndex(index) + specifier];
}

};	//End of namespace TBTK

#endif
