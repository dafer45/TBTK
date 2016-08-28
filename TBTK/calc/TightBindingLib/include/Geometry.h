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
	void addPoint(const Index &index, const double *coordinates, const int *specifiers = NULL);
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

};	//End of namespace TBTK

#endif
