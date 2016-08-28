/** @package TBTKcalc
 *  @file Geometry.h
 *  @brief Contains geometric inforamtion about a model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GEOMETRY
#define COM_DAFER45_TBTK_GEOMETRY

namespace TBTK{

class Geometry{
public:
	Geometry(int dimensions, int additionalSpecifiers);
	~Geometry();
private:
	int dimensions;
	int additionalSpecifiers;

	double *coordinates;
	int *specifiers;
};

};	//End of namespace TBTK

#endif
