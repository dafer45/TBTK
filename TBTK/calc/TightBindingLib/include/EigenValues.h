/** @package TBTKcalc
 *  @file EigenValues.h
 *  @brief Property container for eigen values
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EIGEN_VALUES
#define COM_DAFER45_TBTK_EIGEN_VALUES

namespace TBTK{
namespace Property{

/** Container for local density of states (LDOS). */
class EigenValues{
public:
	EigenValues(int size);
	~EigenValues();

	int getSize();
	const double* getData();
private:
	int size;
	double *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline int EigenValues::getSize(){
	return size;
}

inline const double* EigenValues::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
