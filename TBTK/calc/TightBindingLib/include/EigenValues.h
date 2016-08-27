/** @package TBTKcalc
 *  @file EigenValues.h
 *  @brief Property container for eigen values
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EIGEN_VALUES
#define COM_DAFER45_TBTK_EIGEN_VALUES

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
namespace Property{

/** Container for local density of states (LDOS). */
class EigenValues{
public:
	/** Constructor. */
	EigenValues(int size);

	/** Destructor. */
	~EigenValues();

	/** Get number of eigen values. */
	int getSize();

	/** Get eigen values. */
	const double* getData();
private:
	/** Number of elements in data. */
	int size;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write LDOS data
	 */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write LDOS data
	 */
	friend class TBTK::DPropertyExtractor;
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
