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
	class FileReader;
namespace Property{

/** Container for local density of states (LDOS). */
class EigenValues{
public:
	/** Constructor. */
	EigenValues(int size);

	/** Destructor. */
	~EigenValues();

	/** Get number of eigen values. */
	int getSize() const;

	/** Get eigen values. */
	const double* getData() const;
private:
	/** Number of elements in data. */
	int size;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write
	 * EigenValues data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write
	 *  EigenValues data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write
	 *  EigenValues data. */
	friend class TBTK::FileReader;
};

inline int EigenValues::getSize() const{
	return size;
}

inline const double* EigenValues::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
