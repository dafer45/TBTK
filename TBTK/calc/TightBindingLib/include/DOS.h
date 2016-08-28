/** @package TBTKcalc
 *  @file DOS.h
 *  @brief Property container for density of states (DOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS
#define COM_DAFER45_TBTK_DOS

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for density of states (DOS). */
class DOS{
public:
	/** Constructor. */
	DOS(double lowerBound, double upperBound, int resolution);

	/** Destructor. */
	~DOS();

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Get DOS data. */
	const double* getData() const;
private:
	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	int resolution;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write DOS data
	 */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write DOS data
	 */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write DOS data. */
	friend class TBTK::FileReader;
};

inline double DOS::getLowerBound() const{
	return lowerBound;
}

inline double DOS::getUpperBound() const{
	return upperBound;
}

inline int DOS::getResolution() const{
	return resolution;
}

inline const double* DOS::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
