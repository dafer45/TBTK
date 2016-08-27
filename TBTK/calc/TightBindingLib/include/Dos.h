/** @package TBTKcalc
 *  @file Dos.h
 *  @brief Property container for density of states (DOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS
#define COM_DAFER45_TBTK_DOS

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
namespace Property{

/** Container for density of states (DOS). */
class Dos{
public:
	/** Constructor. */
	Dos(double lowerBound, double upperBound, int resolution);

	/** Destructor. */
	~Dos();

	/** Get lower bound for the energy. */
	double getLowerBound();

	/** Get upper bound for the energy. */
	double getUpperBound();

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution();

	/** Get DOS data. */
	const double* getData();
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
};

inline double Dos::getLowerBound(){
	return lowerBound;
}

inline double Dos::getUpperBound(){
	return upperBound;
}

inline int Dos::getResolution(){
	return resolution;
}

inline const double* Dos::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
