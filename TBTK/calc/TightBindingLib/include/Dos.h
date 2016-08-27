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
	Dos(double lowerLimit, double upperLimit, int resolution);

	/** Destructor. */
	~Dos();

	/** Get lower limit for the energy. */
	double getLowerLimit();

	/** Get upper limit for the energy. */
	double getUpperLimit();

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution();

	/** Get DOS data. */
	const double* getData();
private:
	/** Lower limit for the energy. */
	double lowerLimit;

	/** Upper limit for the energy. */
	double upperLimit;

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

inline double Dos::getLowerLimit(){
	return lowerLimit;
}

inline double Dos::getUpperLimit(){
	return upperLimit;
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
