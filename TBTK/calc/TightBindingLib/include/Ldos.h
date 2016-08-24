/** @package TBTKcalc
 *  @file Ldos.h
 *  @brief Property container for local density of states (LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

namespace TBTK{
namespace Property{

/** Container for local density of states (LDOS). */
class Ldos{
public:
	/** Constructor. */
	Ldos(int dimensions, const int *ranges, double lowerLimit, double upperLimit, int resolution);

	/** Destructor. */
	~Ldos();

	/** Get the dimension of the LDOS. (Excluding energy dimension). */
	int getDimensions();

	/** Get the ranges for the dimensions of the LDOS. */
	const int* getRanges();

	/** Get lower limit for the energy. */
	double getLowerLimit();

	/** Get upper limit for the energy. */
	double getUpperLimit();

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution();

	/** Get the number of data elements. */
	int getSize();

	/** Get LDOS data. */
	const double* getData();
private:
	/** Dimension of the LDOS. */
	int dimensions;

	/** Ranges for the dimensions of the LDOS. */
	int *ranges;

	/** Lower limit for the energy. */
	double lowerLimit;

	/** Upper limit for the energy. */
	double upperLimit;

	/** Energy resolution. (Number of energy intervals). */
	int resolution;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write LDOS data
	 */
	friend class CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write LDOS data
	 */
	friend class DPropertyExtractor;
};

inline int Ldos::getDimensions(){
	return dimensions;
}

inline const int* Ldos::getRanges(){
	return ranges;
}

inline double Ldos::getLowerLimit(){
	return lowerLimit;
}

inline double Ldos::getUpperLimit(){
	return upperLimit;
}

inline int Ldos::getResolution(){
	return resolution;
}

inline int Ldos::getSize(){
	return size;
}

inline const double* Ldos::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
