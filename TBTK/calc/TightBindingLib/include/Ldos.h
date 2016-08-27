/** @package TBTKcalc
 *  @file Ldos.h
 *  @brief Property container for local density of states (LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
namespace Property{

/** Container for local density of states (LDOS). */
class Ldos{
public:
	/** Constructor. */
	Ldos(int dimensions, const int *ranges, double lowerBound, double upperBound, int resolution);

	/** Destructor. */
	~Ldos();

	/** Get the dimension of the LDOS. (Excluding energy dimension). */
	int getDimensions();

	/** Get the ranges for the dimensions of the LDOS. */
	const int* getRanges();

	/** Get lower bound for the energy. */
	double getLowerBound();

	/** Get upper bound for the energy. */
	double getUpperBound();

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

	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals). */
	int resolution;

	/** Number of data elements. */
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

inline int Ldos::getDimensions(){
	return dimensions;
}

inline const int* Ldos::getRanges(){
	return ranges;
}

inline double Ldos::getLowerBound(){
	return lowerBound;
}

inline double Ldos::getUpperBound(){
	return upperBound;
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
