/** @package TBTKcalc
 *  @file SpinPolarizedLdos.h
 *  @brief Property container for spin-polarized local density of states
 *  (spin-polarized LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SPIN_POLARIZED_LDOS
#define COM_DAFER45_TBTK_SPIN_POLARIZED_LDOS

#include <complex>

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
namespace Property{

/** Container for spin-polarized local density of states (spin-polarized LDOS).
 */
class SpinPolarizedLdos{
public:
	/** Constructor. */
	SpinPolarizedLdos(int dimensions, const int *ranges, double lowerBound, double upperBound, int resolution);

	/** Destructor. */
	~SpinPolarizedLdos();

	/** Get the dimension of the spin-polarized LDOS. (Excluding energy
	 *  dimension) */
	int getDimensions();

	/** Get the ranges for the dimensions of the density. */
	const int* getRanges();

	/** Get lower bound for the energy. */
	double getLowerBound();

	/** Get upper bound for the energy. */
	double getUpperBound();

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution();

	/** Get number of data elementes. */
	int getSize();

	/** Get spin-polarized LDOS data. */
	const std::complex<double>* getData();
private:
	/**Dimension of the density. (Excluding energy dimension) */
	int dimensions;

	/** Ranges for the dimensions of the spin-polarized LDOS*/
	int *ranges;

	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	int resolution;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	std::complex<double> *data;

	/** CPropertyExtractor is a friend class to allow it to write
	 *  spin-polarized LDOS data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write
	 *  spin-polarized LDOS data. */
	friend class TBTK::DPropertyExtractor;
};

inline int SpinPolarizedLdos::getDimensions(){
	return dimensions;
}

inline const int* SpinPolarizedLdos::getRanges(){
	return ranges;
}

inline double SpinPolarizedLdos::getLowerBound(){
	return lowerBound;
}

inline double SpinPolarizedLdos::getUpperBound(){
	return upperBound;
}

inline int SpinPolarizedLdos::getResolution(){
	return resolution;
}

inline int SpinPolarizedLdos::getSize(){
	return size;
}

inline const std::complex<double>* SpinPolarizedLdos::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
