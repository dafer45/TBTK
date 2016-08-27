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
	class FileReader;
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
	int getDimensions() const;

	/** Get the ranges for the dimensions of the density. */
	const int* getRanges() const;

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Get number of data elementes. */
	int getSize() const;

	/** Get spin-polarized LDOS data. */
	const std::complex<double>* getData() const;
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

	/** FileReader is a friend class to allow it to write spin-polarized
	 *  LDOS data. */
	friend class TBTK::FileReader;
};

inline int SpinPolarizedLdos::getDimensions() const{
	return dimensions;
}

inline const int* SpinPolarizedLdos::getRanges() const{
	return ranges;
}

inline double SpinPolarizedLdos::getLowerBound() const{
	return lowerBound;
}

inline double SpinPolarizedLdos::getUpperBound() const{
	return upperBound;
}

inline int SpinPolarizedLdos::getResolution() const{
	return resolution;
}

inline int SpinPolarizedLdos::getSize() const{
	return size;
}

inline const std::complex<double>* SpinPolarizedLdos::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
