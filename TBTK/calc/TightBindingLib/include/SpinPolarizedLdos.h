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
namespace Property{

/** Container for spin-polarized local density of states (spin-polarized LDOS).
 */
class SpinPolarizedLdos{
public:
	SpinPolarizedLdos(int rank, const int *dims, double lowerLimit, double upperLimit, int resolution);
	~SpinPolarizedLdos();

	int getRank();
	const int* getDims();
	double getLowerLimit();
	double getUpperLimit();
	int getResolution();
	int getSize();
	const std::complex<double>* getData();
private:
	int rank;
	int *dims;
	double lowerLimit;
	double upperLimit;
	int resolution;
	int size;
	std::complex<double> *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline int SpinPolarizedLdos::getRank(){
	return rank;
}

inline const int* SpinPolarizedLdos::getDims(){
	return dims;
}

inline double SpinPolarizedLdos::getLowerLimit(){
	return lowerLimit;
}

inline double SpinPolarizedLdos::getUpperLimit(){
	return upperLimit;
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
