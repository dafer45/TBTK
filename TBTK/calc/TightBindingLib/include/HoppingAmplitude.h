/** @package TBTKcalc
 *  @file HoppingAmplitude.h
 *  @brief Hopping amplitude from state 'from' to 'to'
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE

#include <complex>
#include <initializer_list>
#include <vector>
#include <iostream>
#include "Index.h"

namespace TBTK{

/** A hopping amplitude is a coefficeint \f$a_{ij}\f$ in a bilinear Hamiltonian
 *  \f$H = \sum_{ij}a_{ij}c_{i}^{\dagger}c_{j}\f$, where \f$i\f$ and \f$j\f$
 *  are reffered to using 'to' and 'from' respectively. The constructors can be
 *  called with the parameters either in the order (from, to, value) or the
 *  order (value, to, from). The former follows the order in which the process
 *  can be thought of as happening, while the later corresponds to the order in
 *  which values and operators stands in the Hamiltonian.
 */
class HoppingAmplitude{
public:
	/** Index to jump from (annihilate). */
	Index fromIndex;

	/** Index to jump to (create). */
	Index toIndex;

	/** Constructor. */
	HoppingAmplitude(Index fromIndex, Index toIndex, std::complex<double> amplitude);

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 * a value for the given indices when called at run time. */
	HoppingAmplitude(Index fromIndex, Index toIndex, std::complex<double> (*amplitudeCallback)(Index, Index));

	/** Constructor. */
	HoppingAmplitude(std::complex<double> amplitude, Index toIndex, Index fromIndex);

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 * a value for the given indices when called at run time. */
	HoppingAmplitude(std::complex<double> (*amplitudeCallback)(Index, Index), Index toIndex, Index fromIndex);

	/** Get the Hermitian cojugate of the HoppingAmplitude. */
	HoppingAmplitude getHermitianConjugate();

	/** Print HoppingAmplitude. Mainly for debugging. */
	void print();

	/** Get the amplitude value \f$a_{ij}\f$. */
	std::complex<double> getAmplitude();
private:
	/** Amplitude \f$a_{ij}\f$. Will be used if amplitudeCallback is NULL. */
	std::complex<double> amplitude;

	/** Callback function for runtime evaluation of amplitudes. Will be
	 *  called if not NULL. */
	std::complex<double> (*amplitudeCallback)(Index fromIndex, Index toIndex);
};

inline std::complex<double> HoppingAmplitude::getAmplitude(){
	if(amplitudeCallback)
		return amplitudeCallback(fromIndex, toIndex);
	else
		return amplitude;
}

};

#endif

