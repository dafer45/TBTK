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

class HoppingAmplitude{
public:
	/** Index to jump from and to, respectively. (Annihilate/create)*/
	Index fromIndex;
	Index toIndex;

	/** Constructors.*/
	HoppingAmplitude(Index fromIndex, Index toIndex, std::complex<double> amplitude);
	HoppingAmplitude(Index fromIndex, Index toIndex, std::complex<double> (*amplitudeCallback)(Index, Index));
	HoppingAmplitude(std::complex<double> amplitude, Index toIndex, Index fromIndex);
	HoppingAmplitude(std::complex<double> (*amplitudeCallback)(Index, Index), Index toIndex, Index fromIndex);

	HoppingAmplitude getHermitianConjugate();

	void print();

	std::complex<double> getAmplitude();
private:
	/** Amplitude for the process. */
	std::complex<double> amplitude;
	std::complex<double> (*amplitudeCallback)(Index fromIndex, Index toIndex);
};

inline std::complex<double> HoppingAmplitude::getAmplitude(){
	if(amplitudeCallback)
		return amplitudeCallback(fromIndex, toIndex);
	else
		return amplitude;
}

#endif

