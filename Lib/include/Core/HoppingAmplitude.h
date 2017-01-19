/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file HoppingAmplitude.h
 *  @brief Hopping amplitude from state 'from' to 'to'
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE

#include "Index.h"

#include <complex>
#include <initializer_list>
#include <vector>

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
/*	HoppingAmplitude(
		Index fromIndex,
		Index toIndex,
		std::complex<double> amplitude
	);*/

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 * a value for the given indices when called at run time. */
/*	HoppingAmplitude(
		Index fromIndex,
		Index toIndex,
		std::complex<double> (*amplitudeCallback)(Index, Index)
	);*/

	/** Constructor. */
	HoppingAmplitude(
		std::complex<double> amplitude,
		Index toIndex,
		Index fromIndex
	);

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 * a value for the given indices when called at run time. */
	HoppingAmplitude(
		std::complex<double> (*amplitudeCallback)(Index, Index),
		Index toIndex,
		Index fromIndex
	);

	/** Constructor. Takes an additional parameter specifying which unit
	 *  cell the toIndex belongs to. */
/*	HoppingAmplitude(
		std::complex<double> amplitude,
		Index toIndex,
		Index fromIndex,
		Index toUnitCell
	);*/

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 *  a value for the given indices when called at run time. Also takes
	 *  an additional Index specifying which unit cell the toIndex belongs
	 *  to. */
/*	HoppingAmplitude(
		std::complex<double> (*amplitudeCallback)(Index, Index),
		Index toIndex,
		Index fromIndex,
		Index toUnitCell
	);*/

	/** Copy constructor. */
	HoppingAmplitude(const HoppingAmplitude &ha);

	/** Get the Hermitian cojugate of the HoppingAmplitude. */
	HoppingAmplitude getHermitianConjugate() const;

	/** Print HoppingAmplitude. Mainly for debugging. */
	void print();

	/** Get the amplitude value \f$a_{ij}\f$. */
	std::complex<double> getAmplitude() const;
private:
	/** Amplitude \f$a_{ij}\f$. Will be used if amplitudeCallback is NULL. */
	std::complex<double> amplitude;

	/** Callback function for runtime evaluation of amplitudes. Will be
	 *  called if not NULL. */
	std::complex<double> (*amplitudeCallback)(
		Index toIndex,
		Index fromIndex
	);
};

inline std::complex<double> HoppingAmplitude::getAmplitude() const{
	if(amplitudeCallback)
		return amplitudeCallback(toIndex, fromIndex);
	else
		return amplitude;
}

};	//End of namespace TBTK

#endif

