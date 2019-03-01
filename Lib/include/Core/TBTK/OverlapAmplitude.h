/* Copyright 2019 Kristofer Björnson
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
 *  @file OverlapAmplitude.h
 *  @brief Overlap amplitude between state 'bra' and 'ket'.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_OVERLAP_AMPLITUDE
#define COM_DAFER45_TBTK_OVERLAP_AMPLITUDE

#include "TBTK/Index.h"
#include "TBTK/PseudoSerializable.h"
#include "TBTK/Serializable.h"

#include <complex>
#include <initializer_list>
#include <vector>

namespace TBTK{

/** @brief Overlap amplitude between state 'bra' and 'ket'. */
class OverlapAmplitude : public PseudoSerializable{
public:
	/** Constructor. */
	OverlapAmplitude();

	/** Constructs an OverlapAmplitude from a value and two @link Index
	 *  Indices @endlink.
	 *
	 *  @param amplitude The amplitude value.
	 *  @param braIndex The Index of the bra state.
	 *  @param ketIndex The Index of the ket state. */
	OverlapAmplitude(
		std::complex<double> amplitude,
		const Index &braIndex,
		const Index &ketIndex
	);

	/** Constructor. Takes a callback function rather than a paramater
	 *  value. The callback function has to be defined such that it returns
	 *  a value for the given indices when called at run time.
	 *
	 *  @param amplitudeCallback A callback function that is able to return
	 *  a value when passed bra.
	 *
	 *  @param braIndex The Index of the bra state.
	 *  @param ketIndex The Index of the ket state. */
	OverlapAmplitude(
		std::complex<double> (*amplitudeCallback)(
			const Index &braIndex,
			const Index &ketIndex
		),
		const Index &braIndex,
		const Index &ketIndex
	);

	/** Constructor. Constructs the OverlapAmplitude from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the OverlapAmplitude.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	OverlapAmplitude(
		const std::string &serializeation,
		Serializable::Mode mode
	);

	/** Get the amplitude value \f$\langle\Psi_{bra}|\Psi_{ket}\rangle\f$.
	 *
	 *  @return The value of the amplitude. */
	std::complex<double> getAmplitude() const;

	/** Get bra index.
	 *
	 *  @return The bra-Index. */
	const Index& getBraIndex() const;

	/** Get ket index.
	 *
	 *  @return The ket-Index. */
	const Index& getKetIndex() const;

	/** Get whether the value of the OverlapAmplitude is determined through
	 *  a callback.
	 *
	 *  @return True if the value of the OverlapAmplitude is determined
	 *  through a callback. */
	bool getIsCallbackDependent() const;

	/** Get the callback that is used to determine the value of the
	 *  OverlapAmplitude.
	 *
	 *  @return The callback that is used to determine the value of the
	 *  OverlapAmplitude. Returns nullptr if no callback is used. */
	std::complex<double> (*getAmplitudeCallback() const)(
		const Index &braIndex,
		const Index &ketIndex
	);

	/** Get string representation of the OverlapAmplitude.
	 *
	 *  @return A string representation of the OverlapAmplitude. */
	std::string toString() const;

	/** Serialize OverlapAmplitude. Note that OverlapAmplitude is
	 *  pseudo-Serializable in that it implements the Serializable
	 *  interface, but does so non-virtually.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string representation of the OverlapAmplitude.
	 */
	std::string serialize(Serializable::Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the OverlapAmplitude. */
	unsigned int getSizeInBytes() const;
private:
	/** Amplitude \f$\langle\Psi_{bra}|\Psi_{ket}\rangle\f$. Will be used
	 *  if amplitudeCallback is nullptr. */
	std::complex<double> amplitude;

	/** Callback function for runtime evaluation of amplitudes. Will be
	 *  called if not nullptr. */
	std::complex<double> (*amplitudeCallback)(
		const Index &braIndex,
		const Index &ketIndex
	);

	/** Index of the bra-state. */
	Index braIndex;

	/** Index of the ket-state. */
	Index ketIndex;

};

inline OverlapAmplitude::OverlapAmplitude(){
	amplitudeCallback = nullptr;
}

inline std::complex<double> OverlapAmplitude::getAmplitude() const{
	if(amplitudeCallback)
		return amplitudeCallback(braIndex, ketIndex);
	else
		return amplitude;
}

inline const Index& OverlapAmplitude::getBraIndex() const{
	return braIndex;
}

inline const Index& OverlapAmplitude::getKetIndex() const{
	return ketIndex;
}

inline bool OverlapAmplitude::getIsCallbackDependent() const{
	if(amplitudeCallback == nullptr)
		return false;
	else
		return true;
}

inline std::complex<double> (*OverlapAmplitude::getAmplitudeCallback() const)(
	const Index &braIndex,
	const Index &ketIndex
){
	return amplitudeCallback;
}

inline std::string OverlapAmplitude::toString() const{
	std::string str;
	str += "("
			+ std::to_string(real(amplitude))
			+ ", " + std::to_string(imag(amplitude))
		+ ")"
		+ ", " + braIndex.toString()
		+ ", " + ketIndex.toString();

	return str;
}

inline unsigned int OverlapAmplitude::getSizeInBytes() const{
	return sizeof(OverlapAmplitude)
		- sizeof(braIndex)
		- sizeof(ketIndex)
		+ braIndex.getSizeInBytes()
		+ ketIndex.getSizeInBytes();
}

};	//End of namespace TBTK

#endif
