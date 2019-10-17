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
 *  @brief Hopping amplitude from state 'from' to state 'to'.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE

#include "TBTK/Index.h"
#include "TBTK/Serializable.h"

#include <complex>
#include <initializer_list>
#include <tuple>
#include <vector>

namespace TBTK{

/** \brief Enum used to indicate the Hermitian conjugate. */
enum HermitianConjugate {HC};

/** @brief Hopping amplitude from state 'from' to state 'to'.
 *
 *  A hopping amplitude is a coefficeint \f$a_{ij}\f$ in a bilinear Hamiltonian
 *  \f$H = \sum_{ij}a_{ij}c_{i}^{\dagger}c_{j}\f$, where \f$i\f$ and \f$j\f$
 *  are reffered to using 'to' and 'from' respectively. The constructors can be
 *  called with the parameters either in the order (from, to, value) or the
 *  order (value, to, from). The former follows the order in which the process
 *  can be thought of as happening, while the later corresponds to the order in
 *  which values and operators stands in the Hamiltonian.
 *
 *  <b>Example:</b>
 *  \snippet Core/HoppingAmplitude.cpp HoppingAmplitude
 *  <b>Output:</b>
 *  \snippet output/Core/HoppingAmplitude.output HoppingAmplitude */
class HoppingAmplitude{
public:
	/** Abstract base class for callbacks that allow for delayed
	 *  determination of the HoppingAmplitude's value. */
	class AmplitudeCallback{
	public:
		/** Function responsible for returning the value of the
		 *  HoppingAmplitude for the given indices.
		 *
		 *  @param to To-index to determine the value of the
		 *  HoppingAmplitude for.
		 *
		 *  @param from From-index to determine the value of the
		 *  HoppingAmplitude for.
		 *
		 *  @return The value of the HoppingAmplitude for the given
		 *  indices. */
		virtual std::complex<double> getHoppingAmplitude(
			const Index &to,
			const Index &from
		) const = 0;
	};

	//TBTKFeature Core.HoppingAmplitude.Construction.1 2019-09-23
	/** Constructs an uninitialized HoppingAmplitude. */
	HoppingAmplitude();

	//TBTKFeature Core.HoppingAmplitude.Construction.2 2019-09-23
	/** Constructs a HoppingAmplitude from a value and two @link Index
	 *  Indices@endlink.
	 *
	 *  @param amplitude The amplitude value.
	 *  @param toIndex The left index (i or to-Index) on the
	 *  HoppingAmplitude.
	 *
	 *  @param fromIndex The right index (j or from-Index) on the
	 *  HoppingAmplitude. */
	HoppingAmplitude(
		std::complex<double> amplitude,
		Index toIndex,
		Index fromIndex
	);

	//TBTKFeature Core.HoppingAmplitude.Construction.3 2019-09-23
	/** Constructor. Takes an AmplitudeCallback rather than a paramater
	 *  value. The AmplitudeCallback has to be defined such that it returns
	 *  a value for the given indices when called at run time.
	 *
	 *  @param amplitudeCallback An AmplitudeCallback that is able to
	 *  return a value when passed toIndex and fromIndex.
	 *
	 *  @param toIndex The left index (i or to-Index) on the
	 *  HoppingAmplitude.
	 *
	 *  @param fromIndex The right index (j or from-Index) on the
	 *  HoppingAmplitude. */
	HoppingAmplitude(
		const AmplitudeCallback &callback,
		Index toIndex,
		Index fromIndex
	);

	//TBTKFeature Core.HoppingAmplitude.Copy.1 2019-09-23
	//TBTKFeature Core.HoppingAmplitude.Copy.2 2019-09-23
	/** Copy constructor.
	 *
	 *  @param ha HoppingAmplitude to copy. */
	HoppingAmplitude(const HoppingAmplitude &ha);

	//TBTKFeature Core.HoppingAmplitude.Serialization.1 2019-09-23
	/** Constructor. Constructs the HoppingAmplitude from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Index.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	HoppingAmplitude(
		const std::string &serialization,
		Serializable::Mode mode
	);

	//TBTKFeature Core.HoppingAmplitude.getHermitianConjugate.1 2019-09-23
	//TBTKFeature Core.HoppingAmplitude.getHermitianConjugate.2 2019-09-23
	/** Get the Hermitian cojugate of the HoppingAmplitude.
	 *
	 *  @return The Hermitian conjugate of the HoppingAmplitude. */
	HoppingAmplitude getHermitianConjugate() const;

	/** Print HoppingAmplitude. Mainly for debugging. */
	void print() const;

	//TBTKFeature Core.HoppingAmplitude.getAmplitude.1 2019-09-23
	//TBTKFeature Core.HoppingAmplitude.getAmplitude.2 2019-09-23
	/** Get the amplitude value \f$a_{ij}\f$.
	 *
	 *  @return The value of the amplitude. */
	std::complex<double> getAmplitude() const;

	//TBTKFeature Core.HoppingAmplitude.operatorAddition.1.C++ 2019-09-23
	/** Addition operator. Creates a tuple containing the HoppingAmplitude
	 *  and its Hermitian conjugate. Used to allow the syntax<br>
	 *  model << hoppingAmplitude + HC.
	 *
	 *  @param hc Should be HC.
	 *
	 *  @return HoppingAmplitude tuple containing the original
	 *  HoppingAmplitude and its Hermitian conjugate. */
	std::tuple<HoppingAmplitude, HoppingAmplitude> operator+(
		const HermitianConjugate hc
	);

	//TBTKFeature Core.HoppingAmplitude.getToIndex.1 2019-09-23
	/** Get to index.
	 *
	 *  @return The to-Index. */
	const Index& getToIndex() const;

	//TBTKFeature Core.HoppingAmplitude.getFromIndex.1 2019-09-23
	/** Get from index.
	 *
	 *  @return The from Index. */
	const Index& getFromIndex() const;

	//TBTKFeature Core.HoppingAmplitude.getIsCallbackDependent.1 2019-09-23
	//TBTKFeature Core.HoppingAmplitude.getIsCallbackDependent.2 2019-09-23
	/** Get whether the value of the HoppingAmplitude is determined through
	 *  an AmplitudeCallback.
	 *
	 *  @return True if the value of the HoppingAmplitude is determined
	 *  through an AmplitudeCallback. */
	bool getIsCallbackDependent() const;

	//TBTKFeature Core.HoppingAmplitude.getAmplitudeCallback.1 2019-09-23
	//TBTKFeature Core.HoppingAmplitude.getAmplitudeCallback.2 2019-09-23
	/** Get the AmplitudeCallback that is used to determine the value of
	 *  the HoppingAmplitude. This function stops execution if no
	 *  AmplitudeCallback is used for the HoppingAmplitude. Therefore
	 *  always first check whether the HoppingAmplitude is callback
	 *  dependent with getIsCallbackDependent().
	 *
	 *  @return The AmplitudeCallback that is used to determine the value
	 *  of the HoppingAmplitude. */
	const AmplitudeCallback& getAmplitudeCallback() const;

	/** Get string representation of the HoppingAmplitude.
	 *
	 *  @return A string representation of the HoppingAmplitude. */
	std::string toString() const;

	/** Writes the HoppingAmplitudes toString()-representation to a stream.
	 *
	 *  @param stream The stream to write to.
	 *  @param hoppingAmplitude The HoppingAmplitude to write.
	 *
	 *  @return Reference to the output stream just written to. */
	friend std::ostream& operator<<(
		std::ostream &stream,
		const HoppingAmplitude &hoppingAmplitude
	);

	//TBTKFeature Core.HoppingAmplitude.Serialization.1 2019-09-23
	/** Serialize HoppingAmplitude. Note that HoppingAmplitude is
	 *  pseudo-Serializable in that it implements the Serializable
	 * interface, but does so non-virtually.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string representation of the HoppingAmplitude.
	 */
	std::string serialize(Serializable::Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the HoppingAmplitude. */
	unsigned int getSizeInBytes() const;
private:
	/** Amplitude \f$a_{ij}\f$. Will be used if amplitudeCallback is NULL.
	 */
	std::complex<double> amplitude;

	/** AmplitudeCallback for runtime evaluation of amplitudes. Will be
	 *  called if not a nullptr. */
	const AmplitudeCallback *amplitudeCallback;

	/** Index to jump from (annihilate). */
	Index fromIndex;

	/** Index to jump to (create). */
	Index toIndex;

};

inline std::complex<double> HoppingAmplitude::getAmplitude() const{
	if(amplitudeCallback){
		return amplitudeCallback->getHoppingAmplitude(
			toIndex,
			fromIndex
		);
	}
	else{
		return amplitude;
	}
}

inline std::tuple<HoppingAmplitude, HoppingAmplitude> HoppingAmplitude::operator+(
	HermitianConjugate hc
){
	return std::make_tuple(*this, this->getHermitianConjugate());
}

inline const Index& HoppingAmplitude::getToIndex() const{
	return toIndex;
}

inline const Index& HoppingAmplitude::getFromIndex() const{
	return fromIndex;
}

inline bool HoppingAmplitude::getIsCallbackDependent() const{
	if(amplitudeCallback == nullptr)
		return false;
	else
		return true;
}

inline const HoppingAmplitude::AmplitudeCallback&
HoppingAmplitude::getAmplitudeCallback() const{
	if(amplitudeCallback != nullptr){
		return *amplitudeCallback;
	}
	else{
		TBTKExit(
			"HoppingAmpliude::getAmplitudeCallback()",
			"Tried to access AmplitudeCallback from a"
			<< " HoppingAmplitude without an AmplitudeCallback.",
			""
		);
	}
}

inline std::string HoppingAmplitude::toString() const{
	std::stringstream stream;
	stream << "HoppingAmplitude:\n";
	if(amplitudeCallback == nullptr){
		stream << "\tIs callback dependent: False\n";
		stream << "\tAmplitude: " << amplitude << "\n";
	}
	else{
		stream << "\tIs callback dependent: True\n";
		stream << "\tAmplitude: "
			<< amplitudeCallback->getHoppingAmplitude(
				toIndex,
				fromIndex
			) << "\n";
	}
	stream << "\tTo-Index: " << toIndex << "\n";
	stream << "\tFrom-Index: " << fromIndex;

	return stream.str();
}

inline std::ostream& operator<<(
	std::ostream &stream,
	const HoppingAmplitude &hoppingAmplitude
){
	stream << hoppingAmplitude.toString();

	return stream;
}

inline unsigned int HoppingAmplitude::getSizeInBytes() const{
	return sizeof(HoppingAmplitude)
		- sizeof(fromIndex)
		- sizeof(toIndex)
		+ fromIndex.getSizeInBytes()
		+ toIndex.getSizeInBytes();
}

};	//End of namespace TBTK

#endif
