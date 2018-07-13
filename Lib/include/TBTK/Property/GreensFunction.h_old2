/* Copyright 2017 Kristofer Björnson
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
 *  @file GreensFunction.h
 *  @brief Property container for Green's function.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GREENS_FUNCTION
#define COM_DAFER45_TBTK_GREENS_FUNCTION

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Property container for Green's function. */
class GreensFunction : public AbstractProperty<std::complex<double>>{
public:
	/** Enum class for specifying Green's function type. */
	enum class Type{
		Advanced,
		Retarded,
		Principal,
		NonPrincipal
	};

	/** Constructs an uninitialized GreensFunction. */
	GreensFunction();

	/** Constructs a GreensFunction on the Custom format. [See
	 *  AbstractProperty for detailed information about the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param type The GreensFunction type.
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	GreensFunction(
		const IndexTree &indexTree,
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructs a GreensFunction on the Custom format and initializes it
	 *  with data. [See AbstractProperty for detailed information about the
	 *  Custom format and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param type The GreensFunction type.
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	GreensFunction(
		const IndexTree &indexTree,
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Copy constructor. */
//	GreensFunction(const GreensFunction &greensFunction);

	/** Move constructor. */
//	GreensFunction(GreensFunction &&greensFunction);

	/** Destructor. */
//	~GreensFunction();

	/** Get Green's function type.
	 *
	 *  @return The Green's function type. */
	Type getType() const;

	/** Get lower bound for the energy.
	 *
	 *  @return Lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy.
	 *
	 *  @return Upper bound for the energy. */
	double getUpperBound() const;

	/** Get the energy resolution (number of points used for the energy
	 *  axis).
	 *
	 *  @return The energy resolution. */
	unsigned int getResolution() const;

	/** Get GreensFunction data. */
//	const std::complex<double>* getData() const;

	/** Assignment operator. */
//	const GreensFunction& operator=(const GreensFunction &rhs);

	/** Move assignment operator. */
//	const GreensFunction& operator=(GreensFunction &&rhs);
private:
	/** Green's function type. */
	Type type;

	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	unsigned int resolution;

	/** Actual data. */
//	std::complex<double> *data;
};

inline GreensFunction::Type GreensFunction::getType() const{
	return type;
}

inline double GreensFunction::getLowerBound() const{
	return lowerBound;
}

inline double GreensFunction::getUpperBound() const{
	return upperBound;
}

inline unsigned int GreensFunction::getResolution() const{
	return resolution;
}

/*inline const std::complex<double>* GreensFunction::getData() const{
	return data;
}*/

/*inline const GreensFunction& GreensFunction::operator=(
	const GreensFunction &rhs
){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);

		type = rhs.type;
		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;

//		data = new std::complex<double>[resolution];
//		for(unsigned int n = 0; n < resolution; n++)
//			data[n] = rhs.data[n];
	}

	return *this;
}

inline const GreensFunction& GreensFunction::operator=(
	GreensFunction &&rhs
){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));

		type = rhs.type;
		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
//		data = rhs.data;
//		rhs.data = nullptr;
	}

	return *this;
}*/

};	//End namespace Property
};	//End namespace TBTK

#endif
