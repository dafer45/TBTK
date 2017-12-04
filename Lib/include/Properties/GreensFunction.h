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
 *  @brief Property container for Green's function
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GREENS_FUNCTION
#define COM_DAFER45_TBTK_GREENS_FUNCTION

#include "AbstractProperty.h"
#include "TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** Container for Green's function. */
class GreensFunction{
public:
	/** Enum class for specifying Green's function type. */
	enum class Type{
		Advanced,
		Retarded,
		Principal,
		NonPrincipal
	};

	/** Constructor. */
	GreensFunction(
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructor. */
	GreensFunction(
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Copy constructor. */
	GreensFunction(const GreensFunction &greensFunction);

	/** Move constructor. */
	GreensFunction(GreensFunction &&greensFunction);

	/** Destructor. */
	~GreensFunction();

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution (number of energy intervals). */
	unsigned int getResolution() const;

	/** Get GreensFunction data. */
	const std::complex<double>* getData() const;

	/** Assignment operator. */
	const GreensFunction& operator=(const GreensFunction &rhs);

	/** Move assignment operator. */
	const GreensFunction& operator=(GreensFunction &&rhs);
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
	std::complex<double> *data;
};

inline double GreensFunction::getLowerBound() const{
	return lowerBound;
}

inline double GreensFunction::getUpperBound() const{
	return upperBound;
}

inline unsigned int GreensFunction::getResolution() const{
	return resolution;
}

inline const std::complex<double>* GreensFunction::getData() const{
	return data;
}

inline const GreensFunction& GreensFunction::operator=(
	const GreensFunction &rhs
){
	if(this != &rhs){
		type = rhs.type;
		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
		data = new std::complex<double>[resolution];
		for(unsigned int n = 0; n < resolution; n++)
			data[n] = rhs.data[n];
	}

	return *this;
}

inline const GreensFunction& GreensFunction::operator=(
	GreensFunction &&rhs
){
	if(this != &rhs){
		type = rhs.type;
		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
