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
		NonPrincipal/*,
		FreePole*/
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

	/** Get lower bound for the energy. (For Format::ArrayFormat). */
	double getArrayLowerBound() const;

	/** Get upper bound for the energy. (For Format::ArrayFormat). */
	double getArrayUpperBound() const;

	/** Get energy resolution (number of energy intervals). (For
	 *  Format::ArrayFormat). */
	unsigned int getArrayResolution() const;

	/** Get GreensFunction data. (For Format::ArrayFormat). */
	const std::complex<double>* getArrayData() const;

	/** Assignment operator. */
	const GreensFunction& operator=(const GreensFunction &rhs);

	/** Move assignment operator. */
	const GreensFunction& operator=(GreensFunction &&rhs);

	/** Function call operator. */
	std::complex<double> operator()(double E) const;

	/** Multiplication operator. */
	GreensFunction operator*(std::complex<double> rhs) const;

	/** Multiplication operator. */
	friend GreensFunction operator*(std::complex<double> lhs, const GreensFunction& rhs);

	/** Division operator. */
	GreensFunction operator/(std::complex<double> rhs) const;
private:
	/** Green's function type. */
	Type type;

	/** Stores the Green's function as a descrete set of values on the real
	 *  axis. */
	class ArrayFormat{
	public:
		/** Lower bound for the energy. */
		double lowerBound;

		/** Upper bound for the energy. */
		double upperBound;

		/** Energy resolution. (Number of energy intervals) */
		unsigned int resolution;

		/** Actual data. */
		std::complex<double> *data;
	};

	/** Union of storage formats. */
	union Storage{
		ArrayFormat arrayFormat;
	};

	/** Actuall storage. */
	Storage storage;
};

inline double GreensFunction::getArrayLowerBound() const{
	return storage.arrayFormat.lowerBound;
}

inline double GreensFunction::getArrayUpperBound() const{
	return storage.arrayFormat.upperBound;
}

inline unsigned int GreensFunction::getArrayResolution() const{
	return storage.arrayFormat.resolution;
}

inline const std::complex<double>* GreensFunction::getArrayData() const{
	return storage.arrayFormat.data;
}

inline const GreensFunction& GreensFunction::operator=(
	const GreensFunction &rhs
){
	if(this != &rhs){
		type = rhs.type;
		storage.arrayFormat.lowerBound = rhs.storage.arrayFormat.lowerBound;
		storage.arrayFormat.upperBound = rhs.storage.arrayFormat.upperBound;
		storage.arrayFormat.resolution = rhs.storage.arrayFormat.resolution;
		storage.arrayFormat.data = new std::complex<double>[storage.arrayFormat.resolution];
		for(unsigned int n = 0; n < storage.arrayFormat.resolution; n++)
			storage.arrayFormat.data[n] = rhs.storage.arrayFormat.data[n];
	}

	return *this;
}

inline const GreensFunction& GreensFunction::operator=(
	GreensFunction &&rhs
){
	if(this != &rhs){
		type = rhs.type;
		storage.arrayFormat.lowerBound = rhs.storage.arrayFormat.lowerBound;
		storage.arrayFormat.upperBound = rhs.storage.arrayFormat.upperBound;
		storage.arrayFormat.resolution = rhs.storage.arrayFormat.resolution;
		storage.arrayFormat.data = rhs.storage.arrayFormat.data;
		rhs.storage.arrayFormat.data = nullptr;
	}

	return *this;
}

inline GreensFunction GreensFunction::operator*(
	std::complex<double> rhs
) const{
	GreensFunction newGreensFunction(
		type,
		storage.arrayFormat.lowerBound,
		storage.arrayFormat.upperBound,
		storage.arrayFormat.resolution
	);

	for(unsigned int n = 0; n < storage.arrayFormat.resolution; n++)
		newGreensFunction.storage.arrayFormat.data[n] = rhs*storage.arrayFormat.data[n];

	return newGreensFunction;
}

inline GreensFunction operator*(
	std::complex<double> lhs,
	const GreensFunction &rhs
){
	GreensFunction newGreensFunction(
		rhs.type,
		rhs.storage.arrayFormat.lowerBound,
		rhs.storage.arrayFormat.upperBound,
		rhs.storage.arrayFormat.resolution
	);

	for(unsigned int n = 0; n < rhs.storage.arrayFormat.resolution; n++)
		newGreensFunction.storage.arrayFormat.data[n] = lhs*rhs.storage.arrayFormat.data[n];

	return newGreensFunction;
}

inline GreensFunction GreensFunction::operator/(
	std::complex<double> rhs
) const{
	GreensFunction newGreensFunction(
		type,
		storage.arrayFormat.lowerBound,
		storage.arrayFormat.upperBound,
		storage.arrayFormat.resolution
	);

	for(unsigned int n = 0; n < storage.arrayFormat.resolution; n++)
		newGreensFunction.storage.arrayFormat.data[n] = storage.arrayFormat.data[n]/rhs;

	return newGreensFunction;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
