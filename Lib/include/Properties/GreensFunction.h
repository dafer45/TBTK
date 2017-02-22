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

#include <complex>
#include <vector>

namespace TBTK{
	class APropertyExtractor;
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
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

	/** Enum class for specifying storage format. */
	enum class Format{
		Array,
		Poles
	};

	/** Constructor. */
	GreensFunction(
		Type type,
		Format format,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructor. */
	GreensFunction(
		Type type,
		Format format,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Constructor. */
	GreensFunction(
		Type type,
		Format format,
		unsigned int numPoles
	);

	/** Constructor. */
	GreensFunction(
		Type type,
		Format format,
		unsigned int numPoles,
		std::complex<double> *positions,
		std::complex<double> *amplitudes
	);

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

	/** Get number of poles. (For Format::PoleFormat). */
	unsigned int getNumPoles() const;

	/** Get pole position. (For Format::PoleFormat). */
	std::complex<double> getPolePosition(unsigned int n) const;

	/** Get pole amplitude. (For Format::PoleFormat). */
	std::complex<double> getPoleAmplitude(unsigned int n) const;
private:
	/** Green's function type. */
	Type type;

	/** Format of the Green's function. */
	Format format;

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

	/** Stores the Green's function as a number of poles. */
	class PoleFormat{
	public:
		/** Number of poles. */
		unsigned int numPoles;

		/** Pole positions. */
		std::complex<double> *positions;

		/** Pole amplitudes. */
		std::complex<double> *amplitudes;
	};

	/** Union of storage formats. */
	union Storage{
		ArrayFormat arrayFormat;
		PoleFormat poleFormat;
	};

	/** Actuall storage. */
	Storage storage;

	/** CPropertyExtractor is a friend class to allow it to write
	 *  GreensFunction data. */
	friend class TBTK::APropertyExtractor;

	/** CPropertyExtractor is a friend class to allow it to write
	 *  GreensFunction data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write
	 *  GreensFunction data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write DOS data. */
	friend class TBTK::FileReader;
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

inline unsigned int GreensFunction::getNumPoles() const{
	return storage.poleFormat.numPoles;
}

inline std::complex<double> GreensFunction::getPolePosition(
	unsigned int n
) const{
	return storage.poleFormat.positions[n];
}

inline std::complex<double> GreensFunction::getPoleAmplitude(
	unsigned int n
) const{
	return storage.poleFormat.amplitudes[n];
}

};	//End namespace Property
};	//End namespace TBTK

#endif
