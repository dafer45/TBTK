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
 *  @file LDOS.h
 *  @brief Property container for local density of states (LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

namespace TBTK{
	class APropertyExtractor;
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for local density of states (LDOS). */
class LDOS{
public:
	/** Constructor. */
	LDOS(
		int dimensions,
		const int *ranges,
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Constructor. */
	LDOS(
		int dimensions,
		const int *ranges,
		double lowerBound,
		double upperBound,
		int resolution,
		const double *data
	);

	/** Destructor. */
	~LDOS();

	/** Get the dimension of the LDOS. (Excluding energy dimension). */
	int getDimensions() const;

	/** Get the ranges for the dimensions of the LDOS. */
	const int* getRanges() const;

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Get the number of data elements. */
	int getSize() const;

	/** Get LDOS data. */
	const double* getData() const;
private:
	/** Dimension of the LDOS. */
	int dimensions;

	/** Ranges for the dimensions of the LDOS. */
	int *ranges;

	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals). */
	int resolution;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	double *data;

	/** APropertyExtractor is a friend class to allow it to write LDOS
	 *  data. */
	friend class TBTK::APropertyExtractor;

	/** CPropertyExtractor is a friend class to allow it to write LDOS
	 *  data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write LDOS
	 * data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write LDOS data. */
	friend class TBTK::FileReader;
};

inline int LDOS::getDimensions() const{
	return dimensions;
}

inline const int* LDOS::getRanges() const{
	return ranges;
}

inline double LDOS::getLowerBound() const{
	return lowerBound;
}

inline double LDOS::getUpperBound() const{
	return upperBound;
}

inline int LDOS::getResolution() const{
	return resolution;
}

inline int LDOS::getSize() const{
	return size;
}

inline const double* LDOS::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
