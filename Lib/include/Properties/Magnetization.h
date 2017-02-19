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
 *  @file Magnetization.h
 *  @brief Property container for magnetization
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MAGNETIZATION
#define COM_DAFER45_TBTK_MAGNETIZATION

#include <complex>

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
	class EDPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for magnetization. */
class Magnetization{
public:
	/** Constructor. */
	Magnetization(int dimensions, const int* ranges);

	/** Constructor. */
	Magnetization(
		int dimensions,
		const int* ranges,
		const std::complex<double> *data
	);

	/** Destructor. */
	~Magnetization();

	/** Get the dimension of the magnetization. */
	int getDimensions() const;

	/** Get the ranges for the dimensions of the magnetization. */
	const int* getRanges() const;

	/** Get the number of data elements. */
	int getSize() const;

	/** Get magnetization data. */
	const std::complex<double>* getData() const;
private:
	/** Dimension of the magnetization. */
	int dimensions;

	/** Ranges for the dimensions of the magnetization. */
	int *ranges;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	std::complex<double> *data;

	/** CPropertyExtractor is a friend class to allow it to write
	 *  magnetiation data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write
	 *  magnetiation data. */
	friend class TBTK::DPropertyExtractor;

	/** EDPropertyExtractor is a friend class to allow it to write
	 *  magnetiation data. */
	friend class TBTK::EDPropertyExtractor;

	/** FileReader is a friend class to allow it to write magnetiation
	 *  data. */
	friend class TBTK::FileReader;
};

inline int Magnetization::getDimensions() const{
	return dimensions;
}

inline const int* Magnetization::getRanges() const{
	return ranges;
}

inline int Magnetization::getSize() const{
	return size;
}

inline const std::complex<double>* Magnetization::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
