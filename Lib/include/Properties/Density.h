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
 *  @file Density.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DENSITY
#define COM_DAFER45_TBTK_DENSITY

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for density. */
class Density{
public:
	/** Constructor. */
	Density(int dimensions, const int *ranges);

	/** Constructor. */
	Density(int dimensions, const int *ranges, const double *data);

	/** Destructor. */
	~Density();

	/** Get the dimension of the density. */
	int getDimensions() const;

	/** Get the ranges for the dimensions of the density. */
	const int* getRanges() const;

	/** Get number of data elements. */
	int getSize() const;

	/** Get density data. */
	const double* getData() const;
private:
	/** Dimension of the density. */
	int dimensions;

	/** Ranges for the dimensions of the density. */
	int *ranges;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write density
	 *  data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write density
	 *  data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write density data. */
	friend class TBTK::FileReader;
};

inline int Density::getDimensions() const{
	return dimensions;
}

inline const int* Density::getRanges() const{
	return ranges;
}

inline int Density::getSize() const{
	return size;
}

inline const double* Density::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
