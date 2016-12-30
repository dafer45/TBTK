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
 *  @file EigenValues.h
 *  @brief Property container for eigen values
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EIGEN_VALUES
#define COM_DAFER45_TBTK_EIGEN_VALUES

namespace TBTK{
	class APropertyExtractor;
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for local density of states (LDOS). */
class EigenValues{
public:
	/** Constructor. */
	EigenValues(int size);

	/** Constructor. */
	EigenValues(int size, const double *data);

	/** Destructor. */
	~EigenValues();

	/** Get number of eigen values. */
	int getSize() const;

	/** Get eigen values. */
	const double* getData() const;
private:
	/** Number of elements in data. */
	int size;

	/** Actual data. */
	double *data;

	/** APropertyExtractor is a friend class to allow it to write
	 * EigenValues data. */
	friend class TBTK::APropertyExtractor;

	/** CPropertyExtractor is a friend class to allow it to write
	 * EigenValues data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write
	 *  EigenValues data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write
	 *  EigenValues data. */
	friend class TBTK::FileReader;
};

inline int EigenValues::getSize() const{
	return size;
}

inline const double* EigenValues::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
