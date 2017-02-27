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
 *  @file DOS.h
 *  @brief Property container for density of states (DOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS
#define COM_DAFER45_TBTK_DOS

namespace TBTK{
	class APropertyExtractor;
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for density of states (DOS). */
class DOS{
public:
	/** Constructor. */
	DOS(double lowerBound, double upperBound, int resolution);

	/** Constructor. */
	DOS(
		double lowerBound,
		double upperBound,
		int resolution,
		const double *data
	);

	/** Copy constructor. */
	DOS(const DOS &dos);

	/** Move constructor. */
	DOS(DOS &&dos);

	/** Destructor. */
	~DOS();

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Get DOS data. */
	const double* getData() const;

	/** Assignment operator. */
	DOS& operator=(const DOS &dos);

	/** Move assignment operator. */
	DOS& operator=(DOS &&dos);
private:
	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	int resolution;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write DOS data
	 */
	friend class TBTK::APropertyExtractor;

	/** CPropertyExtractor is a friend class to allow it to write DOS data
	 */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write DOS data
	 */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write DOS data. */
	friend class TBTK::FileReader;
};

inline double DOS::getLowerBound() const{
	return lowerBound;
}

inline double DOS::getUpperBound() const{
	return upperBound;
}

inline int DOS::getResolution() const{
	return resolution;
}

inline const double* DOS::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
