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
 *  @file PropertyExtractor.h
 *  @brief Base class PropertyExtractors
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR

#include "Index.h"

#include <complex>

namespace TBTK{

/** The PropertyExtractor extracts is a base class for derived
 *  PropertyExtractors that are used to extract common physical properties such
 *  as DOS, Density, LDOS, etc. from a Solvers. */
class PropertyExtractor{
public:
	/** Constructor. */
	PropertyExtractor();

	/** Destructor. */
	~PropertyExtractor();
protected:
	/** Loops over range indices and calls the appropriate callback
	 *  function to calculate the correct quantity. */
	void calculate(
		void (*callback)(
			PropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		void *memory,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier
	);

	/** Hint used to pass information between calculate[Property] and
	 *  calculate[Property]Callback. */
	void *hint;

	/** Ensure that range indices are on compliant format. (Set range to
	 *  one for indices with non-negative pattern value.) */
	void ensureCompliantRanges(const Index &pattern, Index &ranges);

	/** Extract ranges for loop indices. */
	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *lDimensions,
		int **lRanges
	);
};

};	//End of namespace TBTK

#endif
