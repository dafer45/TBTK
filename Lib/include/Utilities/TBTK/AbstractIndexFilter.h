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
 *  @file AbstractIndexFilter.h
 *  @brief Abstract Index filter.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_INDEX_FILTER
#define COM_DAFER45_TBTK_ABSTRACT_INDEX_FILTER

#include "TBTK/Index.h"

namespace TBTK{

class AbstractIndexFilter{
public:
	/** Constructor. */
	AbstractIndexFilter();

	/** Destructor. */
	virtual ~AbstractIndexFilter();

	/** Clone. */
	virtual AbstractIndexFilter* clone() const = 0;

	/** Returns true if the filter includes the Index. */
	virtual bool isIncluded(
		const Index &index
	) const = 0;
private:
};

inline AbstractIndexFilter::AbstractIndexFilter(){
}

inline AbstractIndexFilter::~AbstractIndexFilter(){
}

}; //End of namesapce TBTK

#endif
