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
 *  @file DualIndex.h
 *  @brief Extends an Index with a continuous representation of the index.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DUAL_INDEX
#define COM_DAFER45_TBTK_DUAL_INDEX

#include "TBTK/Index.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

class DualIndex : public Index, public std::vector<double>{
public:
	/** Constructor. */
	DualIndex(
		const Index &index,
		const std::vector<double> &continuousIndex
	);

	/** Destructor. */
	~DualIndex();

	/** Get continuous index. */
//	const std::vector<double>& getContinuousIndex() const;
private:
//	const std::vector<double> continuousIndex;
};

inline DualIndex::DualIndex(
	const Index &index,
	const std::vector<double> &continuousIndex
) :
	Index(index),
//	continuousIndex(continuousIndex)
	std::vector<double>(continuousIndex)
{
	TBTKAssert(
		index.getSize() == continuousIndex.size(),
		"DualIndex::DualIndex()",
		"Incompatible index sizes.",
		"'index' and 'continuousIndex' must have the same number of"
		<< " components."
	);
}

inline DualIndex::~DualIndex(){
}

/*inline const std::vector<double>& DualIndex::getContinuousIndex() const{
	return continuousIndex;
}*/

};	//End of namespace TBTK

#endif
