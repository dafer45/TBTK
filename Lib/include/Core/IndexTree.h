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
 *  @file IndexTree.h
 *  @brief Data structure for mapping physical indices to a linear index
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX_TREE
#define COM_DAFER45_TBTK_INDEX_TREE

#include "Index.h"

#include <vector>

namespace TBTK{

class IndexTree{
public:
	/** Constructor. */
	IndexTree();

	/** Destructor. */
	~IndexTree();

	/** Add index. */
	void add(const Index &index);

	/** Generate linear map. */
	void generateLinearMap();

	/** Get linear index. */
	int getLinearIndex(const Index &index, bool ignoreWildcards = false) const;

	/** Get physical index. */
	Index getPhysicalIndex(int linearIndex) const;

	/** Get size. */
	int getSize() const;
private:
	/** Child nodes.*/
	std::vector<IndexTree> children;

	/** Flag indicating whether the given node corresponds to an index that
	 *  is included in the set. */
	bool indexIncluded;

	/** Flag indicating whether the given node corresponds to a wildcard
	 *  index. */
	bool wildcardIndex;

	/** Linear index. */
	int linearIndex;

	/** Size. Only used for top node. */
	int size;

	/** Add index. Is called by the public function IndexTree:add and is
	 *  called recursively.*/
	void add(const Index& index, unsigned int subindex);

	/** Generate linear map. Is called by the public IndexTree:add and is
	 *  called recursively. */
	int generateLinearMap(int i);

	/** Get linear index. Is called by the public IndexTree::getLinearIndex
	 *  and is called recursively. */
	int getLinearIndex(
		const Index &index,
		unsigned int subindex,
		bool ignoreWildcards
	) const;

	/** Get physical index. Is called by the public
	 *  IndexTree::getPhysicalIndex and is called recursively. */
	void getPhysicalIndex(
		int linearIndex,
		std::vector<int> *indices
	) const;

	/** Get minimum linear index of IndexTree. */
	int getMinIndex() const;

	/** Get maximum linear index of IndexTree. */
	int getMaxIndex() const;
};

inline IndexTree::getSize() const{
	return size;
}

}; //End of namesapce TBTK

#endif
