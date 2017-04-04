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

	/** Enum class for selecting mode used to search indices with
	 *  getLinearIndex(). */
	enum class SearchMode{StrictMatch, IgnoreWildcards, MatchWildcards};

	/** Get linear index. */
	int getLinearIndex(
		const Index &index,
		SearchMode searchMode = SearchMode::StrictMatch
//		bool ignoreWildcards = false
	) const;

	/** Get physical index. */
	Index getPhysicalIndex(int linearIndex) const;

	/** Get size. */
	int getSize() const;

	/** Get subindex. */
	std::vector<unsigned int> getSubindicesMatching(
		int i,
		const Index &index,
		SearchMode searchMode
	) const;

	/** Iterator for iterating through @link Index Indices @link stored in
	 *  the tree structure. */
	class Iterator{
	public:
		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNext();

		/** Get Index currently pointed at. */
		const Index* getIndex() const;
	private:
		/** Root node to iterate from. */
		const IndexTree *indexTree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Flag indicating whether the next encountered Index should
		 *  be ignored. Is set to true at the start of a new search and
		 *  is set to false once the first Index has been encountered
		 *  to ensure that the search moves past the Index encountered
		 *  in the previous step. */
		bool skipNextIndex;

		/** Constructor. */
		Iterator(const IndexTree *indexTree);

		/** Search after next Index. Is used by
		 *  IndexTree::Iterator::searchNext() and is called
		 *  recursively. */
		bool searchNext(
			const IndexTree *indexTree,
			unsigned int subindex
		);

		/** Allow IndexTree to construct Iterator. */
		friend class IndexTree;
	};

	/** Returns Iterator initialized to point at first Index. */
	Iterator begin() const;
private:
	/** Child nodes.*/
	std::vector<IndexTree> children;

	/** Flag indicating whether the given node corresponds to an index that
	 *  is included in the set. */
	bool indexIncluded;

	/** Flag indicating whether the given node corresponds to a wildcard
	 *  index. */
	bool wildcardIndex;

	/** Flag indicating the wildcard type (IDX_ALL, IDX_SUM_ALL, ...) of if
	 *  wildcradIndex = true. */
	int wildcardType;

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
		SearchMode searchMode
//		bool ignoreWildcards
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

inline int IndexTree::getSize() const{
	return size;
}

}; //End of namesapce TBTK

#endif
