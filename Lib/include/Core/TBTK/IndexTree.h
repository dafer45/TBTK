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
 *  @brief Data structure for mapping physical indices to a linear index.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX_TREE
#define COM_DAFER45_TBTK_INDEX_TREE

#include "TBTK/Index.h"
#include "TBTK/Serializable.h"

#include <vector>

namespace TBTK{

/** @brief Data structure for mapping physical indices to linear indices. */
class IndexTree : public Serializable{
public:
	/** Constructors an IndexTree. */
	IndexTree();

	/** Constructor. Constructs the IndexTree from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the IndexTree.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	IndexTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~IndexTree();

	/** Add index to the IndexTree. The IndexTree allows for @link Index
	 *  Indices @enlink with both non-negative as well as negative
	 *  subindices to be added to the tree. For @link Index Indices
	 *  @endlink with non-negative subindices the IndexTree simply adds the
	 *  suplied @link Index Indices @endlink. Negative subindices with the
	 *  value IDX_SEPARATOR are similarly used to add @link Index Indices
	 *  @endlink consisting of multiple @link Index Indices @endlink. All
	 *  other negative subindices are interpreted as wildcards. What this
	 *  means is that the corresponding subindex will be marked as a
	 *  wildcard and the wildcard type (IDX_ALL, IDX_SPIN, etc) is saved in
	 *  the Index structure, which allows for @link Index Indices @endlink
	 *  with arbitrary values in the corresponding subindex to be searched
	 *  for using SearchMode::MatchWildrcards.
	 *  <br/><br/>
	 *  Any wildcard subindex will be replaced by zero and the full Index
	 *  then stored in the Index structure. This means that for example
	 *  {1, IDX_ALL, 3} will be stored as {1, 0, 3} in the IndexTree and
	 *  will count as one towards the size of the Index tree. {1, 0, 3} is
	 *  therefore also the Index that is returned when matching against
	 *  {1, IDX_ALL, 3}.
	 *  <br/><br/>
	 *  Also note that it is a fatal error to add two @link Index Indices
	 *  @endlink with similar Index structure but different wildcard type
	 *  to the IndexTree. For example, it is invalid to simultaneously add
	 *  {1, IDX_ALL, 3} and {1, IDX_SPIN, 4} to the IndexTree. However, it
	 *  is OK to add {1, IDX_ALL, 3} and {2, IDX_SPIN, 4} since the index
	 *  structure of the later pair is differentiated already at the first
	 *  subindex.
	 *
	 *  @param index Index to add. */
	void add(const Index &index);

	/** Generate a linear map from the @link Index Indices @endlink that
	 *  has been added to the non-negative numbers.
	 *  <br/><br/>
	 *  Note that @link Index Indices @endlink that have been added with a
	 *  wildcard subindex have the wildcards replaced by zero and the
	 *  resulting Index is treated as a normal Index. */
	void generateLinearMap();

	/** Enum class for selecting mode used to search indices with
	 *  getLinearIndex(). */
	enum class SearchMode{StrictMatch, /*IgnoreWildcards,*/ MatchWildcards};

	/** Get the linear index corresponding to the given physical Index. The
	 *  function can be executed in one of several different ways. First of
	 *  all the function can be executed in one of the following modes.
	 *  <br/><br/>
	 *  <b>StrictMatch:</b><br/>
	 *    A strict match between the suplied Index and the Index structure
	 *    ecoded in the IndexTree is required. An Index added with a
	 *    wildcard subindex must be retrieved with the same wildcard Index
	 *    specified in the same subindex position. This is the standard
	 *    mode of execution.
	 *  <br/>
	 *  <b>MAtchWildcards:</b><br/>
	 *    Subindices marked as wildcard indices in the IndexTree are
	 *    accepted independently of the subindex value in the Index passed
	 *    to this function. For example, if {1, IDX_ALL, 3} has been added
	 *    to the IndexTree, {1, 2, 3} will be accepted by the IndexTree as
	 *    a valid Index. If a wildcard type is specified in the Index
	 *    passed to this function, its type has to agree with that encoded
	 *    in the IndexTree.
	 *  <br/><br/>
	 *  Second, the flag returnNegativeForMissingIndex can be set to true
	 *  to make the function return -1 if an Index does not exists, or set
	 *  to false to make the function throw an IndexException.
	 *
	 *  @param index The Index to return the linear index for.
	 *  @param searchMode The mode to use when searching for Indices.
	 *  @param returnNegativeForMissingIndex If set to false, the request
	 *  for a missing Index will result in an IndexException being thrown,
	 *  but if set to true, a missing Index results in -1 being returned.
	 *
	 *  @return The linear Index for the given physical Index. Returns -1
	 *  if the Index is not found and returnNegativeForMissingIndex is set
	 *  to true.
	 *
	 *  @throws IndexException If the requested Index does not exist and
	 *  returnNegativeForMissingIndex is set to false. */
	int getLinearIndex(
		const Index &index,
		SearchMode searchMode = SearchMode::StrictMatch,
		bool returnNegativeForMissingIndex = false
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

	/** Generate a list of indices satisfying the specified pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

	/** Iterator for iterating through @link Index Indices @link stored in
	 *  the tree structure. */
	class Iterator{
	public:
		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNext();

		/** Get Index currently pointed at. */
//		const Index* getIndex() const;
		const Index getIndex() const;

		/** Returns true if the iterator has reached the end. */
		bool getHasReachedEnd() const;
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

		/** Flag indicating that the iterator has reached the end. */
		bool hasReachedEnd;

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

	/** Implements Serializable::serialize(). */
	std::string serialize(Mode mode) const;
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

	/** Flag indicating whether the given node is an Index-separator. I.e.,
	 *  wheter the next node level corresponds to the first subindex of a
	 *  new Index. */
	bool indexSeparator;

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
		SearchMode searchMode,
		bool returnNegativeForMissingIndex
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

inline bool IndexTree::Iterator::getHasReachedEnd() const{
	return hasReachedEnd;
}

}; //End of namesapce TBTK

#endif
