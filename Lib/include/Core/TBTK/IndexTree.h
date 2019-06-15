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

	/** Comparison operator.
	 *
	 *  @param lhs The left hand side of the comparison operator.
	 *  @param rhs The right hand side of the comparison operator.
	 *
	 *  @param True if the two @link IndexTree IndexTrees @endlink describe
	 *  identical Index structures, otherwise false. */
	friend bool operator==(const IndexTree &lhs, const IndexTree &rhs);

	/** Inequality operator.
	 *
	 *  @param lhs The left hand side of the comparison operator.
	 *  @param rhs The right hand side of the comparison operator.
	 *
	 *  @param False if the two @link IndexTree IndexTrees @endlink describe
	 *  identical Index structures, otherwise true. */
	friend bool operator!=(const IndexTree &lhs, const IndexTree &rhs);

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

	/** Get whether the linear map has been generated.
	 *
	 *  @return True if the linear map has been generated. */
	bool getLinearMapIsGenerated() const;

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
	 *  <b>MatchWildcards:</b><br/>
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

	/** Get physical index corresponding to given linear index.
	 *
	 *  @param linearIndex Linear index.
	 *
	 *  @return The physical Index that corresponds to the given linear
	 *  Index. */
	Index getPhysicalIndex(int linearIndex) const;

	/** Check whether a given Index is contained in the IndexTree.
	 *
	 *  @param index
	 *
	 *  @return True if the IndexTree contains the given Index, otherwise
	 *  false. */
	bool contains(const Index &index);

	/** Get size.
	 *
	 *  @return The number of @link Index Indices @endlink stored in the
	 *  IndexTree. Returns -1 if IndexTree::generateLinearMap() has not yet
	 *  been called. */
	int getSize() const;

	/** First searches the IndexTree for a matching Index and then returns
	 *  a list of subindiex positions that matches the given subindex
	 *  value. If executed in SearchMode::StrictMatch, the function will
	 *  simply return the matching subindices in the input Index itself.
	 *  However, if executed in SearchMode::MatchWildcards, the IndexTree
	 *  will be searched for an Index that is matching up to a wildcard,
	 *  according to the specification of IndexTree::getLinearIndex(), and
	 *  then perform the subindex matching on the resulting Index.
	 *  <br/><br/>
	 *  <b>Example:</b> If {1, IDX_ALL, 3, IDX_SPIN, 5, IDX_ALL} is stored
	 *  in the IndexTree and the method is called as getSubindicesMatching(
	 *  IDX_ALL, {1, 2, 3, 4, 5, 6}, IndexTree::SearchMode::MatchWildcards)
	 *  then {1, IDX_ALL, 3, IDX_SPIN, 5, IDX_ALL} will first be found and
	 *  then matched against the subindex value IDX_ALL, which results in a
	 *  vector containing 1 and 5 being returned.
	 *
	 *  @param subindexValue The subindex value to match against.
	 *  @param index The Index to match against.
	 *  @param searchMode Mode to use when matching 'index' against the
	 *  @link Index Indices @endlink in the IndexTree.
	 *
	 *  @return An std::vector<int> containing the subindex positions for
	 *  which subindexValue has the same value as the matched Index. */
	std::vector<unsigned int> getSubindicesMatching(
		int subindexValue,
		const Index &index,
		SearchMode searchMode
	) const;

	/** Generate a list containing all the @link Index Indices @endlink in
	 *  the IndexTree that satisfies the specified pattern.
	 *
	 *  @param pattern Pattern to match against.
	 *
	 *  @return An std::vector<Index> containing all the @link Index
	 *  Indices @endlink in the IndexTree that matches the specified
	 *  pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

	/** Returns true if the two @link IndexTree IndexTrees @endlink contain
	 *  the same @link Index Indices @endlink.
	 *
	 *  @param indexTree IndexTree to compare to.
	 *
	 *  @return True if the two @link IndexTree IndexTrees @endlink contain
	 *  the same @link Index Indices @endlink. */
	bool equals(const IndexTree &indexTree) const;

//	class Iterator;
	class ConstIterator;
private:
	/** Base class used by Iterator and ConstIterator for iterating through
	 *  @link Index Indices @link stored in the tree structure. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointer to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const Index,
			Index
		>::type IndexType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		IndexType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs) const;

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs) const;

		/** Reset iterator. */
//		void reset();

		/** Advance the iterator by one. */
//		void searchNext();

		/** Get Index currently pointed at. */
//		const Index* getIndex() const;
//		const Index getIndex() const;

		/** Returns true if the iterator has reached the end. */
//		bool getHasReachedEnd() const;
	private:
		/** Typedef to allow for pointer to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const IndexTree*,
			IndexTree*
		>::type IndexTreePointerType;

		/** Root node to iterate from. */
		IndexTreePointerType indexTree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Flag indicating whether the next encountered Index should
		 *  be ignored. Is set to true at the start of a new search and
		 *  is set to false once the first Index has been encountered
		 *  to ensure that the search moves past the Index encountered
		 *  in the previous step. */
		bool skipNextIndex;

		/** Flag indicating that the iterator has reached the end. */
//		bool hasReachedEnd;

		/** Constructor. */
		_Iterator(IndexTreePointerType indexTree, bool end = false);

		/** Search after next Index. Is used by
		 *  IndexTree::_Iterator::searchNext() and is called
		 *  recursively. */
		bool searchNext(
			const IndexTree *indexTree,
			unsigned int subindex
		);

		/** Allow Iterator and ConstIterator to construct _Iterator. */
		friend class Iterator;
		friend class ConstIterator;
	};
public:
	/** ConstIterator for iterating through the elements stored in the
	 *  IndexTree. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const IndexTree *indexTree,
			bool end = false
		) : _Iterator<true>(indexTree, end){}

		/** Make the IndexTree able to construct a ConstIterator. */
		friend class IndexTree;
	};

	/** Get ConstIterator.
	 *
	 *  @return ConstIterator initialized to point at the first Index. */
	ConstIterator begin() const;

	/** Get ConstIterator.
	 *
	 *  @return ConstIterator initialized to point at the first Index. */
	ConstIterator cbegin() const;

	/** Create ConstIterator pointing to the end.
	 *
	 *  @return ConstIterator pointing to the end of the IndexTree. */
	ConstIterator end() const;

	/** Create ConstIterator pointing to the end.
	 *
	 *  @return ConstIterator pointing to the end of the IndexTree. */
	ConstIterator cend() const;

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

inline bool operator!=(const IndexTree &lhs, const IndexTree &rhs){
	return !(lhs == rhs);
}

inline bool IndexTree::getLinearMapIsGenerated() const{
	if(size == -1)
		return false;
	else
		return true;
}

inline IndexTree::ConstIterator IndexTree::begin() const{
	return ConstIterator(this);
}

inline IndexTree::ConstIterator IndexTree::cbegin() const{
	return ConstIterator(this);
}

inline IndexTree::ConstIterator IndexTree::end() const{
	return ConstIterator(this, true);
}

inline IndexTree::ConstIterator IndexTree::cend() const{
	return ConstIterator(this, true);
}

inline int IndexTree::getSize() const{
	return size;
}

/*inline bool IndexTree::Iterator::getHasReachedEnd() const{
	return hasReachedEnd;
}*/

template<bool isConstIterator>
IndexTree::_Iterator<isConstIterator>::_Iterator(IndexTreePointerType indexTree, bool end){
	if(end){
		this->indexTree = indexTree;
		currentIndex.push_back(indexTree->children.size());
	}
	else{
		this->indexTree = indexTree;
		currentIndex.push_back(0);
		skipNextIndex = false;
//		hasReachedEnd = false;
//		if(!searchNext(this->indexTree, 0))
//			hasReachedEnd = true;
		searchNext(this->indexTree, 0);
	}
}

/*void IndexTree::Iterator::reset(){
	currentIndex.clear();
	currentIndex.push_back(0);
	skipNextIndex = false;
	hasReachedEnd = false;
	if(!searchNext(indexTree, 0))
		hasReachedEnd = true;
}*/

/*void IndexTree::Iterator::searchNext(){
	skipNextIndex = true;
	if(!searchNext(indexTree, 0))
		hasReachedEnd = true;
}*/

template<bool isConstIterator>
void IndexTree::_Iterator<isConstIterator>::operator++(){
	skipNextIndex = true;
//	if(!searchNext(indexTree, 0))
//		hasReachedEnd = true;
	searchNext(indexTree, 0);
}

template<bool isConstIterator>
bool IndexTree::_Iterator<isConstIterator>::searchNext(
	const IndexTree *indexTree,
	unsigned int subindex
){
	if(indexTree->children.size() == 0){
		if(indexTree->indexIncluded){
			if(skipNextIndex)
				skipNextIndex = false;
			else
				return true;
		}
	}

	unsigned int  n = currentIndex.at(subindex);
	while(n < indexTree->children.size()){
		if(subindex+1 == currentIndex.size())
			currentIndex.push_back(0);
		if(searchNext(&indexTree->children.at(n), subindex+1))
			return true;

		currentIndex.pop_back();
		n = ++currentIndex.back();
	}

	return false;
}

//const Index* IndexTree::Iterator::getIndex() const{
/*const Index IndexTree::Iterator::getIndex() const{
//	if(currentIndex.at(0) == (int)indexTree->children.size())
//		return NULL;
	if(currentIndex.at(0) == (int)indexTree->children.size())
		return Index();

//	Index *index = new Index({});
//	Index *index = new Index();
	Index index;

	const IndexTree *indexTreeBranch = this->indexTree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
//		if(indexTreeBranch->indexSeparator)
//			index->push_back(IDX_SEPARATOR);
		if(indexTreeBranch->indexSeparator)
			index.push_back(IDX_SEPARATOR);

//		if(indexTreeBranch->wildcardIndex)
//			index->push_back(indexTreeBranch->wildcardType);
//		else
//			index->push_back(currentIndex.at(n));
		if(indexTreeBranch->wildcardIndex)
			index.push_back(indexTreeBranch->wildcardType);
		else
			index.push_back(currentIndex.at(n));

		if(n < currentIndex.size()-1)
			indexTreeBranch = &indexTreeBranch->children.at(currentIndex.at(n));
	}

	return index;
}*/

template<bool isConstIterator>
typename IndexTree::_Iterator<isConstIterator>::IndexType IndexTree::_Iterator<isConstIterator>::operator*(){
	if(currentIndex.at(0) == (int)indexTree->children.size()){
		TBTKExit(
			"IndexTree::_Iterator<isConstIterator>::operator*()",
			"Out of range access. Tried to access an element using"
			<< " an Iterator that points beyond the last element.",
			""
		);
	}

	Index index;

	const IndexTree *indexTreeBranch = this->indexTree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		if(indexTreeBranch->indexSeparator)
			index.push_back(IDX_SEPARATOR);

		if(indexTreeBranch->wildcardIndex)
			index.push_back(indexTreeBranch->wildcardType);
		else
			index.push_back(currentIndex.at(n));

		if(n < currentIndex.size()-1)
			indexTreeBranch = &indexTreeBranch->children.at(currentIndex.at(n));
	}

	return index;
}

template<bool isConstIterator>
bool IndexTree::_Iterator<isConstIterator>::operator==(const _Iterator &rhs) const{
	if(indexTree != rhs.indexTree)
		return false;

	if(currentIndex.size() != rhs.currentIndex.size())
		return false;

	for(unsigned int n = 0; n < currentIndex.size(); n++)
		if(currentIndex[n] != rhs.currentIndex[n])
			return false;

	return true;
};

template<bool isConstIterator>
bool IndexTree::_Iterator<isConstIterator>::operator!=(const _Iterator &rhs) const{
	return !operator==(rhs);
}

}; //End of namesapce TBTK

#endif

