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
 *  @file HoppingAmplitudeTree.h
 *  @brief Node in tree used by HoppingAmplitudeSet to store @link
 *    HoppingAmplitude HoppingAmplitudes @endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TREE_NODE
#define COM_DAFER45_TBTK_TREE_NODE

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/IndexTree.h"
#include "TBTK/Serializable.h"

#include <vector>

namespace TBTK{

/** @brief Node in tree used by HoppingAmplitudeSet to store @link
 *    HoppingAmplitude HoppingAmplitudes @endlink.
 *
 *  HoppingAmplitudeTree is a tree structure used to build a tree for stroing
 *  @link HoppingAmplitude HoppingAmplitudes @endlink. Used by AmplitudeSet.*/
class HoppingAmplitudeTree : virtual public Serializable{
public:
	/** Constructs a HoppingAmplitudeTree. */
	HoppingAmplitudeTree();

	/** Constructs a HoppingAmplitudeTree with a preallocated tree
	 *  structure such that the addition of HoppingAmplitudes with indices
	 *  that have the same subindex structure as 'capacity', but with
	 *  smaller subindices will not cause reallocation for the main tree
	 *  stucture. Internal containers for HoppingAmplitudes may still be
	 *  reallocated.
	 *
	 *  @param capacity The 'Index capacity'. */
	HoppingAmplitudeTree(
		const std::vector<unsigned int> &capacity
	);

	/** Constructor. Constructs the HoppingAmplitudeTree from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Index.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	HoppingAmplitudeTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~HoppingAmplitudeTree();

	/** Add a HoppingAmplitude.
	 *
	 *  @param ha HoppingAmplitude to add. */
	void add(HoppingAmplitude ha);

	/** Get basis size.
	 *
	 *  @return The basis size if the basis has been generated using the
	 *  call to HoppingAmplitudeTree::generateBasisSize(), otherwise -1. */
	int getBasisSize() const;

	/** Get HoppingAmplitudeTree containing the @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink of the specified subspace. If the
	 *  original HoppingAMplitudeTree has an Index structure with
	 *  {subspace, intra subspace indices}, then the new
	 *  HoppingAmplitudeTree has the Index-structure
	 *  {intra subspace indices}. This function does not guarantee to
	 *  return a closed subspace and can contain HoppingAmplitudes to
	 *  components with other subspace indices if the specified subspace is
	 *  not a proper subspace. If in doubt, use
	 *  HoppingAmplitudeTree::isProperSubspace() to check whether a given
	 *  subspace is a proper subspace before calling this function. Empty
	 *  subspaces for which no HoppingAmplitudes have been added return
	 *  empty subspaces.
	 *
	 *  @param subspace A number of subindices that when used as leftmost
	 *  subindices in an Index specifies a subspace.
	 *
	 *  @return A (possibly improper) subspace of the HoppingAmplitudeTree. */
	HoppingAmplitudeTree* getSubTree(const Index &subspace);

	/** Get HoppingAmplitudeTree containing the @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink of the specified subspace. If the
	 *  original HoppingAMplitudeTree has an Index structure with
	 *  {subspace, intra subspace indices}, then the new
	 *  HoppingAmplitudeTree has the Index-structure
	 *  {intra subspace indices}. This function does not guarantee to
	 *  return a closed subspace and can contain HoppingAmplitudes to
	 *  components with other subspace indices if the specified subspace is
	 *  not a proper subspace. If in doubt, use
	 *  HoppingAmplitudeTree::isProperSubspace() to check whether a given
	 *  subspace is a proper subspace before calling this function. Empty
	 *  subspaces for which no HoppingAmplitudes have been added return
	 *  empty subspaces.
	 *
	 *  @param subspace A number of subindices that when used as leftmost
	 *  subindices in an Index specifies a subspace.
	 *
	 *  @return A (possibly improper) subspace of the HoppingAmplitudeTree. */
	const HoppingAmplitudeTree* getSubTree(const Index &subspace) const;

	/** Returns true if the subspace is a proper subspace. That is, if the
	 *  corresponding subtree only contains HoppingAmplitudes that connects
	 *  states within the subtree (see exception below). Empty subspaces
	 *  for which no @link HoppingAmplitude HoppingAmplitudes @endlink have
	 *  been added are considered proper subspaces.<br>
	 *  <br>
	 *  %Exception: For a subspace to be considered a proper subspace, each
	 *  of it subindices needs to completely split the problem into
	 *  independent parts. For example, for the set of @link
	 *  HoppingAmplitude HoppingAmplitudes @endlink below, {0, 0, 0} is a
	 *  proper subspace in the sense that it contains no @link
	 *  HoppingAmplitude HoppingAmplitudes @endlink to other subspaces.
	 *  However, the second subindex does not split the problem completely
	 *  since {0, 0, 1} and {0, 0, 2} belong to the same subspace. In this
	 *  case {0, 0} therefore is the most specific proper subspace.<br>
	 *  HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0});<br>
	 *  HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2});<br>
	 *  HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1});
	 *
	 *  @param subspace A number of subindices that when used as leftmost
	 *  subindices in an Index specifies a subspace.
	 *
	 *  @return True if the subspace is a proper subspace according to the
	 *  definition above, otherwise false. */
	bool isProperSubspace(const Index &subspace) const;

	/** Returns an IndexTree containing all proper subspace indices.
	 *
	 *  @return An IndexTree containing all proper subspace indices. */
	IndexTree getSubspaceIndices() const;

	/** Returns the subspace part of an Index.
	 *
	 *  @param index An Index.
	 *
	 *  @return A new Index containing only the subspace Index of index. */
	Index getSubspaceIndex(const Index &index) const;

	/** Get first index in subspace.
	 *
	 *  @param subspaceIndex The physical Index of the subspace.
	 *
	 *  @return The first Hilbert space index in the given subspace. If the
	 *  subspace is empty, -1 is returned. */
	int getFirstIndexInSubspace(const Index &subspaceIndex) const;

	/** Get last index in subspace.
	 *
	 *  @param subspaceIndex The physical Index of the subspace.
	 *
	 *  @return The last Hilbert space index in the given subspace. If the
	 *  subspace is empty, -1 is returned. */
	int getLastIndexInSubspace(const Index &subspaceIndex) const;

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 *  given 'from'-index.
	 *
	 *  @param index From-Index.
	 *
	 *  @return All @link HoppingAmplitude HoppingAmplitudes @endlink with
	 *  the given from-Index. */
	const std::vector<HoppingAmplitude>& getHoppingAmplitudes(
		Index index
	) const;

	/** Get Hilbert space basis index for given physical index.
	 *
	 *  @param index Physical Index for which to obtain the Hilbert space
	 *  index.
	 *
	 *  @return The Hilbert space index corresponding to the given Physical
	 *  Index. Returns -1 if HoppingAmplitudeTree::generateBasisIndices()
	 *  has not been called. */
	int getBasisIndex(const Index &index) const;

	/** Get physical Index for given Hilbert space basis index.
	 *
	 *  @param basisIndex Hilbert space index for which to obtain the
	 *  physical Index.
	 *
	 *  @return The physical Index corresponding to the given Hilbert space
	 *  index. */
	Index getPhysicalIndex(int basisIndex) const;

	/**  Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

	/** Generate a list containing the indices in the HoppingAmplitudeTree
	 *  that satisfies the specified patterns. The indices are ordered in
	 *  terms of rising Hilbert space indices.
	 *
	 *  @param patterns Patterns to match against. IDX_ALL can be used as a
	 *  wildcard.
	 *
	 *  @return A list of physical indices that match the specified
	 *  patterns. */
	std::vector<Index> getIndexList(const std::vector<Index> &patterns) const;

	/** Sort HoppingAmplitudes in row order. */
	void sort(HoppingAmplitudeTree *rootNode);

	/** Print @link HoppingAmplitude HoppingAmplitudes @endlink. Mainly for
	 *  debuging purposes. */
	void print();

	class Iterator;
	class ConstIterator;
private:
	/** Base class Iterator and ConstIterator for iterating through the
	 *  @link HoppingAmplitude HoppingAmplitudes @endlink stored in the
	 *  tree structure. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const HoppingAmplitude&,
			HoppingAmplitude&
		>::type HoppingAmplitudeReferenceType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		HoppingAmplitudeReferenceType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs) const;

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs) const;

		/** Get minimum basis index. */
		int getMinBasisIndex() const;

		/** Get maximum basis index. */
		int getMaxBasisIndex() const;

		/** Get number of basis indices. */
		int getNumBasisIndices() const;
	private:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator tpye. */
		typedef typename std::conditional<
			isConstIterator,
			const HoppingAmplitudeTree*,
			HoppingAmplitudeTree*
		>::type HoppingAmplitudeTreePointerType;

		/** Root node to iterate from. */
		HoppingAmplitudeTreePointerType tree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Current HoppingAmplitude that the iterator points at at the
		 *  currentIndex. */
		int currentHoppingAmplitude;

		/** Give Iterator and ConstIterator access to the constructor. */
		friend class Iterator;
		friend class ConstIterator;

		/** Constructor. */
		_Iterator(HoppingAmplitudeTreePointerType tree, bool end = false);

		/** Search after next HoppingAmplitude. Is used by
		 *  HoppingAmplitudeTree::Iterator::searchNext and called
		 *  recursively. */
		bool searchNext(
			HoppingAmplitudeTreePointerType hoppingAmplitudeTree,
			int subindex
		);
	};
public:
	/** Iterator for iterating through the @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink stored in the HoppingAmplitudeTree. */
	class Iterator : public _Iterator<false>{
	private:
		Iterator(
			HoppingAmplitudeTree *hoppingAmplitudeTree,
			bool end = false
		) : _Iterator<false>(hoppingAmplitudeTree, end){};

		/** Make the HoppingAmplitudeTree able to construct an
		 *  Iterator. */
		friend class HoppingAmplitudeTree;
	};

	/** Iterator for iterating through the @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink stored in the HoppingAmplitudeTree. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const HoppingAmplitudeTree *hoppingAmplitudeTree,
			bool end = false
		) : _Iterator<true>(hoppingAmplitudeTree, end){};

		/** Make the HoppingAmplitudeTree able to construct a
		 *  ConstIterator. */
		friend class HoppingAmplitudeTree;
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing to the first element in the
	 *  HoppingAmplitudeTree. */
	Iterator begin();

	/** Create ConstIterator.
	 *
	 *  @return ConstIterator pointing to the first element in the
	 *  HoppingAmplitudeTree. */
	ConstIterator begin() const;

	/** Create ConstIterator.
	 *
	 *  @return ConstIterator pointing to the first element in the
	 *  HoppingAmplitudeTree. */
	ConstIterator cbegin() const;

	/** Create Iterator pointing to the end.
	 *
	 *  @return Iterator pointing to the end of the HoppingAmplitudeTree.
	 */
	Iterator end();

	/** Create ConstIterator pointing to the end.
	 *
	 *  @return ConstIterator pointing to the end of the
	 *  HoppingAmplitudeTree. */
	ConstIterator end() const;

	/** Create ConstIterator pointing to the end.
	 *
	 *  @return ConstIterator pointing to the end of the
	 *  HoppingAmplitudeTree. */
	ConstIterator cend() const;

	/** Implements Serializable::serialize.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string represenation of the Index. */
	virtual std::string serialize(Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the Index. */
	unsigned int getSizeInBytes() const;
private:
	/** Basis index for the Hamiltonian. */
	int basisIndex;

	/** Basis size of Hamiltonian. */
	int basisSize;

	/** Flag indicating whether all HoppingAmplitudes passed to this nodes
	 *  child nodes have the same 'to' and 'from' subindex in the position
	 *  corresponding this node level. Is set to true when the node is
	 *  constructed, and is set to false as soon as a HoppingAmplitude with
	 *  differing 'to' and 'from' subindices is encountered.
	 *
	 *  If a node and all its parents has isPotentialBlockSeparator set to
	 *  true, the child nodes correspond to independent blocks.
	 *
	 *  A node that has isPotentialBlockSeparator set to true, but for
	 *  which at least one parent has it set to false, may or may not be a
	 *  block separating node depending on the rest of the tree. It is in
	 *  such cases possible that a reordering of the subindices can lead to
	 *  a better exploitation of the block structure.
	 *
	 *  For leaf nodes the value is irrelevant. */
	bool isPotentialBlockSeparator;

	/** HoppingAmplitudes stored on this node, should only be non-empty if
	 *  the node is a leaf node. That is, if the node corresponds to a last
	 *   subindex index. */
	std::vector<HoppingAmplitude> hoppingAmplitudes;

	/** Child nodes. Never non-empty at the same time as hoppingAmplitudes
	*/
	std::vector<HoppingAmplitudeTree> children;

	/** Empty tree that is returned by HoppingAMplitudeTree::getSubTree
	 *  when a non-existing subspace is requested. */
	static const HoppingAmplitudeTree emptyTree;

	/** Add HoppingAmplitude. Is called by the public
	 *  HoppingAmplitudeTree::add and is called recursively. */
	void _add(HoppingAmplitude &ha, unsigned int subindex);

	/** Get sub tree. Is called by HoppingAmplitudeTree::getSubTree and is
	 *  called recursively. */
	const HoppingAmplitudeTree* getSubTree(
		const Index &subspace,
		unsigned int subindex
	) const;

	/** Returns true if the subspace is a proper subsapce. Is called by
	 *  HoppingAmplitudeTree::isProperSubspace and is called recursively.
	 */
	bool _isProperSubspace(
		const Index &subspace,
		unsigned int subindex
	) const;

	/** Adds indices of all separate blocks below the current node to the
	 *  indexTree. Is calleed by HoppingAmplitudeTree::getBlockIndices()
	 *  and is called recuresively. */
	void getBlockIndices(IndexTree &blockIndices, Index index) const;

	/** Returns the block part of an Index. Is called by
	 *  HoppingAmplitudeTree:getSubspaceIndex() and is called recuresively.
	 */
	void getBlockIndex(
		const Index &index,
		unsigned int subindex,
		Index &blockIndex
	) const;

	/** Get HoppingAmpilitudes. Is called by the public
	 *  HoppingAmplitudeTree::getHoppingAmplitudes and is called
	 *  recursively. */
	const std::vector<HoppingAmplitude>& _getHoppingAmplitudes(
		Index index,
		unsigned int subindex
	) const;

	/** Get Hilbert space index for given physical index. Is called by the
	 *  public HoppingAmplitudeTree::getBasisIndex and is called
	 *  recursively. */
	int _getBasisIndex(const Index &index, unsigned int subindex) const;

	/** Get physical index for given Hilbert space index. Is called by the
	 *  public HoppingAmplitudeTreee::getPhysicalIndex and is called
	 *  recursively. */
	void _getPhysicalIndex(int basisIndex, std::vector<Subindex> &indices) const;

	/** Get minimum index on HoppingAmplitudeTree. */
	int getMinIndex() const;

	/** Get max index on HoppingAmplitudeTree. */
	int getMaxIndex() const;

	/** Generate Hilbert space indices. Is called by the public
	 *  HoppingAmplitudeTree::generateBasisIndices and is called
	 *  recursively. */
	int generateBasisIndices(int i);

	/** Generate a list containing the indices in the HoppingAmplitudeTree
	 *  that satisfies the specified pattern. Is called by
	 *  HoppingAmplitudeTree::getIndexList. */
	std::vector<Index> _getIndexList(const Index &pattern) const;

	/** Print HoppingAmplitudes. Is called by the public
	 *  HoppingAmplitudeTree::print and is called recursively. Mainly for
	 *  debuging purposes. */
	void print(unsigned int subindex);

	/** Returns (depth) first HoppingAmplitude as an example, in case of
	 *  error while adding HoppingAmplitudes to the tree. */
	HoppingAmplitude getFirstHA() const;
};

inline int HoppingAmplitudeTree::getBasisSize() const{
	return basisSize;
}

inline int HoppingAmplitudeTree::getFirstIndexInSubspace(
	const Index &subspaceIndex
) const{
	TBTKAssert(
		isProperSubspace(subspaceIndex),
		"HoppingAmplitudeTree::getFirstIndexInSubspace()",
		"The index " << subspaceIndex.toString() << " is not an Index"
		<< " of a proper subspace.",
		""
	);

	const HoppingAmplitudeTree *subspace = getSubTree(subspaceIndex);

	return subspace->getMinIndex();
}

inline int HoppingAmplitudeTree::getLastIndexInSubspace(
	const Index &subspaceIndex
) const{
	TBTKAssert(
		isProperSubspace(subspaceIndex),
		"HoppingAmplitudeTree::getLastIndexInSubspace()",
		"The index " << subspaceIndex.toString() << " is not an Index"
		<< " of a proper subspace.",
		""
	);

	const HoppingAmplitudeTree *subspace = getSubTree(subspaceIndex);

	return subspace->getMaxIndex();
}

inline HoppingAmplitudeTree::Iterator HoppingAmplitudeTree::begin(){
	return Iterator(this);
}

inline HoppingAmplitudeTree::ConstIterator HoppingAmplitudeTree::begin() const{
	return ConstIterator(this);
}

inline HoppingAmplitudeTree::ConstIterator HoppingAmplitudeTree::cbegin() const{
	return ConstIterator(this);
}

inline HoppingAmplitudeTree::Iterator HoppingAmplitudeTree::end(){
	return Iterator(this, true);
}

inline HoppingAmplitudeTree::ConstIterator HoppingAmplitudeTree::end() const{
	return ConstIterator(this, true);
}

inline HoppingAmplitudeTree::ConstIterator HoppingAmplitudeTree::cend() const{
	return ConstIterator(this, true);
}

inline unsigned int HoppingAmplitudeTree::getSizeInBytes() const{
	unsigned int size = 0;
	for(unsigned int n = 0; n < hoppingAmplitudes.size(); n++)
		size += hoppingAmplitudes[n].getSizeInBytes();
	for(unsigned int n = 0; n < children.size(); n++)
		size += children[n].getSizeInBytes();

	size += (
		hoppingAmplitudes.capacity() - hoppingAmplitudes.size()
	)*sizeof(HoppingAmplitude);

	size += (
		children.capacity() - children.size()
	)*sizeof(HoppingAmplitudeTree);

	return size + sizeof(HoppingAmplitudeTree);
}

template<bool isConstIterator>
HoppingAmplitudeTree::_Iterator<isConstIterator>::_Iterator(
	HoppingAmplitudeTreePointerType tree,
	bool end
)
{
	if(end){
		this->tree = tree;
		if(tree->children.size() == 0){
			currentHoppingAmplitude = -1;
		}
		else{
			currentIndex.push_back(tree->children.size());
			currentHoppingAmplitude = -1;
		}
	}
	else{
		this->tree = tree;
		if(tree->children.size() == 0){
			//Handle the special case when the data is stored on the head
			//node. Can for example be the case when iterating over a
			//single leaf node.
			currentHoppingAmplitude = -1;
			searchNext(tree, -1);
		}
		else{
			currentIndex.push_back(0);
			currentHoppingAmplitude = -1;
			searchNext(tree, 0);
		}
	}
}

template<bool isConstIterator>
void HoppingAmplitudeTree::_Iterator<isConstIterator>::operator++(){
	if(tree->children.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		searchNext(tree, -1);
	}
	else{
		searchNext(tree, 0);
	}
}

template<bool isConstIterator>
bool HoppingAmplitudeTree::_Iterator<isConstIterator>::searchNext(
	HoppingAmplitudeTreePointerType hoppingAmplitudeTree,
	int subindex
){
	if(subindex+1 == (int)currentIndex.size()){
		//If the node level corresponding to the current index is
		//reached, try to execute leaf node actions.

		if(currentHoppingAmplitude != -1){
			//The iterator is in the process of iterating over
			//HoppingAmplitudes on this leaf node. Try to iterate further.

			currentHoppingAmplitude++;
			if(currentHoppingAmplitude == (int)hoppingAmplitudeTree->hoppingAmplitudes.size()){
				//Last HoppingAmplitude already reached. Reset
				//currentHoppingAmplitude and return false to
				//indicate that no more HoppingAMplitudes exist
				//on this node.
				currentHoppingAmplitude = -1;
				return false;
			}
			else{
				//Return true to indicate that the next
				//HoppingAmplitude succesfully has been found.
				return true;
			}
		}

		//We are here guaranteed that the iterator is not currently in
		//a state where it is iterating over HoppingAmplitudes on this
		//node.

		if(hoppingAmplitudeTree->children.size() == 0){
			//The node has no children and is therefore either a
			//leaf node with HoppingAmplitudes stored on it, or an
			//empty dummy node.

			if(hoppingAmplitudeTree->hoppingAmplitudes.size() != 0){
				//There are HoppingAMplitudes on this node,
				//initialize the iterator to start iterating
				//over these. Return true to indicate that a
				//HoppingAmplitude was found.
				currentHoppingAmplitude = 0;
				return true;
			}
			else{
				//The node is an empty dymmy node. Return false
				//to indicate that no more HoppingAmplitudes
				//exist on this node.
				return false;
			}
		}
	}

	//We are here guaranteed that this is not a leaf or dummy node. We know
	//this because either the tests inside the previous if-statements
	//failed, or we are iterating through children that already have been
	//visited on an earlier call to searchNext if the outer if-statement
	//itself failed.

	//Perform depth first search for the next HoppingAmplitude. Starts from
	//the child node reffered to by currentIndex.
	unsigned int n = currentIndex.at(subindex);
	while(n < hoppingAmplitudeTree->children.size()){
		if(subindex+1 == (int)currentIndex.size()){
			//The deepest point visited so far on this branch has
			//been reached. Initialize the depth first search for
			//child n to start from child n's zeroth child.
			currentIndex.push_back(0);
		}
		if(searchNext(&hoppingAmplitudeTree->children.at(n), subindex+1)){
			//Depth first search on child n succeded at finding a
			//HoppingAmplitude. Return true to indicate success.
			return true;
		}
		//Child n does not have any more HoppingAmplitudes. Pop
		//the subindex corresponding to child n's node level and
		//increment the subindex corresponding to this node level to
		//prepare for depth first search of child n+1.
		currentIndex.pop_back();
		n = ++currentIndex.back();
	}

	//Return false to indicate that no more HoppingAmplitudes could be
	//found on this node.
	return false;
}

template<bool isConstIterator>
typename HoppingAmplitudeTree::_Iterator<
	isConstIterator
>::HoppingAmplitudeReferenceType HoppingAmplitudeTree::_Iterator<
	isConstIterator
>::operator*(){
	if(currentIndex.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		if(currentHoppingAmplitude == -1){
			TBTKExit(
				"HoppingAmplitudeTree::_Iterator::operator*()",
				"Out of range access. Tried to access an"
				<< " element using an iterator that points"
				<< " beyond the last element.",
				""
			);
		}
		else{
			return tree->hoppingAmplitudes.at(currentHoppingAmplitude);
		}
	}

	if(currentIndex.at(0) == (int)tree->children.size()){
		TBTKExit(
			"HoppingAmplitudeTree::_Iterator::operator*()",
			"Out of range access. Tried to access an"
			<< " element using an iterator that points"
			<< " beyond the last element.",
			""
		);
	}
	const HoppingAmplitudeTree *tn = this->tree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		tn = &tn->children.at(currentIndex.at(n));
	}

	return tn->hoppingAmplitudes.at(currentHoppingAmplitude);
}

template<bool isConstIterator>
bool HoppingAmplitudeTree::_Iterator<isConstIterator>::operator==(const _Iterator &rhs) const{
	if(
		this->tree == rhs.tree
		&& currentIndex.size() == rhs.currentIndex.size()
	){
		for(unsigned int n = 0; n < currentIndex.size(); n++){
			if(currentIndex[n] != rhs.currentIndex[n])
				return false;
		}

		if(currentHoppingAmplitude == rhs.currentHoppingAmplitude)
			return true;
		else
			return false;
	}
	else{
		return false;
	}
}

template<bool isConstIterator>
bool HoppingAmplitudeTree::_Iterator<isConstIterator>::operator!=(
	const _Iterator &rhs
) const{
	return !operator==(rhs);
}

template<bool isConstIterator>
int HoppingAmplitudeTree::_Iterator<isConstIterator>::getMinBasisIndex() const{
	return tree->getMinIndex();
}

template<bool isConstIterator>
int HoppingAmplitudeTree::_Iterator<isConstIterator>::getMaxBasisIndex() const{
	return tree->getMaxIndex();
}

template<bool isConstIterator>
int HoppingAmplitudeTree::_Iterator<isConstIterator>::getNumBasisIndices(
) const{
	if(getMaxBasisIndex() == -1)
		return 0;
	else
		return 1 + getMaxBasisIndex() - getMinBasisIndex();
}

};	//End of namespace TBTK

#endif
