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
#include "TBTK/Serializeable.h"

#include <vector>

namespace TBTK{

/** @brief Node in tree used by HoppingAmplitudeSet to store @link
 *    HoppingAmplitude HoppingAmplitudes @endlink.
 *
 *  HoppingAmplitudeTree is a tree structure used to build a tree for stroing
 *  @link HoppingAmplitude HoppingAmplitudes @endlink. Used by AmplitudeSet.*/
class HoppingAmplitudeTree : public Serializeable{
public:
	/** Constructs a HoppingAmplitudeTree. */
	HoppingAmplitudeTree();

	/** Constructs a HoppingAmplitude with a preallocated tree structure
	 *  such that the addition of HoppingAmplitudes with indices that have
	 *  the same subindex structure as 'capacity', but with smaller
	 *  subindices will not cause reallocation for the main tree stucture.
	 *  Internal containers for HoppingAmplitudes may still be reallocated.
	 *
	 *  @param The 'Index capacity'. */
	HoppingAmplitudeTree(
		const std::vector<unsigned int> &capacity
	);

	/** Constructor. Constructs the HoppingAmplitudeTree from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Index
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
	const std::vector<HoppingAmplitude>* getHAs(Index index) const;

	/** Get Hilbert space basis index for given physical index.
	 *
	 *  @param index Physical Index for which to obtain the Hilbert space
	 *  index.
	 *
	 *  @return The Hilbert space index corresponding to the given Physical
	 *  Index. Returns -1 if HoppingAmplitudeTree::generateBasisIndices()
	 *  has not been called. */
	int getBasisIndex(const Index &index) const;

	/** Get physical index for given Hilbert space basis index.
	 *
	 *  @param basisIndex Hilbert space index for which to obtain the
	 *  physical Index.
	 *
	 *  @return The physical index corresponding to the given Hilbert space
	 *  index. */
	Index getPhysicalIndex(int basisIndex) const;

	/**  Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

	/** Generate a list containing the indices in the HoppingAmplitudeTree
	 *  that satisfies the specified pattern. The indices are ordered in
	 *  terms of rising Hilbert space indices.
	 *
	 *  @param pattern Pattern to match agains. IDX_ALL can be used as a
	 *  wildcard.
	 *
	 *  @return A list of physical indices that match the specified
	 *  pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

	/** Sort HoppingAmplitudes in row order. */
	void sort(HoppingAmplitudeTree *rootNode);

	/** Print @link HoppingAmplitude HoppingAmplitudes @endlink. Mainly for
	 *  debuging purposes. */
	void print();

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the Index. */
	unsigned int getSizeInBytes() const;

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink stored in the tree structure. */
	class Iterator{
	public:
		/** Root node to iterate from. */
		const HoppingAmplitudeTree* tree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Current HoppingAmplitude that the iterator points at at the
		 *  currentIndex. */
		int currentHoppingAmplitude;

		/** Copy constructor. */
		Iterator(const Iterator &iterator);

		/** Move constructor. */
		Iterator(Iterator &&iterator);

		/** Constructor. */
		Iterator(const HoppingAmplitudeTree *tree);

		/** Assignment operator. */
		Iterator& operator=(const Iterator &rhs);

		/** Move assignment operator. */
		Iterator& operator=(Iterator &&rhs);

		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNextHA();

		/** Get HoppingAmplitude currently pointed at. */
		const HoppingAmplitude* getHA() const;

		/** Get minimum basis index. */
		int getMinBasisIndex() const;

		/** Get maximum basis index. */
		int getMaxBasisIndex() const;

		/** Get number of basis indices. */
		int getNumBasisIndices() const;
	private:
		/** Search after next HoppingAmplitude. Is used by
		 *  HoppingAmplitudeTree::Iterator::searchNext and called
		 *  recursively. */
		bool searchNext(
			const HoppingAmplitudeTree *hoppingAmplitudeTree,
			unsigned int subindex
		);
	};

	/** Returns Iterator initialized to point at first HoppingAmplitude. */
	Iterator begin() const;

	/** Implements Serializeable::serialize.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string represenation of the Index. */
	virtual std::string serialize(Mode mode) const;
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
	void add(HoppingAmplitude &ha, unsigned int subindex);

	/** Get sub tree. Is called by HoppingAmplitudeTree::getSubTree and is
	 *  called recursively. */
	const HoppingAmplitudeTree* getSubTree(
		const Index &subspace,
		unsigned int subindex
	) const;

	/** Returns true if the subspace is a proper subsapce. Is called by
	 *  HoppingAmplitudeTree::isProperSubspace and is called recursively.
	 */
	bool isProperSubspace(
		const Index &subspace,
		unsigned int subindex
	) const;

	/** Adds indices of all separate blocks below the current node to the
	 *  indexTree. Is calleed by HoppingAmplitudeTree::getBlockIndices()
	 *  and is called recuresively. */
	void getBlockIndices(IndexTree &blockIndices, Index index) const;

	/** Get HoppingAmpilitudes. Is called by the public
	 *  HoppingAmplitudeTree::getHAs and is called recursively. */
	const std::vector<HoppingAmplitude>* getHAs(
		Index index,
		unsigned int subindex
	) const;

	/** Get Hilbert space index for given physical index. Is called by the
	 *  public HoppingAmplitudeTree::getBasisIndex and is called recursively. */
	int getBasisIndex(const Index &index, unsigned int subindex) const;

	/** Get physical index for given Hilbert space index. Is called by the
	 *  public HoppingAmplitudeTreee::getPhysicalIndex and is called
	 *  recursively. */
	void getPhysicalIndex(int basisIndex, std::vector<int> *indices) const;

	/** Get minimum index on HoppingAmplitudeTree. */
	int getMinIndex() const;

	/** Get max index on HoppingAmplitudeTree. */
	int getMaxIndex() const;

	/** Generate Hilbert space indices. Is called by the public
	 *  HoppingAmplitudeTree::generateBasisIndices and is called
	 *  recursively. */
	int generateBasisIndices(int i);

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

};	//End of namespace TBTK

#endif
