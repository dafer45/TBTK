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
 *  @brief Node in tree used by AmplitudeSet to store @link HoppingAmplitude HoppingAmplitudes @endlink
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TREE_NODE
#define COM_DAFER45_TBTK_TREE_NODE

#include "HoppingAmplitude.h"
#include "IndexTree.h"
#include "Serializeable.h"

#include <vector>

namespace TBTK{

/** HoppingAmplitudeTree is a tree structure used to build a tree for stroing
 *  @link HoppingAmplitude HoppingAmplitudes @endlink. Used by AmplitudeSet.*/
class HoppingAmplitudeTree : public Serializeable{
public:
	/** Constructor. */
	HoppingAmplitudeTree();

	/** Constructor. Preallocates a HoppingAmplitudeTree such that the
	 *  addition of HoppingAMplitudes with indices that have the same
	 *  subindex structure as 'capacity', but with smaller subindices will
	 *  not cause reallocation for the main tree stucture. Internal
	 *  containers for HoppingAMplitudes may still be reallocated. */
	HoppingAmplitudeTree(
		const std::vector<unsigned int> &capacity
	);

	/** Constructor. Constructs the HoppingAmplitudeTree from a
	 *  serialization string. */
	HoppingAmplitudeTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~HoppingAmplitudeTree();

	/** Add a HoppingAmplitude. */
	void add(HoppingAmplitude ha);

	/** Get basis size. */
	int getBasisSize() const;

	/** Get IndexTree containing the Indices that describe the subspace
	 *  structure. */
	const HoppingAmplitudeTree* getSubTree(const Index &subspace) const;

	/** Returns true if the subspace is a proper subsapce. That is, if the
	 *  corresponding subtree only contains HoppingAmplitudes that connects
	 *  sistes within the subtree. */
	bool isProperSubspace(const Index &subspace) const;

	/** Returns an IndexTree containing all subspace indices. */
	IndexTree getSubspaceIndices() const;

	/** Get first index in subspace. */
	int getFirstIndexInSubspace(const Index &subspaceIndex) const;

	/** Get last index in subspace. */
	int getLastIndexInSubspace(const Index &subspaceIndex) const;

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 *  given 'from'-index. */
	const std::vector<HoppingAmplitude>* getHAs(Index index) const;

	/** Get Hilbert space basis index for given physical index. */
	int getBasisIndex(const Index &index) const;

	/** Get physical index for given Hilbert space basis index. */
	Index getPhysicalIndex(int basisIndex) const;

	/** Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

	/** Generate a list of indices satisfying the specified pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

	/** Sort HoppingAmplitudes in row order. */
	void sort(HoppingAmplitudeTree *rootNode);

	/** Print @link HoppingAmplitude HoppingAmplitudes @endlink. Mainly for
	 *  debuging purposes. */
	void print();

	/** Get size in bytes. */
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

	/** Implements Serializeable::serialize. */
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

	return size + sizeof(HoppingAmplitudeTree);
}

};	//End of namespace TBTK

#endif

