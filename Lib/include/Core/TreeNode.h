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
 *  @file TreeNode.h
 *  @brief Node in tree used by AmplitudeSet to store @link HoppingAmplitude HoppingAmplitudes @endlink
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TREE_NODE
#define COM_DAFER45_TBTK_TREE_NODE

#include "HoppingAmplitude.h"

#include <vector>

namespace TBTK{

/** TreeNode structure used to build a tree for stroing @link HoppingAmplitude
 *   HoppingAmplitudes @endlink. Used by AmplitudeSet.*/
class TreeNode{
public:
	/** Basis index for the Hamiltonian. */
	int basisIndex;

	/** Basis size of Hamiltonian. */
	int basisSize;

	/** HoppingAmplitudes stored on this node, should only be non-empty if
	 *  the node is a leaf node. That is, if the node corresponds to a last
	 *   subindex index. */
	std::vector<HoppingAmplitude> hoppingAmplitudes;

	/** Child nodes. Never non-empty at the same time as hoppingAmplitudes
	*/
	std::vector<TreeNode> children;

	/** Constructor. */
	TreeNode();

	/** Add a HoppingAmplitude. */
	void add(HoppingAmplitude ha);

	/** Get sub tree. */
	const TreeNode* getSubTree(const Index &subspace) const;

	/** Returns true if the subspace is a proper subsapce. That is, if the
	 *  corresponding subtree only contains HoppingAmplitudes that connects
	 *  sistes within the subtree. */
	bool isProperSubspace(const Index &subspace);

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 *  given 'from'-index. */
	const std::vector<HoppingAmplitude>* getHAs(Index index) const;

	/** Get Hilbert space basis index for given physical index. */
	int getBasisIndex(const Index &index) const;

	/** Get physical index for given Hilbert space absis index. */
	Index getPhysicalIndex(int basisIndex) const;

	/** Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

	/** Generate a list of indices satisfying the specified pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

	/** Sort HoppingAmplitudes in row order. */
	void sort(TreeNode *rootNode);

	/** Print @link HoppingAmplitude HoppingAmplitudes @endlink. Mainly for
	 *  debuging purposes. */
	void print();

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink stored in the tree structure. */
	class Iterator{
	public:
		/** Root node to iterate from. */
		const TreeNode* tree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Current HoppingAmplitude that the iterator points at at the
		 *  currentIndex. */
		int currentHoppingAmplitude;

		/** Constructor. */
		Iterator(const TreeNode *tree);

		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNextHA();

		/** Get HoppingAmplitude currently pointed at. */
		const HoppingAmplitude* getHA() const;
	private:
		/** Search after next HoppingAmplitude. Is used by
		 *  TreeNode::Iterator::searchNext and called recursively. */
		bool searchNext(const TreeNode *treeNode, unsigned int subindex);
	};

	/** Returns Iterator initialized to point at first HoppingAmplitude. */
	Iterator begin() const;
private:
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

	/** Add HoppingAmplitude. Is called by the public TreeNode::add and is
	 *  called recursively. */
	void add(HoppingAmplitude &ha, unsigned int subindex);

	/** Get sub tree. Is called by TreeNode::getSubTree and is called
	 *  recursively. */
	const TreeNode* getSubTree(
		const Index &subspace,
		unsigned int subindex
	) const;

	/** Returns true if the subspace is a proper subsapce. Is called by
	 *  TreeNode::isProperSubspace and is called recursively. */
	bool isProperSubspace(const Index &subspace, unsigned int subindex);

	/** Get HoppingAmpilitudes. Is called by the public TreeNode::getHAs
	 *  and is called recursively. */
	const std::vector<HoppingAmplitude>* getHAs(
		Index index,
		unsigned int subindex
	) const;

	/** Get Hilbert space index for given physical index. Is called by the
	 *  public TreeNode::getBasisIndex and is called recursively. */
	int getBasisIndex(const Index &index, unsigned int subindex) const;

	/** Get physical index for given Hilbert space index. Is called by the
	 *  public TreeNode::getPhysicalIndex and is called recursively. */
	void getPhysicalIndex(int basisIndex, std::vector<int> *indices) const;

	/** Get minimum index on TreeNode. */
	int getMinIndex() const;

	/** Get max index on TreeNode. */
	int getMaxIndex() const;

	/** Generate Hilbert space indices. Is called by the public
	 *  TreeNode::generateBasisIndices and is called recursively. */
	int generateBasisIndices(int i);

	/** Print HoppingAmplitudes. Is called by the public TreeNode::print
	 *  and is called recursively. Mainly for debuging purposes. */
	void print(unsigned int subindex);

	/** Returns (depth) first HoppingAmplitude as an example, in case of
	 *  error while adding HoppingAmplitudes to the tree. */
	HoppingAmplitude getFirstHA() const;
};

};	//End of namespace TBTK

#endif
