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

#include <vector>
#include "HoppingAmplitude.h"

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

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 *  given 'from'-index. */
	std::vector<HoppingAmplitude>* getHAs(Index index);

	/** Get Hilbert space basis index for given physical index. */
	int getBasisIndex(const Index &index);

	/** Get physical index for given Hilbert space absis index. */
	Index getPhysicalIndex(int basisIndex);

	/** Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

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
		TreeNode* tree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Current HoppingAmplitude that the iterator points at at the
		 *  currentIndex. */
		int currentHoppingAmplitude;

		/** Constructor. */
		Iterator(TreeNode *tree);

		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNextHA();

		/** Get HoppingAmplitude currently pointed at. */
		HoppingAmplitude* getHA();
	private:
		/** Search after next HoppingAmplitude. Is used by
		 *  TreeNode::Iterator::searchNext and called recursively. */
		bool searchNext(TreeNode *treeNode, unsigned int subindex);
	};

	/** Returns Iterator initialized to point at first HoppingAmplitude. */
	Iterator begin();
private:
	/** Add HoppingAmplitude. Is called by the public TreeNode::add and is
	 *  called recursively. */
	void add(HoppingAmplitude &ha, unsigned int subindex);

	/** Get HoppingAmpilitudes. Is called by the public TreeNode::getHAs
	 *  and is called recursively. */
	std::vector<HoppingAmplitude>* getHAs(
		Index index,
		unsigned int subindex
	);

	/** Get Hilbert space index for given physical index. Is called by the
	 *  public TreeNode::getBasisIndex and is called recursively. */
	int getBasisIndex(const Index &index, unsigned int subindex);

	/** Get physical index for given Hilbert space index. Is called by the
	 *  public TreeNode::getPhysicalIndex and is called recursively. */
	void getPhysicalIndex(int basisIndex, std::vector<int> *indices);

	/** Get minimum index on TreeNode. */
	int getMinIndex();

	/** Get max index on TreeNode. */
	int getMaxIndex();

	/** Generate Hilbert space indices. Is called by the public
	 *  TreeNode::generateBasisIndices and is called recursively. */
	int generateBasisIndices(int i);

	/** Print HoppingAmplitudes. Is called by the public TreeNode::print
	 *  and is called recursively. Mainly for debuging purposes. */
	void print(unsigned int subindex);

	/** Returns (depth) first HoppingAmplitude as an example, in case of
	 *  error while adding HoppingAmplitudes to the tree. */
	HoppingAmplitude getFirstHA();
};

};	//End of namespace TBTK

#endif
