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

	/** Generate Hilbert space indices. No more @link HoppingAmplitude
	 *   HoppingAmplitudes @endlink should be added after this call. */
	void generateBasisIndices();

	/** Print @link HoppingAmplitude HoppingAmplitudes @endlink. Mainly for
	 *  debuging purposes. */
	void print();

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink stored in the tree structure. */
	class iterator{
	public:
		/** Root node to iterate from. */
		TreeNode* tree;

		/** Current index at which the iterator points at. */
		std::vector<int> currentIndex;

		/** Current HoppingAmplitude that the iterator points at at the
		 *  currentIndex. */
		int currentHoppingAmplitude;

		/** Constructor. */
		iterator(TreeNode *tree);

		/** Reset iterator. */
		void reset();

		/** Advance the iterator by one. */
		void searchNextHA();

		/** Get HoppingAmplitude currently pointed at. */
		HoppingAmplitude* getHA();
	private:
		/** Search after next HoppingAmplitude. Is used by
		 *  TreeNode::iterator::searchNext and called recursively. */
		bool searchNext(TreeNode *treeNode, unsigned int subindex);
	};

	/** Returns iterator initialized to point at first HoppingAmplitude. */
	iterator begin();
private:
	/** Add HoppingAmplitude. Is called by the public TreeNode::add and is
	 *  called recursively. */
	void add(HoppingAmplitude ha, unsigned int subindex);

	/** Get HoppingAmpilitudes. Is called by the public TreeNode::getHAs
	 *  and is called recursively. */
	std::vector<HoppingAmplitude>* getHAs(Index index, unsigned int subindex);

	/** Get Hilbert space index for given physical index. Is called by the
	 *  public TreeNode::getBasisIndex and is called recursively. */
	int getBasisIndex(const Index &index, unsigned int subindex);

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

#endif
