/** @file TreeNode
 *  @brief Node in tree structure used by AmplitudeSet to store HoppingAmplitudes
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TREE_NODE
#define COM_DAFER45_TBTK_TREE_NODE

#include <vector>
#include "HoppingAmplitude.h"

class TreeNode{
public:
	//Basis index for the Hamiltonian
	int basisIndex;
	int basisSize;
	//HoppingAmplitudes stored on this node, should only be non-empty
	//if the node is a leaf node. That is, if the node corresponds to
	//a last subindex index.
	std::vector<HoppingAmplitude> hoppingAmplitudes;
	//Child nodes. Never non-empty at the same time as
	//hoppingAmplitudes
	std::vector<TreeNode> children;

	TreeNode();

	void add(HoppingAmplitude ha);
	std::vector<HoppingAmplitude>* getHAs(Index index);
	int getBasisIndex(const Index &index);

	void generateBasisIndices();

	void print();

	class iterator{
	public:
		TreeNode* tree;
		std::vector<int> currentIndex;
		int currentHoppingAmplitude;
		iterator(TreeNode *tree);
		void reset();
		void searchNextHA();
		HoppingAmplitude* getHA();
//		void operator++();
//		HoppingAmplitude& operator*();
	private:
		bool searchNext(TreeNode *treeNode, unsigned int subindex);
	};

	iterator begin();
private:
	void add(HoppingAmplitude ha, unsigned int subindex);
	std::vector<HoppingAmplitude>* getHAs(Index index, unsigned int subindex);
	int getBasisIndex(const Index &index, unsigned int subindex);
	int generateBasisIndices(int i);
	void print(unsigned int subindex);
	//Returns (depth) first HoppingAmplitude as an example, in
	//case of error while adding HoppingAmplitudes to the tree.
	HoppingAmplitude getFirstHA();
};

#endif

