/** @file AmplitudeSet.h
 *  @brief HoopingAmplitude container
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_AMPLITUDE_SET
#define COM_DAFER45_TBTK_AMPLITUDE_SET

#include <vector>
#include "HoppingAmplitude.h"
#include "TreeNode.h"
#include <iostream>

class AmplitudeSet{
public:
	TreeNode tree;

	void addHA(HoppingAmplitude ha);
	void addHAAndHC(HoppingAmplitude ha);
	std::vector<HoppingAmplitude>* getHAs(Index index);
	int getBasisIndex(const Index &index);
	int getBasisSize();

	void construct();

	class iterator{
	public:
		~iterator();

		void reset();
		void searchNextHA();
		HoppingAmplitude* getHA();
	private:
		friend class AmplitudeSet;
		iterator(TreeNode *tree);
		TreeNode::iterator* it;
	};

	AmplitudeSet::iterator getIterator();

	void print();

	void tabulate(int **table, int *dims);
private:
};

inline void AmplitudeSet::addHA(HoppingAmplitude ha){
	tree.add(ha);
}

inline void AmplitudeSet::addHAAndHC(HoppingAmplitude ha){
	tree.add(ha);
	tree.add(ha.getHermitianConjugate());
}

inline std::vector<HoppingAmplitude>* AmplitudeSet::getHAs(Index index){
	return tree.getHAs(index);
}

inline int AmplitudeSet::getBasisIndex(const Index &index){
	return tree.getBasisIndex(index);
}

inline int AmplitudeSet::getBasisSize(){
	return tree.basisSize;
}

inline void AmplitudeSet::construct(){
	tree.generateBasisIndices();
}

#endif

