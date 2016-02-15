/** @package TBTKcalc
 *  @file AmplitudeSet.h
 *  @brief HoppingAmplitude container
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_AMPLITUDE_SET
#define COM_DAFER45_TBTK_AMPLITUDE_SET

#include <vector>
#include "HoppingAmplitude.h"
#include "TreeNode.h"
#include <iostream>

/** An AmplitudeSet is a container for @link HoppingAmplitude HoppingAmplitudes
 *  @endlink. The structure contains the root node for the tree structure in
 *  which the @link HoppingAmplitude HoppingAmplitudes @endlink are stored, as
 *  well as functions for adding and accessing HoppingAmplitudes. Once all
 *  @link HoppingAmplitude HoppingAmplitudes @endlink have been added to the
 *  AmplitudeSet, the construct method has to be called in order to construct
 *  an appropriate Hilbert space. The AmplitudeSet is most importantly used by
 *  the Model to store the Hamiltonian.
 */
class AmplitudeSet{
public:
	/** Root node for the tree structure in which HoppingAmplitudes are
	 *  stored*/
	TreeNode tree;

	/** Add a single HoppingAmplitude. */
	void addHA(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate. */
	void addHAAndHC(HoppingAmplitude ha);

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 * given 'from'-index.
	 *  @param index 'From'-index to get HoppingAmplitudes for. */
	std::vector<HoppingAmplitude>* getHAs(Index index);

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *  @param index 'From'-index to get Hilbert space index for. */
	int getBasisIndex(const Index &index);

	/** Get size of Hilbert space. */
	int getBasisSize();

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	class iterator{
	public:
		/** Destructor. */
		~iterator();

		/** Reset iterator. */
		void reset();

		/** Iterate to next HoppingAmplitude. */
		void searchNextHA();

		/** Get current HoppingAmplitude. */
		HoppingAmplitude* getHA();
	private:
		/** The iterator can only be constructed by the AmplitudeSet. */
		friend class AmplitudeSet;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the AmplitudeSet. */
		iterator(TreeNode *tree);

		/** TreeNode iterator. Implements the actual iteration. */
		TreeNode::iterator* it;
	};

	/** Returns an iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	AmplitudeSet::iterator getIterator();

	/** Print tree structure. Mainly for debuging. */
	void print();

	/** Tabulates @link HoppingAmplitude HoppingAmplitudes @endlink to make
	 *  them possible to export. */
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

