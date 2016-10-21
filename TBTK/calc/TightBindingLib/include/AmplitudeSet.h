/** @package TBTKcalc
 *  @file AmplitudeSet.h
 *  @brief HoppingAmplitude container
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_AMPLITUDE_SET
#define COM_DAFER45_TBTK_AMPLITUDE_SET

#include "HoppingAmplitude.h"
#include "TreeNode.h"
#include "Streams.h"

#include <vector>
#include <complex>

namespace TBTK{

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
	 *  stored. */
	TreeNode tree;

	/** Constructor. */
	AmplitudeSet();

	/** Destructor. */
	~AmplitudeSet();

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

	/** Returns true if the Hilbert space basis has been constructed. */
	bool getIsConstructed();

	/** Sort HoppingAmplitudes. */
	void sort();

	/** Construct Hamiltonian on COO format. */
	void constructCOO();

	/** Destruct Hamiltonian on COO format. */
	void destructCOO();

	/** Reconstruct Hamiltonian on COO format. Only has any effect if a
	 *  Hamiltonian on COO format already is constructed. Is necessary to
	 *  reflect changes in the Hamiltonain due to changes in values
	 *  returned by HoppingAmplitude-callback functions. The function is
	 *  intended to be called by the Model whenever it is notified of
	 *  possible changes in values returned by the callback-functions. */
	void reconstructCOO();

	/** Get number of matrix elements in the Hamiltonian corresponding to
	 *  the AmplitudeSet. */
	int getNumMatrixElements();

	/** Get row indices on COO format. */
	const int* getCOORowIndices();

	/** Get col indices on COO format. */
	const int* getCOOColIndices();

	/** Get row indices on COO format. */
	const std::complex<double>* getCOOValues();

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	class Iterator{
	public:
		/** Destructor. */
		~Iterator();

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
		Iterator(TreeNode *tree);

		/** TreeNode iterator. Implements the actual iteration. */
		TreeNode::Iterator* it;
	};

	/** Returns an iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	AmplitudeSet::Iterator getIterator();

	/** Print tree structure. Mainly for debuging. */
	void print();

	/** Tabulates @link HoppingAmplitude HoppingAmplitudes @endlink to make
	 *  them possible to export. */
	void tabulate(
		std::complex<double> **amplitudes,
		int **indices,
		int *numHoppingAmplitudes,
		int *maxIndexSize
	);
private:
	/** Flag indicating whether the AmplitudeSet have been constructed. */
	bool isConstructed;

	/** Flag indicating whether the AmplitudeSet have been sorted. */
	bool isSorted;

	/** Number of matrix elements in AmplitudeSet. */
	int numMatrixElements;

	/** COO format row indices. */
	int *cooRowIndices;

	/** COO format column indices. */
	int *cooColIndices;

	/** COO format values. */
	std::complex<double> *cooValues;
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
	if(isConstructed){
		Util::Streams::err << "Error in AmplitudeSet::construct(): AmplitudeSet is already constructed.\n";
		Util::Streams::closeLog();
		exit(1);
	}

	tree.generateBasisIndices();
	isConstructed = true;
}

inline bool AmplitudeSet::getIsConstructed(){
	return isConstructed;
}

inline void AmplitudeSet::sort(){
	if(!isConstructed){
		Util::Streams::err << "Error in AmplitudeSet::sort(): AmplitudeSet has to be constructed first.\n";
		Util::Streams::closeLog();
		exit(1);
	}

	if(!isSorted){
		tree.sort(&tree);
		isSorted = true;
	}
}

inline const int* AmplitudeSet::getCOORowIndices(){
	return cooRowIndices;
}

inline const int* AmplitudeSet::getCOOColIndices(){
	return cooColIndices;
}

inline const std::complex<double>* AmplitudeSet::getCOOValues(){
	return cooValues;
}

};	//End of namespace TBTK

#endif
