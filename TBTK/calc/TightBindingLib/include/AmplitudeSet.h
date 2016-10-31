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
#include "TBTKMacros.h"

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

	/** Add a single HoppingAmplitude.
	 *
	 *  @param ha HoppingAMplitude to add. */
	void addHA(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate.
	 *
	 *  @param HoppingAmplitude to add. */
	void addHAAndHC(HoppingAmplitude ha);

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 * given 'from'-index.
	 *
	 *  @param index 'From'-index to get HoppingAmplitudes for. */
	const std::vector<HoppingAmplitude>* getHAs(Index index) const;

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *
	 *  @param index 'From'-index to get Hilbert space index for. */
	int getBasisIndex(const Index &index) const;

	/** Get size of Hilbert space. */
	int getBasisSize() const;

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** Returns true if the Hilbert space basis has been constructed. */
	bool getIsConstructed() const;

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
	int getNumMatrixElements() const;

	/** Get row indices on COO format. */
	const int* getCOORowIndices() const;

	/** Get col indices on COO format. */
	const int* getCOOColIndices() const;

	/** Get row indices on COO format. */
	const std::complex<double>* getCOOValues() const;

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
		const HoppingAmplitude* getHA() const;
	private:
		/** The iterator can only be constructed by the AmplitudeSet. */
		friend class AmplitudeSet;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the AmplitudeSet. */
		Iterator(const TreeNode *tree);

		/** TreeNode iterator. Implements the actual iteration. */
		TreeNode::Iterator* it;
	};

	/** Returns an iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	AmplitudeSet::Iterator getIterator() const;

	/** Print tree structure. Mainly for debuging. */
	void print();

	/** Tabulates @link HoppingAmplitude HoppingAmplitudes @endlink to make
	 *  them possible to export.
	 *
	 *  @param amplitudes
	 *	Pointer to amplitude table pointer. Memory will be allocated
	 *	and has to be freed by the user. The array will contain all the
	 *	HoppingAmplitude values when the function returns.
	 *  @param table
	 *	Pointer to index table pointer. Memory will be allocated and
	 *	has to be freed by the user. The array will contain the 'to'-
	 *	and 'from'-indices for the corresponding HoppingAmplitude
	 *	values in amplitudes. The values are stored sequentially using
	 *	the format [to0] [padding] [from0] [padding] [to1] ..., where
	 *	the padding is added to align 'to'- and 'from'-indices in
	 *	memory in case multiple index sizes are encounterd. The number
	 *	of padding elements will be zero for indices of size
	 *	maxIndexSize and the padding value is -1. The total array size
	 *	is 2*numHoppingAmplitudes*maxIndexSize.
	 *  @param numHoppingAmplitudes
	 *	Pointer to int that will contain the number of
	 *	HoppingAMplitudes when the function returns.
	 *  @param maxIndexSize
	 *	Pointer to int that will contain the maximum number of
	 *	subindices encountered. */
	void tabulate(
		std::complex<double> **amplitudes,
		int **indices,
		int *numHoppingAmplitudes,
		int *maxIndexSize
	) const;
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

inline const std::vector<HoppingAmplitude>* AmplitudeSet::getHAs(Index index) const{
	return tree.getHAs(index);
}

inline int AmplitudeSet::getBasisIndex(const Index &index) const{
	return tree.getBasisIndex(index);
}

inline int AmplitudeSet::getBasisSize() const{
	return tree.basisSize;
}

inline void AmplitudeSet::construct(){
	TBTKAssert(
		!isConstructed,
		"AmplitudeSet::construct()",
		"AmplitudeSet is already constructed.",
		""
	);

	tree.generateBasisIndices();
	isConstructed = true;
}

inline bool AmplitudeSet::getIsConstructed() const{
	return isConstructed;
}

inline void AmplitudeSet::sort(){
	TBTKAssert(
		isConstructed,
		"AmplitudeSet::sort()",
		"AmplitudeSet has to be constructed first.",
		""
	);

	if(!isSorted){
		tree.sort(&tree);
		isSorted = true;
	}
}

inline const int* AmplitudeSet::getCOORowIndices() const{
	return cooRowIndices;
}

inline const int* AmplitudeSet::getCOOColIndices() const{
	return cooColIndices;
}

inline const std::complex<double>* AmplitudeSet::getCOOValues() const{
	return cooValues;
}

};	//End of namespace TBTK

#endif
