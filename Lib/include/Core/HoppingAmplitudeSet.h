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
 *  @file HoppingAmplitudeSet.h
 *  @brief HoppingAmplitude container
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE_SET
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE_SET

#include "HoppingAmplitude.h"
#include "Serializeable.h"
#include "Streams.h"
#include "TBTKMacros.h"
#include "TreeNode.h"

#include <complex>
#include <vector>

namespace TBTK{

/** A HoppingAmplitudeSet is a container for @link HoppingAmplitude
 *  HoppingAmplitudes @endlink. The structure contains the root node for the
 *  tree structure in which the @link HoppingAmplitude HoppingAmplitudes
 *  @endlink are stored, as well as functions for adding and accessing
 *  HoppingAmplitudes. Once all @link HoppingAmplitude HoppingAmplitudes
 *  @endlink have been added to the HoppingAmplitudeSet, the construct method
 *  has to be called in order to construct an appropriate Hilbert space. The
 *  HoppingAmplitudeSet is most importantly used by the Model to store the
 *  Hamiltonian. */
class HoppingAmplitudeSet : public Serializeable{
public:
	/** Constructor. */
	HoppingAmplitudeSet();

	/** Copy constructor. */
	HoppingAmplitudeSet(const HoppingAmplitudeSet &hoppingAmplitudeSet);

	/** Move constructor. */
	HoppingAmplitudeSet(HoppingAmplitudeSet &&hoppingAmplitudeSet);

	/** Constructor. Constructs the HoppingAmplitudeSet from a
	 *  serialization string. */
	HoppingAmplitudeSet(const std::string &serializeation, Mode mode);

	/** Destructor. */
	virtual ~HoppingAmplitudeSet();

	/** Assignment operator. */
	HoppingAmplitudeSet& operator=(const HoppingAmplitudeSet &rhs);

	/** Move assignment operator. */
	HoppingAmplitudeSet& operator=(HoppingAmplitudeSet &&rhs);

	/** Add a single HoppingAmplitude.
	 *
	 *  @param ha HoppingAmplitude to add. */
	void addHoppingAmplitude(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate.
	 *
	 *  @param HoppingAmplitude to add. */
	void addHoppingAmplitudeAndHermitianConjugate(HoppingAmplitude ha);

	/** Get all @link HoppingAmplitude HoppingAmplitudes @endlink with
	 * given 'from'-index.
	 *
	 *  @param index 'From'-index to get HoppingAmplitudes for. */
	const std::vector<HoppingAmplitude>* getHAs(Index index) const;

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *
	 *  @param index 'From'-index to get Hilbert space index for. */
	int getBasisIndex(const Index &index) const;

	/** Get Physical index for given Hilbert space basis index. */
	Index getPhysicalIndex(int basisIndex) const;

	/** Get size of Hilbert space. */
	int getBasisSize() const;

	/** Returns true if the subspace is a proper subspace. That is, if all
	 *  indices starting with the indices in 'subspace' only are connected
	 *  to other indices with the same initial subspace indices. */
	bool isProperSubspace(const Index &subspace);

	/** Returns and IndexTree containing all subspace indices. */
	IndexTree getSubspaceIndices() const;

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** Returns true if the Hilbert space basis has been constructed. */
	bool getIsConstructed() const;

	/** Generate a list of indices satisfying the specified pattern. */
	std::vector<Index> getIndexList(const Index &pattern) const;

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
	 *  the HoppingAmplitudeSet. Only used if COO format has been
	 *  constructed. */
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

		/** Get minimum index. */
		int getMinBasisIndex() const;

		/** Get maximum index. */
		int getMaxBasisIndex() const;

		/** Get number of basis indices. */
		int getNumBasisIndices() const;
	private:
		/** The iterator can only be constructed by the
		 *  HoppingAmplitudeSet. */
		friend class HoppingAmplitudeSet;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the HoppingAmplitudeSet. */
		Iterator(const TreeNode *tree);

		/** TreeNode iterator. Implements the actual iteration. */
		TreeNode::Iterator* it;
	};

	/** Returns an iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	HoppingAmplitudeSet::Iterator getIterator() const;

	/** Returns an iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. The iterator is restricted to the
	 *  subspace for which the 'from'-index starts with the indices in
	 *  the argument 'subspace'. */
	HoppingAmplitudeSet::Iterator getIterator(const Index &subspace) const;

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

	/** Implements Serializeable::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
	/** Root node for the tree structure in which HoppingAmplitudes are
	 *  stored. */
	TreeNode tree;

	/** Flag indicating whether the HoppingAmplitudeSet have been
	 *  constructed. */
	bool isConstructed;

	/** Flag indicating whether the HoppingAmplitudeSet have been sorted.
	 */
	bool isSorted;

	/** Number of matrix elements in HoppingAmplitudeSet. Is only used and
	 *  if COO format has been constructed and is otherwise -1. */
	int numMatrixElements;

	/** COO format row indices. */
	int *cooRowIndices;

	/** COO format column indices. */
	int *cooColIndices;

	/** COO format values. */
	std::complex<double> *cooValues;
};

inline void HoppingAmplitudeSet::addHoppingAmplitude(HoppingAmplitude ha){
	tree.add(ha);
}

inline void HoppingAmplitudeSet::addHoppingAmplitudeAndHermitianConjugate(
	HoppingAmplitude ha
){
	tree.add(ha);
	tree.add(ha.getHermitianConjugate());
}

inline const std::vector<HoppingAmplitude>* HoppingAmplitudeSet::getHAs(
	Index index
) const{
	return tree.getHAs(index);
}

inline int HoppingAmplitudeSet::getBasisIndex(const Index &index) const{
	return tree.getBasisIndex(index);
}

inline Index HoppingAmplitudeSet::getPhysicalIndex(int basisIndex) const{
	return tree.getPhysicalIndex(basisIndex);
}

inline int HoppingAmplitudeSet::getBasisSize() const{
//	return tree.basisSize;
	return tree.getBasisSize();
}

inline bool HoppingAmplitudeSet::isProperSubspace(const Index &subspace){
	return tree.isProperSubspace(subspace);
}

inline IndexTree HoppingAmplitudeSet::getSubspaceIndices() const{
	return tree.getSubspaceIndices();
}

inline void HoppingAmplitudeSet::construct(){
	TBTKAssert(
		!isConstructed,
		"HoppingAmplitudeSet::construct()",
		"HoppingAmplitudeSet is already constructed.",
		""
	);

	tree.generateBasisIndices();
	isConstructed = true;
}

inline bool HoppingAmplitudeSet::getIsConstructed() const{
	return isConstructed;
}

inline std::vector<Index> HoppingAmplitudeSet::getIndexList(
	const Index &pattern
) const{
	return tree.getIndexList(pattern);
}

inline void HoppingAmplitudeSet::sort(){
	TBTKAssert(
		isConstructed,
		"HoppingAmplitudeSet::sort()",
		"HoppingAmplitudeSet has to be constructed first.",
		""
	);

	if(!isSorted){
		tree.sort(&tree);
		isSorted = true;
	}
}

inline const int* HoppingAmplitudeSet::getCOORowIndices() const{
	return cooRowIndices;
}

inline const int* HoppingAmplitudeSet::getCOOColIndices() const{
	return cooColIndices;
}

inline const std::complex<double>* HoppingAmplitudeSet::getCOOValues() const{
	return cooValues;
}

};	//End of namespace TBTK

#endif
