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
 *  @brief HoppingAmplitude container.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE_SET
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE_SET

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/HoppingAmplitudeTree.h"
#include "TBTK/IndexTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/SparseMatrix.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{

/** @brief HoppingAmplitude container.
 *
 *  A HoppingAmplitudeSet is a container for @link HoppingAmplitude
 *  HoppingAmplitudes @endlink. The structure contains the root node for the
 *  tree structure in which the @link HoppingAmplitude HoppingAmplitudes
 *  @endlink are stored, as well as functions for adding and accessing
 *  HoppingAmplitudes. Once all @link HoppingAmplitude HoppingAmplitudes
 *  @endlink have been added to the HoppingAmplitudeSet, the construct method
 *  has to be called in order to construct an appropriate Hilbert space. The
 *  HoppingAmplitudeSet is most importantly used by the Model to store the
 *  Hamiltonian. */
class HoppingAmplitudeSet :
	virtual public Serializable,
	private HoppingAmplitudeTree
{
public:
	using HoppingAmplitudeTree::add;
	using HoppingAmplitudeTree::getHoppingAmplitudes;
	using HoppingAmplitudeTree::getBasisIndex;
	using HoppingAmplitudeTree::getPhysicalIndex;
	using HoppingAmplitudeTree::getBasisSize;
	using HoppingAmplitudeTree::isProperSubspace;
	using HoppingAmplitudeTree::getSubspaceIndices;
	using HoppingAmplitudeTree::getIndexList;

	/** Constructs a HoppingAmplitudeSet. */
	HoppingAmplitudeSet();

	/** Constructs a HoppingAmplitudeSet with a preallocated storage
	 *  structure such that the addition of HoppingAmplitudes with indices
	 *  that have the same subindex structure as 'capacity', but with
	 *  smaller subindices will not cause reallocation for the main storage
	 *  structure. Internal containers for @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink may still be reallocated.
	 *
	 *  @param capacity 'Index capacity'. */
	HoppingAmplitudeSet(const std::vector<unsigned int> &capacity);

	/** Copy constructor.
	 *
	 *  @param hoppingAmplitudeSet HoppingAmplitudeSet to copy. */
	HoppingAmplitudeSet(const HoppingAmplitudeSet &hoppingAmplitudeSet);

	/** Move constructor.
	 *
	 *  @param hoppingAMplitudeSet HoppingAmplitudeSet to move. */
	HoppingAmplitudeSet(HoppingAmplitudeSet &&hoppingAmplitudeSet);

	/** Constructor. Constructs the HoppingAmplitudeSet from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the HoppingAmplitudeSet.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	HoppingAmplitudeSet(const std::string &serializeation, Mode mode);

	/** Destructor. */
	virtual ~HoppingAmplitudeSet();

	/** Assignment operator.
	 *
	 *  @param rhs HoppingAmplitude to assign to the left hand side.
	 *
	 *  @return Reference to the assigned HoppingAmplitudeSet. */
	HoppingAmplitudeSet& operator=(const HoppingAmplitudeSet &rhs);

	/** Move assignment operator.
	 *
	 *  @param rhs HoppingAmplitude to assign to the left hand side.
	 *
	 *  @return Reference to the assigned HoppingAmplitudeSet. */
	HoppingAmplitudeSet& operator=(HoppingAmplitudeSet &&rhs);

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** Check whether the Hilbert space basis has been constructed.
	 *
	 *  @return True if the Hilbert space basis has been constructed. */
	bool getIsConstructed() const;

	/** Get first index in block.
	 *
	 *  @param subspaceIndex The physical Index of the subspace.
	 *
	 *  @return The first Hilbert space index in the given subspace. If the
	 *  subspace is emtpy, -1 is returned. */
	int getFirstIndexInBlock(const Index &blockIndex) const;

	/** Get last index in block.
	 *
	 *  @param subspaceIndex The physical Index of the subspace.
	 *
	 *  @return The last Hilbert space index in the given subspace. If the
	 *  subspace is empty, -1 is returned. */
	int getLastIndexInBlock(const Index &blockIndex) const;

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

	/** Get a sprase matrix corresponding to the HoppingAMplitudeSet. The
	 *  basis of the matrix is the Hilbert space basis.
	 *
	 *  @return A sparse matrix representation of the HoppingAmplitudeSet.
	 */
	SparseMatrix<std::complex<double>> getSparseMatrix() const;

	/** Iterator for iterating through @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	class Iterator{
	public:
		/** Copy constructor. */
		Iterator(const Iterator &iterator);

		/** Move constructor. */
		Iterator(Iterator &&iterator);

		/** Destructor. */
		~Iterator();

		/** Assignment operator. */
		Iterator& operator=(const Iterator &rhs);

		/** Move assignment operator. */
		Iterator& operator=(Iterator &&rhs);

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
		Iterator(const HoppingAmplitudeTree *hoppingAmplitudeTree);

		/** HoppingAmplitudeTree iterator. Implements the actual
		 *  iteration. */
		HoppingAmplitudeTree::Iterator* it;
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

	/** Implements Serializable::serialize(). */
	virtual std::string serialize(Mode mode) const;

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;
private:
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

inline void HoppingAmplitudeSet::construct(){
	TBTKAssert(
		!isConstructed,
		"HoppingAmplitudeSet::construct()",
		"HoppingAmplitudeSet is already constructed.",
		""
	);

	HoppingAmplitudeTree::generateBasisIndices();
	isConstructed = true;
}

inline bool HoppingAmplitudeSet::getIsConstructed() const{
	return isConstructed;
}

inline int HoppingAmplitudeSet::getFirstIndexInBlock(
	const Index &blockIndex
) const{
	return HoppingAmplitudeTree::getFirstIndexInSubspace(blockIndex);
}

inline int HoppingAmplitudeSet::getLastIndexInBlock(
	const Index &blockIndex
) const{
	return HoppingAmplitudeTree::getLastIndexInSubspace(blockIndex);
}

inline void HoppingAmplitudeSet::sort(){
	TBTKAssert(
		isConstructed,
		"HoppingAmplitudeSet::sort()",
		"HoppingAmplitudeSet has to be constructed first.",
		""
	);

	if(!isSorted){
		HoppingAmplitudeTree::sort(this);
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

inline SparseMatrix<std::complex<double>> HoppingAmplitudeSet::getSparseMatrix(
) const{
	TBTKAssert(
		isConstructed,
		"HoppingAmplitudeSet::getSparseMatrix()",
		"HoppingAmplitudeSet has to be constructed first.",
		""
	);

	SparseMatrix<std::complex<double>> sparseMatrix(
		SparseMatrix<std::complex<double>>::StorageFormat::CSC
	);

	Iterator iterator = getIterator();
	const HoppingAmplitude *hoppingAmplitude;
	while((hoppingAmplitude = iterator.getHA())){
		sparseMatrix.add(
			getBasisIndex(hoppingAmplitude->getToIndex()),
			getBasisIndex(hoppingAmplitude->getFromIndex()),
			hoppingAmplitude->getAmplitude()
		);

		iterator.searchNextHA();
	}
	sparseMatrix.construct();

	return sparseMatrix;
}

inline unsigned int HoppingAmplitudeSet::getSizeInBytes() const{
	unsigned int size = sizeof(*this) - sizeof(HoppingAmplitudeTree);
	size += HoppingAmplitudeTree::getSizeInBytes();
	if(numMatrixElements > 0){
		size += numMatrixElements*(
			sizeof(*cooRowIndices)
			+ sizeof(*cooColIndices)
			+ sizeof(*cooValues)
		);
	}

	return size;
}

};	//End of namespace TBTK

#endif
