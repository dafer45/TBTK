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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file BasisState.h
 *  @brief State class with index and unit cell based overlap evaluation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BASIC_STATE
#define COM_DAFER45_TBTK_BASIC_STATE

#include "TBTK/AbstractState.h"
#include "TBTK/DefaultOperator.h"
#include "TBTK/Index.h"
#include "TBTK/IndexTree.h"

#include <complex>
#include <tuple>

namespace TBTK{

class BasicState : public AbstractState{
public:
	/** Constructor. */
	BasicState(const Index &index, const Index &unitCellIndex = {});

	/** Destructor. */
	virtual ~BasicState();

	/** Implements AbstracState::clone(). */
	virtual BasicState* clone() const;

	/** Add overlap. */
	void addOverlap(
		std::complex<double> overlap,
		const Index &braIndex,
		const Index &braRelativeUnitCell = {}
	);

	/** Add matrix element. */
	void addMatrixElement(
		std::complex<double> matrixElement,
		const Index &bra,
		const Index &braRelativeUnitCell = {}
	);

	/** Implements AbstractState::getOverlapWith(). */
	virtual std::complex<double> getOverlap(const AbstractState &bra) const;

	/** Implements AbstractState::getMatrixElementWith(). */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &bra,
		const AbstractOperator &o = DefaultOperator()
	) const;
private:
	class Storage{
	public:
		/** Maximum number of elements for which linear search is
		 *  performed to find overlaps and matrix elements. For larger
		 *  number of elements, IndexTrees are used instead. */
//		static constexpr unsigned int MAX_ELEMENTS_LINEAR_SEARCH = 50;

		/** Overlaps between the ket corresponding to this state and the bras
		 *  index by the indices. The first index in the tuple is the intra
		 *  cell index, while the second index is the unit cell index */
		std::vector<std::tuple<std::complex<double>, Index, Index>> overlaps;

		/** Flag indicating whether overlaps is sorted. */
		bool overlapsIsSorted;

		/** IndexTree used to speed up lookup in overlaps. */
//		IndexTree *overlapsIndexTree;

		/** Storage used to lookup overlaps using overlapsIndexTree. */
//		std::complex<double> *indexedOverlaps;

		/** Matrix elements between the ket corresponding to this state and the
		 *  bras index by the indices. The first index in the tuple is the
		 *  intra cell index, while the second index is the unit cell index */
		std::vector<std::tuple<std::complex<double>, Index, Index>> matrixElements;

		/** Flag indicating whether matrixElements is sorted. */
		bool matrixElementsIsSorted;

		/** IndexTree used to speed up lookup in matrixElements. */
//		IndexTree *matrixElementsIndexTree;

		/** Storage used to lookup matrix elements using matrixElementsIndexTree. */
//		std::complex<double> *indexedMatrixElements;

		/** Constructor. */
		Storage();

		/** Destructor. */
		~Storage();

		/** Grab reference (increments the reference counter). */
		void grab();

		/** Release reference. If the function returns true, the caller
		 *  should delete the Storage. */
		bool release();

		/** Sort overlaps. */
		void sortOverlaps();

		/** Sort matrix elements. */
		void sortMatrixElements();
	private:
		/** Reference counter. */
		unsigned int referenceCounter;
	};

	Storage *storage;
};

inline void BasicState::Storage::grab(){
	referenceCounter++;
}

inline bool BasicState::Storage::release(){
	referenceCounter--;
	if(referenceCounter == 0)
		return true;
	else
		return false;
}

};	//End of namespace TBTK

#endif
/// @endcond
