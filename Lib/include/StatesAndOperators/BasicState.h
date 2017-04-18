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
 *  @file BasisState.h
 *  @brief State class with index and unit cell based overlap evaluation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BASIC_STATE
#define COM_DAFER45_TBTK_BASIC_STATE

#include "AbstractState.h"
#include "DefaultOperator.h"
#include "Index.h"

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
		/** Reference counter. */
		unsigned int referenceCounter;

		/** Overlaps between the ket corresponding to this state and the bras
		 *  index by the indices. The first index in the tuple is the intra
		 *  cell index, while the second index is the unit cell index */
		std::vector<std::tuple<std::complex<double>, Index, Index>> overlaps;

		/** Matrix elements between the ket corresponding to this state and the
		 *  bras index by the indices. The first index in the tuple is the
		 *  intra cell index, while the second index is the unit cell index */
		std::vector<std::tuple<std::complex<double>, Index, Index>> matrixElements;
	};

	Storage *storage;
};

};	//End of namespace TBTK

#endif
