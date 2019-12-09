/* Copyright 2017 Kristofer Björnson
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
 *  @file ParallelepipedArrayState.h
 *  @brief State class with parallelepiped array based overlap evaluation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARALLELEPIPED_ARRAY_STATE
#define COM_DAFER45_TBTK_PARALLELEPIPED_ARRAY_STATE

#include "TBTK/ArrayState.h"
#include "TBTK/Field.h"
#include "TBTK/ParallelepipedCell.h"

namespace TBTK{

class ParallelepipedArrayState : public ArrayState, public Field<std::complex<double>, double>{
public:
	/** Constructor. */
	ParallelepipedArrayState(
		const std::vector<std::vector<double>> &basisVectors,
		std::initializer_list<unsigned int> numMeshPoints
	);

	/** Destructor. */
	virtual ~ParallelepipedArrayState();

	/** Implements AbstracState::clone(). */
	virtual ParallelepipedArrayState* clone() const;

	/** Implements AbstractState::getOverlapWith(). */
	virtual std::complex<double> getOverlap(const AbstractState &bra) const;

	/** Implements AbstractState::getMatrixElementWith(). */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &bra,
		const AbstractOperator &o = DefaultOperator()
	) const;

	/** Set amplitude. */
	void setAmplitude(
		std::complex<double> amplitude,
		std::initializer_list<double> coordinate
	);

	/** Get amplitude. */
	const std::complex<double>& getAmplitude(
		std::initializer_list<double> coordinate
	) const;

	/** Get amplitude. */
	const std::complex<double>& getAmplitude(
		const std::vector<double> &coordinate
	) const;

	/** Implements Field::operator(). */
	virtual std::complex<double> operator()(std::initializer_list<double> argument) const;

	/** Implements Field::getCoordinates(). */
	virtual const std::vector<double>& getCoordinates() const;

	/** Implements Field::getExtent(). */
	virtual double getExtent() const;
private:
	/** The parallelepiped that determines the indexing into the underlying
	 *   ArrayState storage. */
	ParallelepipedCell parallelepiped;
};

};	//End of namespace TBTK

#endif
/// @endcond
