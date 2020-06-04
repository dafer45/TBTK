/* Copyright 2020 Kristofer Björnson
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
 *  @file SlaterKosterState.h
 *  @brief Slater-Koster @link AbstractState State@endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SLATER_KOSTER_STATE
#define COM_DAFER45_TBTK_SLATER_KOSTER_STATE

#include "TBTK/AbstractState.h"
#include "TBTK/DefaultOperator.h"
#include "TBTK/Index.h"
#include "TBTK/IndexTree.h"

#include <complex>
#include <tuple>

namespace TBTK{

/** @brief Slater-Koster @link AbstractState State@endlink. */
class SlaterKosterState : public AbstractState{
public:
	/** The RadialFunction provides an interface for implementing the
	 *  radial behavior of the Slater-Koster parameters. The derived
	 *  class must implement clone() and operator(). */
	class RadialFunction{
	public:
		/** Enum class for specifying the orbital type. */
		enum class Orbital{s, p, d};

		/** Enum class for specifying the bond type. */
		enum class Bond{Sigma, Pi, Delta};

		/** Destructor. */
		virtual ~RadialFunction();

		/** The derived class should return a clone of the
		 *  corresponding object.
		 *
		 *  @return A pointer to a clone of the object. */
		virtual RadialFunction* clone() const = 0;

		/** Function call operator. The derived class must implement
		 *  this function to be possible to use together with a
		 *  SlaterKosterState. The function should return the
		 *  Slater-Koster parameter \f$V_{o_1o_2b}\f$, where \f$o_1\f$
		 *  and \f$o_2\f$ are orbital indices and \f$b\f$ is a bond
		 *  type.
		 *
		 *  @param distance The distance between the two orbital
		 *  centers.
		 *
		 *  @param orbital0 The orbital type of the first state.
		 *  @param orbital1 The orbital type of the second state.
		 *  @param bond The bond type. */
		virtual std::complex<double> operator()(
			double distance,
			Orbital orbital0,
			Orbital orbital1,
			Bond bond
		) const = 0;

		/** Get on-site term. The derived class must implement this
		 *  function to be possible to use together with a
		 *  SlaterKosterState. The function should return the on-site
		 *  energies for the s, p, eg, and t2g basis functions.
		 *
		 *  @param orbital The orbital to return the value for. The
		 *  implementing class should accept the inputs "s", "p", "d",
		 *  and "t2g".
		 *
		 *  @return The on-site energy for the given orbital. */
		virtual std::complex<double> getOnSiteTerm(
			Orbital orbital
		) const = 0;
	};

	SlaterKosterState();

	/** Constructor. */
	SlaterKosterState(
		const Vector3d &position,
		const std::string &orbital,
		const RadialFunction &radialFunction
	);

	/** Copy constructor. */
	SlaterKosterState(const SlaterKosterState &slaterKosterState);

	/** Destructor. */
	virtual ~SlaterKosterState();

	/** Assignment operator. */
	SlaterKosterState& operator=(const SlaterKosterState &rhs);

	/** Implements AbstracState::clone(). */
	virtual SlaterKosterState* clone() const;

	/** Implements AbstractState::getOverlapWith(). */
	virtual std::complex<double> getOverlap(const AbstractState &bra) const;

	/** Implements AbstractState::getMatrixElementWith(). */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &bra,
		const AbstractOperator &o = DefaultOperator()
	) const;
private:
	/** Enum class for specifying the orbital type. */
	enum class Orbital{s, x, y, z, xy, yz, zx, x2my2, z2mr2};

	/** Position. */
	Vector3d position;

	/** Orbital. */
	Orbital orbital;

	/** Radial function. */
	RadialFunction *radialFunction;

	/** Constructor. */
	SlaterKosterState(
		const Vector3d &position,
		Orbital orbital,
		const RadialFunction &radialFunction
	);

	/** Convert string to orbital. */
	static Orbital getOrbital(const std::string &orbital);
};

};	//End of namespace TBTK

#endif
/// @endcond
