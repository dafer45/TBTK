/* Copyright 2018 Kristofer Björnson
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
 *  @file EnergyResolvedProperty.h
 *  @brief Base class for energy resolved Properties.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_ENERGY_RESOLVED_PROPERTY
#define COM_DAFER45_TBTK_PROPERTY_ENERGY_RESOLVED_PROPERTY

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/Range.h"
#include "TBTK/TBTKMacros.h"

#include <cmath>
#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Base class for energy resolved Properties.
 *
 *  The EnergyResolvedProperty is a base class for energy dependent @link
 *  AbstractProperty Properties@endlink. For more information about specific
 *  EnergyDependentProperties, see:
 *
 *  - DOS
 *  - GreensFunction
 *  - LDOS
 *  - SpinPolarizedLDOS
 *
 *  # Block structure
 *  If the EnergyResolvedProperty has no Index structure, it contains a single
 *  block, otherwise it contains one block per Index. Inside of a block, values
 *  are stored for a range of energies. See AbstractProperty for more
 *  information about blocks.
 *
 *  # EnergyType
 *  The EnergyResolvedProperty can have one of three different energy types.
 *  Depending on the EnergyType, different member functions are usefull. It is
 *  possible to get the EnergyType using
 *  ```cpp
 *      EnergyResolvedProperty<DataType>::EnergyType energyType
 *          = energyResolvedProperty.getEnergyType();
 *  ```
 *
 *  ## EnergyResolvedProperty<DataType>::EnergyType::Real
 *  The data is stored for a range of real energies on a regular grid of on an
 *  interval [lowerBound, upperBound]. It is possible to get the lower and
 *  upper bound and the number of points with which the energy grid is resolved
 *  using
 *  ```cpp
 *      double lowerBound = energyResolvedProperty.getLowerBound();
 *      double upperBound = energyResolvedProperty.getUpperBound();
 *      unsigned int resolution = energyResolvedProperty.getResolution();
 *  ```
 *  It is also possible to get the distance \f$\Delta E\f$ between neighboring
 *  points and the energy for specific points on the grid using
 *  ```cpp
 *      double dE = energyResolvedProperty.getDeltaE();
 *      double E = energyResolvedProperty.getEnergy(n);
 *  ```
 *
 *  ## EnergyResolvedProperty<DataType>::EnergyType::FermionicMatsubara
 *  ## EnergyResolvedProperty<DataType>::EnergyType::BosonicMatsubara
 *  Not yet publicly supported. */
template<typename DataType>
class EnergyResolvedProperty : public AbstractProperty<DataType>{
public:
	/** Enum class for specifying the energy type. */
	enum class EnergyType{Real, FermionicMatsubara, BosonicMatsubara};

	/** Constructs an uninitialized EnergyResolvedProperty. */
	EnergyResolvedProperty();

	/** Copy constructor.
	 *
	 *  @param energyResolvedProperty The EnergyResolvedProperty to copy.
	 */
	EnergyResolvedProperty(
		const EnergyResolvedProperty &energyResolvedProperty
	);

	/** Constructs an EnergyResolvedProperty with real energies on the None
	 *  format. [See AbstractProperty for detailed information about the
	 *  None format.]
	 *
	 *  @param energyWindow The energy window over which the
	 *  EnergyResolvedProperty is defined. */
	EnergyResolvedProperty(const Range &energyWindow);

	/** Constructs an EnergyResolvedProperty with real energies on the None
	 *  format. [See AbstractProperty for detailed information about the
	 *  None format.]
	 *
	 *  @param energyWindow The energy window over which the
	 *  EnergyResolvedProperty is defined. */
	EnergyResolvedProperty(
		const Range &energyWindow,
		const DataType *data
	);

	/** Construct an EnergyResolvedProperty with real energies on the
	 *  Ranges format. [See AbstractProperty for detailed information about
	 *  the Ranges format.]
	 *
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param energyWindow The energy window over which the
	 *  ENergyResolvedProperty is defined. */
	EnergyResolvedProperty(
		const std::vector<int> &ranges,
		const Range &energyWindow
	);

	/** Construct an EnergyResolvedProperty with real energies on the
	 *  Ranges format. [See AbstractProperty for detailed information about
	 *  the Ranges format.]
	 *
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param energyWindow The energy window over which the
	 *  EnergyResolvedProperty is defined.
	 *
	 *  @param data Raw data to initialize the EnergyResolvedProperty with.
	 */
	EnergyResolvedProperty(
		const std::vector<int> &ranges,
		const Range &energyWindow,
		const DataType *data
	);

	/** Constructs an EnergyResolvedProperty with real energies on the
	 *  Custom format. [See AbstractProperty for detailed information about
	 *  the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param energyWindow The energy window over which the
	 *  EnergyResolvedProperty is defined. */
	EnergyResolvedProperty(
		const IndexTree &indexTree,
		const Range &energyWindow
	);

	/** Constructs an EnergyResolvedProperty with real energie on the
	 *  Custom format and initializes it with data. [See AbstractProperty
	 *  for detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param energyWindow The energy window over which the
	 *  EnergyResolvedProperty is defined.
	 *
	 *  @param data Raw data to initialize the EnergyResolvedProperty with.
	 */
	EnergyResolvedProperty(
		const IndexTree &indexTree,
		const Range &energyWindow,
		const DataType *data
	);

	/** Constructs an EnergyResolvedProperty with Matsubara energies
	 *  on the Custom format. [See AbstractProperty for detailed
	 *  information about the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	EnergyResolvedProperty(
		EnergyType energyType,
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergtIndex,
		double fundamentalMatsubaraEnergy
	);

	/** Constructs an EnergyResolvedProperty with Matsubara energies
	 *  on the Custom format and initializes it with data. [See
	 *  AbstractProperty for detailed information about the Custom format
	 *  and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the EnergyResolvedProperty with.
	 */
	EnergyResolvedProperty(
		EnergyType energyType,
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergtIndex,
		double fundamentalMatsubaraEnergy,
		const DataType *data
	);

	/** Constructor. Constructs the EnergyResolvedProperty from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the EnergyResolvedProperty. */
	EnergyResolvedProperty(
		const std::string &serialization,
		Serializable::Mode mode
	);

	/** Assignment operator.
	 *
	 *  @rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after assignment. */
	EnergyResolvedProperty& operator=(const EnergyResolvedProperty &rhs);

	/*** Get energy type.
	 *
	 *  @return The EnergyType. */
	EnergyType getEnergyType() const;

	/** Get lower bound for the energy.
	 *
	 *  @return Lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy.
	 *
	 *  @return Upper bound for the energy. */
	double getUpperBound() const;

	/** Get the energy resolution (number of points used for the energy
	 *  axis).
	 *
	 *  @return The energy resolution. */
	unsigned int getResolution() const;

	/** Get the step length \f$\Delta E\f$ between neigboring energy
	 *  indices.
	 *
	 *  @return The energy step length \f$\Delta\f$. */
	double getDeltaE() const;

	/** Get the nth energy value.
	 *
	 *  @param n The energy index to get the energy for.
	 *
	 *  @return The energy for the nth energy index. */
	double getEnergy(unsigned int n) const;

	/** Get the lower Matsubara energy index. That is, l in the expression
	 *  E = (l + 2*n)*E_0.
	 *
	 *  @return The lowest Matsubara energy index. */
	int getLowerMatsubaraEnergyIndex() const;

	/** Get the upper Matsubara energy index. That is, l+N-1, where l is
	 *  the lowest Matsubara energy index and N is the number of Matsubara
	 *  energies.
	 *
	 *  @return The largest Matsubara energy index. */
	int getUpperMatsubaraEnergyIndex() const;

	/** Get the number of Matsubara energies.
	 *
	 *  @return The number of Matsubara energies. */
	unsigned int getNumMatsubaraEnergies() const;

	/** Get the fundamental Matsubara energy E_0 in the expression
	 *  E = (l + 2*n)*E_0. */
	double getFundamentalMatsubaraEnergy() const;

	/** Get the lower Matsubara energy.
	 *
	 *  @return The lowest Matsubara energy. */
	double getLowerMatsubaraEnergy() const;

	/** Get the upper Matsubara energy.
	 *
	 *  @return The highest Matsubara energy. */
	double getUpperMatsubaraEnergy() const;

	/** Get the nth Matsubara energy. */
	std::complex<double> getMatsubaraEnergy(unsigned int n) const;

	/** Get the number of energies. The same as
	 *  EnergyResolvedProperty::getResolution() for EnergyType::Real and
	 *  EnergyResolvedProperty::getNumMatsubaraEnergies() for
	 *  EnergyType::FermionicMatsubara and EnergyTpe::BosnicMatsubara.
	 *
	 *  @return The number of energies. */
	unsigned int getNumEnergies() const;

	/** Check whether the energy windows are equal. The energy windows are
	 *  considered equal if the number of energies are equal and the upper
	 *  and lower bounds (upper and lower Matsubara frequencies) are the
	 *  same within a given relative precision.
	 *
	 *  @param energyResolvedProperty The EnergyResolvedProperty to compare
	 *  with.
	 *
	 *  @param precision The required precision with which the bounds must
	 *  agree. A value of 1 means that the difference between the bounds
	 *  must be no larger than the difference between two succesive energy
	 *  points.
	 *
	 *  @return True if the energy windows are equal, otherwise false. */
	bool energyWindowsAreEqual(
		const EnergyResolvedProperty &energyResolvedProperty,
		double precision = 1e-1
	) const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Serializable::Mode mode) const;
protected:
	/** Overrides AbstractProperty::operator+=(). */
	EnergyResolvedProperty& operator+=(const EnergyResolvedProperty &rhs);

	/** Overrides AbstractProperty::operator-=(). */
	EnergyResolvedProperty& operator-=(const EnergyResolvedProperty &rhs);

	/** Overrides AbstractProperty::operator*=(). */
	EnergyResolvedProperty& operator*=(const DataType &rhs);

	/** Overrides AbstractProperty::operator/=(). */
	EnergyResolvedProperty& operator/=(const DataType &rhs);
private:
	/** The energy type for the property. */
	EnergyType energyType;

	class RealEnergy{
	public:
		Range energyWindow;
	};

	class MatsubaraEnergy{
	public:
		/** The lowest Matsubara energy index l in the Expression . */
		int lowerMatsubaraEnergyIndex;

		/** The number of Matsubara energies. */
		int numMatsubaraEnergies;

		/** The energy E_0 in the expression E = (l + 2*n)*E_0. */
		double fundamentalMatsubaraEnergy;
	};

	/** Union of energy descriptors. */
	union EnergyDescriptor{
		EnergyDescriptor(){}

		RealEnergy realEnergy;
		MatsubaraEnergy matsubaraEnergy;
	};

	/** The actual energy descriptor. */
	EnergyDescriptor descriptor;
};

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(){
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const EnergyResolvedProperty &energyResolvedProperty
) :
	AbstractProperty<DataType>(energyResolvedProperty)
{
	energyType = energyResolvedProperty.energyType;
	switch(energyType){
	case EnergyType::Real:
		descriptor.realEnergy.energyWindow
			= energyResolvedProperty.descriptor.realEnergy.energyWindow;
		break;
	case EnergyType::FermionicMatsubara:
	case EnergyType::BosonicMatsubara:
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= energyResolvedProperty.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies
			= energyResolvedProperty.descriptor.matsubaraEnergy.numMatsubaraEnergies;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= energyResolvedProperty.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
		break;
	default:
		TBTKExit(
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"Unkown energy type.",
			"This should never happen, contact the developer."
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const Range &energyWindow
) :
	AbstractProperty<DataType>(energyWindow.getResolution())
{
	TBTKAssert(
		energyWindow[0] <= energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accedning.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must have a non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const Range &energyWindow,
	const DataType *data
) :
	AbstractProperty<DataType>(energyWindow.getResolution(), data)
{
	TBTKAssert(
		energyWindow[0] <= energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accending.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must have non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const std::vector<int> &ranges,
	const Range &energyWindow
) :
	AbstractProperty<DataType>(ranges, energyWindow.getResolution())
{
	TBTKAssert(
		energyWindow[0] <= energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accending.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must have non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const std::vector<int> &ranges,
	const Range &energyWindow,
	const DataType *data
) :
	AbstractProperty<DataType>(ranges, energyWindow.getResolution(), data)
{
	TBTKAssert(
		energyWindow[0] <= energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accending.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must have non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const IndexTree &indexTree,
	const Range &energyWindow
) :
	AbstractProperty<DataType>(indexTree, energyWindow.getResolution())
{
	TBTKAssert(
		energyWindow[0] <= energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accending.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must hve non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const IndexTree &indexTree,
	const Range &energyWindow,
	const DataType *data
) :
	AbstractProperty<DataType>(
		indexTree,
		energyWindow.getResolution(),
		data
	)
{
	TBTKAssert(
		energyWindow[0] < energyWindow.getLast(),
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'energyWindow' must be accending.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'energyWindow' must have non-zero resolution.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.energyWindow = energyWindow;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	EnergyType energyType,
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy
) :
	AbstractProperty<DataType>(
		indexTree,
		(upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2 + 1
	)
{
	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid Matsubara energy bounds. The"
		" 'lowerMatsubaraEnergyIndex=" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the 'upperMatsubaraEnergyIndex="
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		fundamentalMatsubaraEnergy > 0,
		"EnergyResolvedProperty::energyResolvedProperty()",
		"The 'fundamentalMatsubaraEnergy' must be larger than 0.",
		""
	);

	switch(energyType){
	case EnergyType::FermionicMatsubara:
		TBTKAssert(
			abs(lowerMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);
		TBTKAssert(
			abs(upperMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'upperMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			lowerMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);
		TBTKAssert(
			upperMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'upperMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	default:
		TBTKExit(
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'energyType' must be"
			" EnergyType::FermionicMatsubara or"
			" EnergyType::BosonicMatsubara.",
			""
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	EnergyType energyType,
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy,
	const DataType *data
) :
	AbstractProperty<DataType>(
		indexTree,
		(upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2 + 1,
		data
	)
{
	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid Matsubara energy bounds. The"
		" 'lowerMatsubaraEnergyIndex=" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the 'upperMatsubaraEnergyIndex="
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		fundamentalMatsubaraEnergy > 0,
		"EnergyResolvedProperty::energyResolvedProperty()",
		"The 'fundamentalMatsubaraEnergy' must be larger than 0.",
		""
	);

	switch(energyType){
	case EnergyType::FermionicMatsubara:
		TBTKAssert(
			abs(lowerMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);
		TBTKAssert(
			abs(upperMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'uppererMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			lowerMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);
		TBTKAssert(
			upperMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'uppererMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	default:
		TBTKExit(
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'energyType' must be"
			" EnergyType::FermionicMatsubara or"
			" EnergyType::BosonicMatsubara.",
			""
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const std::string &serialization,
	Serializable::Mode mode
) :
	AbstractProperty<DataType>(
		Serializable::extract(
			serialization,
			mode,
			"abstractProperty"
		),
		mode
	)
{
	TBTKAssert(
		Serializable::validate(serialization, "EnergyResolvedProperty", mode),
		"Property::EnergyResolvedProperty::EnergyResolvedProperty()",
		"Unable to parse string as EnergyResolvedProperty '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			std::string et = j.at("energyType").get<std::string>();
			if(et.compare("Real") == 0){
				energyType = EnergyType::Real;
				descriptor.realEnergy.energyWindow = Range(
					j.at("energyWindow").dump(),
					mode
				);
			}
			else if(et.compare("FermionicMatsubara") == 0){
				energyType = EnergyType::FermionicMatsubara;
				descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
					= j.at("lowerMatsubaraEnergyIndex");
				descriptor.matsubaraEnergy.numMatsubaraEnergies
					= j.at("numMatsubaraEnergies");
				descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
					= j.at("fundamentalMatsubaraEnergy");
			}
			else if(et.compare("BosonicMatsubara") == 0){
				energyType = EnergyType::BosonicMatsubara;
				descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
					= j.at("lowerMatsubaraEnergyIndex");
				descriptor.matsubaraEnergy.numMatsubaraEnergies
					= j.at("numMatsubaraEnergies");
				descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
					= j.at("fundamentalMatsubaraEnergy");
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"Proerty::EnergyResolvedProperty::EnergyResolvedProperty()",
				"Unable to parse string as"
				<< " EnergyResolvedProperty '" << serialization
				<< "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::EnergyResolvedProperty()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>& EnergyResolvedProperty<DataType>::operator=(
	const EnergyResolvedProperty &rhs
){
	if(this != &rhs){
		AbstractProperty<DataType>::operator=(rhs);
		energyType = rhs.energyType;
		switch(energyType){
		case EnergyType::Real:
			descriptor.realEnergy.energyWindow
				= rhs.descriptor.realEnergy.energyWindow;
			break;
		case EnergyType::FermionicMatsubara:
		case EnergyType::BosonicMatsubara:
			descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
				= rhs.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
			descriptor.matsubaraEnergy.numMatsubaraEnergies
				= rhs.descriptor.matsubaraEnergy.numMatsubaraEnergies;
			descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
				= rhs.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
			break;
		default:
			TBTKExit(
				"EnergyResolvedProperty::operator=()",
				"Unkown energy type.",
				"This should never happen, contact the developer."
			);
		}
	}

	return *this;
}

template<typename DataType>
inline typename EnergyResolvedProperty<DataType>::EnergyType
EnergyResolvedProperty<DataType>::getEnergyType() const{
	return energyType;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getLowerBound() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"EnergyResolvedProperty::getLowerBound()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.energyWindow[0];
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getUpperBound() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"EnergyResolvedProperty::getUpperBound()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.energyWindow.getLast();
}

template<typename DataType>
inline unsigned int EnergyResolvedProperty<DataType>::getResolution() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"EnergyResolvedProperty::getResolution()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.energyWindow.getResolution();
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getDeltaE() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"EnergyResolvedProperty::getDeltaE()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	double dE;
	if(descriptor.realEnergy.energyWindow.getResolution() == 1)
		dE = 0;
	else
		dE = (
			descriptor.realEnergy.energyWindow.getLast()
			- descriptor.realEnergy.energyWindow[0]
		)/(descriptor.realEnergy.energyWindow.getResolution() - 1);

	return dE;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getEnergy(
	unsigned int n
) const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"EnergyResolvedProperty::getEnergy()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.energyWindow[n];
}

template<typename DataType>
inline int EnergyResolvedProperty<DataType>::getLowerMatsubaraEnergyIndex(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getLowerMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
}

template<typename DataType>
inline int EnergyResolvedProperty<DataType>::getUpperMatsubaraEnergyIndex(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getUpperMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
	 + 2*(descriptor.matsubaraEnergy.numMatsubaraEnergies - 1);
}

template<typename DataType>
inline unsigned int EnergyResolvedProperty<DataType>::getNumMatsubaraEnergies() const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getNumMatsubaraEnergies()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.numMatsubaraEnergies;
}

template<typename DataType>
inline double EnergyResolvedProperty<
	DataType
>::getFundamentalMatsubaraEnergy() const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getFundamentalMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getLowerMatsubaraEnergy(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getLowerMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
		*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getUpperMatsubaraEnergy(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getUpperMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return (
			descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			+ 2*(descriptor.matsubaraEnergy.numMatsubaraEnergies-1)
		)*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline std::complex<double> EnergyResolvedProperty<
	DataType
>::getMatsubaraEnergy(
	unsigned int n
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"EnergyResolvedProperty::getMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return std::complex<double>(
		0,
		(descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex + 2*(int)n)
		*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
	);
}

template<typename DataType>
inline unsigned int EnergyResolvedProperty<DataType>::getNumEnergies() const{
	switch(energyType){
	case EnergyType::Real:
		return getResolution();
	case EnergyType::FermionicMatsubara:
	case EnergyType::BosonicMatsubara:
		return getNumMatsubaraEnergies();
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::getNumEnergies()",
			"Unknown energy type.",
			"This should never happen, contact the developer."
		);
	}
}

template<typename DataType>
inline bool EnergyResolvedProperty<DataType>::energyWindowsAreEqual(
	const EnergyResolvedProperty &energyResolvedProperty,
	double precision
) const{
	if(energyType != energyResolvedProperty.energyType)
		return false;

	if(getNumEnergies() != energyResolvedProperty.getNumEnergies())
		return false;

	switch(energyType){
	case EnergyType::Real:
	{
		double lowerBound0 = getLowerBound();
		double upperBound0 = getUpperBound();
		double lowerBound1 = energyResolvedProperty.getLowerBound();
		double upperBound1 = energyResolvedProperty.getUpperBound();
		double dE = getDeltaE();

		if(std::abs(lowerBound0 - lowerBound1) > precision*dE)
			return false;
		if(std::abs(upperBound0 - upperBound1) > precision*dE)
			return false;

		return true;
	}
	case EnergyType::FermionicMatsubara:
	case EnergyType::BosonicMatsubara:
	{
		double lowerMatsubaraEnergy0 = getLowerMatsubaraEnergy();
		double upperMatsubaraEnergy0 = getUpperMatsubaraEnergy();
		double lowerMatsubaraEnergy1
			= energyResolvedProperty.getLowerMatsubaraEnergy();
		double upperMatsubaraEnergy1
			= energyResolvedProperty.getUpperMatsubaraEnergy();
		double dE;
		if(getNumEnergies() > 1)
			dE = imag(getMatsubaraEnergy(1) - getMatsubaraEnergy(0));
		else
			dE = 0;

		if(
			std::abs(lowerMatsubaraEnergy0 - lowerMatsubaraEnergy1)
				> precision*dE
		){
			return false;
		}
		if(
			std::abs(upperMatsubaraEnergy0 - upperMatsubaraEnergy1)
			> precision*dE
		){
			return false;
		}

		return true;
	}
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::energyWindowsAreEqual()",
			"Unknown energy type.",
			"This should never happen, contact the developer."
		);
	}
}

template<typename DataType>
inline std::string EnergyResolvedProperty<DataType>::serialize(
	Serializable::Mode mode
) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "EnergyResolvedProperty";
		switch(energyType){
		case EnergyType::Real:
			j["energyType"] = "Real";
			j["energyWindow"] = nlohmann::json::parse(
				descriptor.realEnergy.energyWindow.serialize(mode)
			);

			break;
		case EnergyType::FermionicMatsubara:
			j["energyType"] = "FermionicMatsubara";
			j["lowerMatsubaraEnergyIndex"]
				= descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
			j["numMatsubaraEnergies"]
				= descriptor.matsubaraEnergy.numMatsubaraEnergies;
			j["fundamentalMatsubaraEnergy"]
				= descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;

			break;
		case EnergyType::BosonicMatsubara:
			j["energyType"] = "BosonicMatsubara";
			j["lowerMatsubaraEnergyIndex"]
				= descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
			j["numMatsubaraEnergies"]
				= descriptor.matsubaraEnergy.numMatsubaraEnergies;
			j["fundamentalMatsubaraEnergy"]
				= descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;

			break;
		default:
			TBTKExit(
				"Property::EnergyResolvedProperty::serialize()",
				"Unknown EnergyType.",
				"This should never happen, contact the developer."
			);
		}
		j["abstractProperty"] = nlohmann::json::parse(
			AbstractProperty<DataType>::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
inline EnergyResolvedProperty<DataType>&
EnergyResolvedProperty<DataType>::operator+=(
	const EnergyResolvedProperty<DataType> &rhs
){
	TBTKAssert(
		energyType == rhs.energyType,
		"Property::EnergyResolvedProperty::operator+=()",
		"Incompatible energy types.",
		""
	);

	switch(energyType){
	case EnergyType::Real:
		TBTKAssert(
			descriptor.realEnergy.energyWindow
				== rhs.descriptor.realEnergy.energyWindow,
			"Property::EnergyResolvedProperty::operator+=()",
			"Incompatible energy bounds.",
			""
		);
		break;
	case EnergyType::FermionicMatsubara:
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
				== rhs.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex,
			"Property::EnergyResolvedProperty::operator+=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has lower Matsubara energy index '"
			<< descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			<< "', while the right hand side has lower Matsubara"
			<< " energy index '"
			<< rhs.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex << "'.",
			""
		);
		TBTKAssert(
			descriptor.matsubaraEnergy.numMatsubaraEnergies
				== rhs.descriptor.matsubaraEnergy.numMatsubaraEnergies,
			"Property::EnergyResolvedProperty::operator+=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has '"
			<< descriptor.matsubaraEnergy.numMatsubaraEnergies
			<< "' number of Matsubara energies, while the right"
			<< " hand side has '"
			<< rhs.descriptor.matsubaraEnergy.numMatsubaraEnergies
			<< "' number of Matsubara energies.",
			""
		);
		TBTKAssert(
			descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
				== rhs.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy,
			"Property::EnergyResolvedProperty::operator+=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has fundamental Matsubara energy '"
			<< descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			<< "', while the right hand side has fundamental Matsubara"
			<< " energy '"
			<< rhs.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			<< "'.",
			""
		);
	}

	AbstractProperty<DataType>::operator+=(rhs);

	return *this;
}

template<typename DataType>
inline EnergyResolvedProperty<DataType>&
EnergyResolvedProperty<DataType>::operator-=(
	const EnergyResolvedProperty<DataType> &rhs
){
	TBTKAssert(
		energyType == rhs.energyType,
		"Property::EnergyResolvedProperty::operator-=()",
		"Incompatible energy types.",
		""
	);

	switch(energyType){
	case EnergyType::Real:
		TBTKAssert(
			descriptor.realEnergy.energyWindow
				== rhs.descriptor.realEnergy.energyWindow,
			"Property::EnergyResolvedProperty::operator-=()",
			"Incompatible energy bounds.",
			""
		);
		break;
	case EnergyType::FermionicMatsubara:
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
				== rhs.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex,
			"Property::EnergyResolvedProperty::operator-=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has lower Matsubara energy index '"
			<< descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			<< "', while the right hand side has lower Matsubara"
			<< " energy index '"
			<< rhs.descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex << "'.",
			""
		);
		TBTKAssert(
			descriptor.matsubaraEnergy.numMatsubaraEnergies
				== rhs.descriptor.matsubaraEnergy.numMatsubaraEnergies,
			"Property::EnergyResolvedProperty::operator-=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has '"
			<< descriptor.matsubaraEnergy.numMatsubaraEnergies
			<< "' number of Matsubara energies, while the right"
			<< " hand side has '"
			<< rhs.descriptor.matsubaraEnergy.numMatsubaraEnergies
			<< "' number of Matsubara energies.",
			""
		);
		TBTKAssert(
			descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
				== rhs.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy,
			"Property::EnergyResolvedProperty::operator-=()",
			"Incompatible Matsubara energies. The left hand side"
			<< " has fundamental Matsubara energy '"
			<< descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			<< "', while the right hand side has fundamental Matsubara"
			<< " energy '"
			<< rhs.descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			<< "'.",
			""
		);
	}

	AbstractProperty<DataType>::operator-=(rhs);

	return *this;
}

template<typename DataType>
inline EnergyResolvedProperty<DataType>&
EnergyResolvedProperty<DataType>::operator*=(
	const DataType &rhs
){
	AbstractProperty<DataType>::operator*=(rhs);

	return *this;
}

template<typename DataType>
inline EnergyResolvedProperty<DataType>&
EnergyResolvedProperty<DataType>::operator/=(
	const DataType &rhs
){
	AbstractProperty<DataType>::operator/=(rhs);

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
