/* Copyright 2019 Kristofer Björnson
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
 *  @file Derived.h
 *  @brief Derived Quantity.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_DERIVED
#define COM_DAFER45_TBTK_QUANTITY_DERIVED

#include "TBTK/Quantity/Base.h"
#include "TBTK/Quantity/Quantity.h"
#include "TBTK/TBTKMacros.h"

#include <iostream>
#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

/** Initialize the Derived Quantitites. */
void initializeDerivedQuantities();

/** @brief Derived Quantity.
 *
 *  The Derived Quantity is a Quantity with the compile time directive
 *  IsBaseQuantity set to std::true_false to differentiate it from Base
 *  Quantities. The Derived extends the Quantity with functions for getting the
 *  conversion factor and exponents. For more information, see Quantity and the
 *  individual typedefs below.
 *
 *  # Get conversion factor
 *  Since the Derived unit is defined in terms of Base units, the conversion
 *  factor from the reference unit is dependent on all of the Base units. The
 *  Derived Quantity therefore provides an additional function for retreiving
 *  the conversion factor. For example, the conversion factor between the
 *  default base unit for Mass and the units rad, C, pcs, eV, nm, K, and s can
 *  be obtained using.
 *  ```cpp
 *    double conversionFactor = Quantity::Mass::getConversionFactor(
 *      Quantity::Angle::Unit::rad,
 *      Quantity::Charge::Unit::C,
 *      Quantity::Count::Unit::pcs,
 *      Quantity::Energy::Unit::eV,
 *      Quantity::Length::Unit::nm,
 *      Quantity::Temperature::Unit::K,
 *      Quantity::Time::Unit::s
 *    );
 *  ```
 *
 *  # Get exponent
 *  The Derived units are product of exponents of the Base unit. For example,
 *  Velocity (m/s) corresponds to the exponents 1 and -1 for Length and Time.
 *  The exponents can be obtained as follows.
 *  ```cpp
 *    int lengthExponent = Quantity::Velocity::getExponent(Quantity::Length);
 *    int timeExponent = Quantity::Velocity::getExponent(Quantity::Time);
 *  ``` */
template<typename Units, typename Exponents>
class Derived : public Quantity<Units, Exponents>{
public:
	using IsBaseQuantity = std::false_type;
	using Quantity<Units, Exponents>::Quantity;
	using Quantity<Units, Exponents>::getConversionFactor;
	using Unit = typename Quantity<Units, Exponents>::Unit;
	using Exponent = typename Quantity<Units, Exponents>::Exponent;

	/** Get the conversion factor for converting from the reference unit to
	 *  the given units.
	 *
	 *  @param angleUnit The unit of angle to convert to.
	 *  @param chargeUnit The unit of charge to convert to.
	 *  @param countUnit The unit of charge to convert to.
	 *  @param energyUnit The unit of energy to convert to.
	 *  @param lengthUnit The unit of length to convert to.
	 *  @param temperatureUnit The unit of temeprature to convert to.
	 *  @param timeUnit The unit of time to convert to. */
	static double getConversionFactor(
		Angle::Unit angleUnit,
		Charge::Unit chargeUnit,
		Count::Unit countUnit,
		Energy::Unit energyUnit,
		Length::Unit lengthUnit,
		Temperature::Unit temperatureUnit,
		Time::Unit timeUnit
	);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Angle);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Charge);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Count);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Energy);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Length);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Temperature);

	/** Get the exponent for the given Quantity. */
	static int getExponent(Time);
private:
	friend void initializeDerivedQuantities();
};

template<typename Units, typename Exponents>
double Derived<Units, Exponents>::getConversionFactor(
	Angle::Unit angleUnit,
	Charge::Unit chargeUnit,
	Count::Unit countUnit,
	Energy::Unit energyUnit,
	Length::Unit lengthUnit,
	Temperature::Unit temperatureUnit,
	Time::Unit timeUnit
){
	return pow(
		Angle::getConversionFactor(angleUnit),
		static_cast<int>(Exponent::Angle)
	)*pow(
		Charge::getConversionFactor(chargeUnit),
		static_cast<int>(Exponent::Charge)
	)*pow(
		Count::getConversionFactor(countUnit),
		static_cast<int>(Exponent::Count)
	)*pow(
		Energy::getConversionFactor(energyUnit),
		static_cast<int>(Exponent::Energy)
	)*pow(
		Length::getConversionFactor(lengthUnit),
		static_cast<int>(Exponent::Length)
	)*pow(
		Temperature::getConversionFactor(temperatureUnit),
		static_cast<int>(Exponent::Temperature)
	)*pow(
		Time::getConversionFactor(timeUnit),
		static_cast<int>(Exponent::Time)
	);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Angle){
	return static_cast<int>(Exponents::Angle);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Charge){
	return static_cast<int>(Exponents::Charge);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Count){
	return static_cast<int>(Exponents::Count);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Energy){
	return static_cast<int>(Exponents::Energy);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Length){
	return static_cast<int>(Exponents::Length);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Temperature){
	return static_cast<int>(Exponents::Temperature);
}

template<typename Units, typename Exponents>
int Derived<Units, Exponents>::getExponent(Time){
	return static_cast<int>(Exponents::Time);
}

//Mass
enum class MassUnit {kg, g, mg, ug, ng, pg, fg, ag, u};
enum class MassExponent {
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = -2,
	Temperature = 0,
	Time = 2
};
/** @relates Derived
 *  The Quantity::Mass is a Quantity::Quantity with the following predefined
 *  derived units.
 *  - Quantity::Mass::Unit::kg
 *  - Quantity::Mass::Unit::g
 *  - Quantity::Mass::Unit::mg
 *  - Quantity::Mass::Unit::ug
 *  - Quantity::Mass::Unit::ng
 *  - Quantity::Mass::Unit::pg
 *  - Quantity::Mass::Unit::fg
 *  - Quantity::Mass::Unit::ag
 *  - Quantity::Mass::Unit::u
 *
 *  The default base unit signature is eV m^-2 s^2. */
typedef Derived<MassUnit, MassExponent> Mass;

//MagneticField
enum class MagneticFieldUnit {MT, kT, T, mT, uT, nT, GG, MG, kG, G, mG, uG};
enum class MagneticFieldExponent {
	Angle = 0,
	Charge = -1,
	Count = 0,
	Energy = 1,
	Length = -2,
	Temperature = 0,
	Time = 1
};
/** @relates Derived
 *  The Quantity::MagneticField is a Quantity::Quantity with the following
 *  predefined derived units.
 *  - Quantity::MagneticField::Unit::MT
 *  - Quantity::MagneticField::Unit::kT
 *  - Quantity::MagneticField::Unit::T
 *  - Quantity::MagneticField::Unit::mT
 *  - Quantity::MagneticField::Unit::uT
 *  - Quantity::MagneticField::Unit::nT
 *  - Quantity::MagneticField::Unit::GG
 *  - Quantity::MagneticField::Unit::MG
 *  - Quantity::MagneticField::Unit::kG
 *  - Quantity::MagneticField::Unit::G
 *  - Quantity::MagneticField::Unit::mG
 *  - Quantity::MagneticField::Unit::uG
 *
 *  The default base unit signature is C^-1 eV m^-2 s. */
typedef Derived<MagneticFieldUnit, MagneticFieldExponent> MagneticField;

//Voltage
enum class VoltageUnit {GV, MV, kV, V, mV, uV, nV};
enum class VoltageExponent {
	Angle = 0,
	Charge = -1,
	Count = 0,
	Energy = 1,
	Length = 0,
	Temperature = 0,
	Time = 0
};
/** @relates Derived
 *  The Quantity::Voltage is a Quantity::Quantity with the following predefined
 *  derived units.
 *  - Quantity::Voltage::Unit::GV
 *  - Quantity::Voltage::Unit::MV
 *  - Quantity::Voltage::Unit::kV
 *  - Quantity::Voltage::Unit::V
 *  - Quantity::Voltage::Unit::mV
 *  - Quantity::Voltage::Unit::uV
 *  - Quantity::Voltage::Unit::nV
 *
 *  The default base unit signature is C^-1 eV. */
typedef Derived<VoltageUnit, VoltageExponent> Voltage;

//Velocity
enum class VelocityUnit {};
enum class VelocityExponent {
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 1,
	Temperature = 0,
	Time = -1
};
/** @relates Derived
 *  The Quantity::Velocity is a Quantity::Quantity without any predefined
 *  units.
 *
 *  The default base unit signature is m s^-1. */
typedef Derived<VelocityUnit, VelocityExponent> Velocity;

//Planck
enum class PlanckUnit {};
enum class PlanckExponent {
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = 0,
	Temperature = 0,
	Time = 1
};
/** @relates Derived
 *  The Quantity::Planck is a Quantity::Quantity without any predefined units.
 *
 *  The default base unit signature is eV s. */
typedef Derived<PlanckUnit, PlanckExponent> Planck;

//Boltzmann
enum class BoltzmannUnit {};
enum class BoltzmannExponent{
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = 0,
	Temperature = -1,
	Time = 0
};
/** @relates Derived
 *  The Quantity::Boltzmann is a Quantity::Quantity without any predefined
 *  units.
 *
 *  The default base unit signature is eV K^-1. */
typedef Derived<BoltzmannUnit, BoltzmannExponent> Boltzmann;

//Permeability
enum class PermeabilityUnit {};
enum class PermeabilityExponent {
	Angle = 0,
	Charge = -2,
	Count = 0,
	Energy = 1,
	Length = -1,
	Temperature = 0,
	Time = 2
};
/** @relates Derived
 *  The Quantity::Permeability is a Quantity::Quantity without any predefined
 *  units.
 *
 *  The default base unit signature is C^-2 eV m^-1 s^2. */
typedef Derived<PermeabilityUnit, PermeabilityExponent> Permeability;

//Permittivity
enum class PermittivityUnit {};
enum class PermittivityExponent{
	Angle = 0,
	Charge = 2,
	Count = 0,
	Energy = -1,
	Length = -1,
	Temperature = 0,
	Time = 0
};
/** @relates Derived
 *  The Quantity::Permittivity is a Quantity::Quantity without any predefined
 *  units.
 *
 *  The default base unit signature is C^2 eV^-1 m^-1. */
typedef Derived<PermittivityUnit, PermittivityExponent> Permittivity;

//Magneton
enum class MagnetonUnit {};
enum class MagnetonExponent{
	Angle = 0,
	Charge = 1,
	Count = 0,
	Energy = 0,
	Length = 2,
	Temperature = 0,
	Time = -1
};
/** @relates Derived
 *  The Quantity::Magneton is a Quantity::Quantity without any predefined
 *  units.
 *
 *  The default base unit signature is C m^2 s^-1. */
typedef Derived<MagnetonUnit, MagnetonExponent> Magneton;

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

