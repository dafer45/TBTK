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
 *  @brief Derived.
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

/** @brief Derived.
 *
 *  Derived provides the means for defining derived Quantities.
 */
template<typename Units, typename Exponents>
class Derived : public Quantity<Units, Exponents>{
public:
	using IsBaseQuantity = std::false_type;
	using Quantity<Units, Exponents>::Quantity;
	using Quantity<Units, Exponents>::getConversionFactor;
	using Unit = typename Quantity<Units, Exponents>::Unit;
	using Exponent = typename Quantity<Units, Exponents>::Exponent;

	/** Get the conversion factor for converting from the reference unit to
	 *  the given units. */
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
typedef Derived<MagnetonUnit, MagnetonExponent> Magneton;

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

