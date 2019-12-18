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
 *  @file Base.h
 *  @brief Base Quantity.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_BASE
#define COM_DAFER45_TBTK_QUANTITY_BASE

#include "TBTK/Quantity/Quantity.h"

namespace TBTK{
namespace Quantity{

/** Initialize the Base Quantities. */
void initializeBaseQuantities();

/** @brief Base Quantity.
 *
 *  The Base Quantity is a Quantity with the compile time directive
 *  IsBaseQuantity set to std::true_type to differentiate it from Derived
 *  Quantities. The Base Quantity is instantiated by the seven Base Quantities
 *  Angle, Charge, Count, Energy, Length, Temperature, and Time. For more
 *  information, see Quantity and the individual typedefs below.
 */
template<typename Units, typename Exponents>
class Base : public Quantity<Units, Exponents>{
public:
	using IsBaseQuantity = std::true_type;
	using Quantity<Units, Exponents>::Quantity;
private:
	friend void initializeBaseQuantities();
};

//Angle
enum class AngleUnit{rad, degree};
enum class AngleExponent{
	Angle = 1,
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
/** @relates Base
 *  The Quantity::Angle is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Angle::Unit::degree
 *  - Quantity::Angle::Unit::rad */
typedef Base<AngleUnit, AngleExponent> Angle;

//Charge
enum class ChargeUnit{kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e};
enum class ChargeExponent{
	Angle = 0,
	Charge = 1,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
/** @relates Base
 *  The Quantity::Charge is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Charge::Unit::kC
 *  - Quantity::Charge::Unit::C
 *  - Quantity::Charge::Unit::mC
 *  - Quantity::Charge::Unit::uC
 *  - Quantity::Charge::Unit::nC
 *  - Quantity::Charge::Unit::pC
 *  - Quantity::Charge::Unit::fC
 *  - Quantity::Charge::Unit::aC
 *  - Quantity::Charge::Unit::Te
 *  - Quantity::Charge::Unit::Ge
 *  - Quantity::Charge::Unit::Me
 *  - Quantity::Charge::Unit::ke
 *  - Quantity::Charge::Unit::e */
typedef Base<ChargeUnit, ChargeExponent> Charge;

//Count
enum class CountUnit{pcs, mol};
enum class CountExponent{
	Angle = 0,
	Charge = 0,
	Count = 1,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
/** @relates Base
 *  The Quantity::Count is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Count::Unit::pcs
 *  - Quantity::Count::Unit::mol */
typedef Base<CountUnit, CountExponent> Count;

//Energy
enum class EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};
enum class EnergyExponent{
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = 0,
	Temperature = 0,
	Time = 0
};
/** @relates Base
 *  The Quantity::Energy is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Energy::Unit::GeV
 *  - Quantity::Energy::Unit::MeV
 *  - Quantity::Energy::Unit::keV
 *  - Quantity::Energy::Unit::eV
 *  - Quantity::Energy::Unit::meV
 *  - Quantity::Energy::Unit::ueV
 *  - Quantity::Energy::Unit::J */
typedef Base<EnergyUnit, EnergyExponent> Energy;

//Length
enum class LengthUnit{m, mm, um, nm, pm, fm, am, Ao};
enum class LengthExponent{
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 1,
	Temperature = 0,
	Time = 0
};
/** @relates Base
 *  The Quantity::Length is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Length::Unit::m
 *  - Quantity::Length::Unit::mm
 *  - Quantity::Length::Unit::um
 *  - Quantity::Length::Unit::nm
 *  - Quantity::Length::Unit::pm
 *  - Quantity::Length::Unit::fm
 *  - Quantity::Length::Unit::am
 *  - Quantity::Length::Unit::Ao */
typedef Base<LengthUnit, LengthExponent> Length;

//Temperature
enum class TemperatureUnit{kK, K, mK, uK, nK};
enum class TemperatureExponent{
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 1,
	Time = 0
};
/** @relates Base
 *  The Quantity::Temperature is a Quantity::Base with the following predefined
 *  base units
 *  - Quantity::Temperature::Unit::kK
 *  - Quantity::Temperature::Unit::K
 *  - Quantity::Temperature::Unit::mK
 *  - Quantity::Temperature::Unit::uK
 *  - Quantity::Temperature::Unit::nK */
typedef Base<TemperatureUnit, TemperatureExponent> Temperature;

//Time
enum class TimeUnit{s, ms, us, ns, ps, fs, as};
enum class TimeExponent{
	Angle = 0,
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 1
};
/** @relates Base
 *  The Quantity::Time is a Quantity::Base with the following predefined base
 *  units
 *  - Quantity::Time::Unit::s
 *  - Quantity::Time::Unit::ms
 *  - Quantity::Time::Unit::us
 *  - Quantity::Time::Unit::ns
 *  - Quantity::Time::Unit::ps
 *  - Quantity::Time::Unit::fs
 *  - Quantity::Time::Unit::as */
typedef Base<TimeUnit, TimeExponent> Time;

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

