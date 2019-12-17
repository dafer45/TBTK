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

void initializeBaseQuantities();

template<typename Units, typename Exponents>
class Base : public Quantity<Units, Exponents>{
public:
	using IsBaseQuantity = std::true_type;
	using Quantity<Units, Exponents>::Quantity;
};

//Charge
enum class ChargeUnit{kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e};
enum class ChargeExponent{
	Charge = 1,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
typedef Base<ChargeUnit, ChargeExponent> Charge;

//Count
enum class CountUnit{pcs, mol};
enum class CountExponent{
	Charge = 0,
	Count = 1,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
typedef Base<CountUnit, CountExponent> Count;

//Energy
enum class EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};
enum class EnergyExponent{
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = 0,
	Temperature = 0,
	Time = 0
};
typedef Base<EnergyUnit, EnergyExponent> Energy;

//Length
enum class LengthUnit{m, mm, um, nm, pm, fm, am, Ao};
enum class LengthExponent{
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 1,
	Temperature = 0,
	Time = 0
};
typedef Base<LengthUnit, LengthExponent> Length;

//Temperature
enum class TemperatureUnit{kK, K, mK, uK, nK};
enum class TemperatureExponent{
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 1,
	Time = 0
};
typedef Base<TemperatureUnit, TemperatureExponent> Temperature;

//Time
enum class TimeUnit{s, ms, us, ns, ps, fs, as};
enum class TimeExponent{
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 1
};
typedef Base<TimeUnit, TimeExponent> Time;

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

