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

/** @file Quantity.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Base.h"
#include "TBTK/Quantity/Constants.h"

using namespace std;

namespace TBTK{
namespace Quantity{

template<>
Quantity<
	AngleUnit,
	AngleExponent
>::ConversionTable Quantity<
	AngleUnit,
	AngleExponent
>::conversionTable({});

template<>
Quantity<
	ChargeUnit,
	ChargeExponent
>::ConversionTable Quantity<
	ChargeUnit,
	ChargeExponent
>::conversionTable({});

template<>
Quantity<
	CountUnit,
	CountExponent
>::ConversionTable Quantity<
	Count::Unit,
	CountExponent
>::conversionTable({});

template<>
Quantity<
	EnergyUnit,
	EnergyExponent
>::ConversionTable Quantity<
	EnergyUnit,
	EnergyExponent
>::conversionTable({});

template<>
Quantity<
	LengthUnit,
	LengthExponent
>::ConversionTable Quantity<
	LengthUnit,
	LengthExponent
>::conversionTable({});

template<>
Quantity<
	TemperatureUnit,
	TemperatureExponent
>::ConversionTable Quantity<
	TemperatureUnit,
	TemperatureExponent
>::conversionTable({});

template<>
Quantity<
	TimeUnit,
	TimeExponent
>::ConversionTable Quantity<
	TimeUnit,
	TimeExponent
>::conversionTable({});

void initializeBaseQuantities(){
	double J_per_eV = Constants::get("e");
	double pcs_per_mol = Constants::get("N_A");

	Angle::conversionTable = Angle::ConversionTable({
		{Angle::Unit::rad,	{"rad", 1}},
		{Angle::Unit::degree,	{"degree", 360/(2*M_PI)}}
	});

	Charge::conversionTable = Charge::ConversionTable({
		{Charge::Unit::kC,	{"kC",	1e-3}},
		{Charge::Unit::C,	{"C",	1}},
		{Charge::Unit::mC,	{"mC",	1e3}},
		{Charge::Unit::uC,	{"uC",	1e6}},
		{Charge::Unit::nC,	{"nC",	1e9}},
		{Charge::Unit::pC,	{"pC",	1e12}},
		{Charge::Unit::fC,	{"fC",	1e15}},
		{Charge::Unit::aC,	{"aC",	1e18}},
		{Charge::Unit::Ge,	{"Te",	1e-12/J_per_eV}},
		{Charge::Unit::Ge,	{"Ge",	1e-9/J_per_eV}},
		{Charge::Unit::Me,	{"Me",	1e-6/J_per_eV}},
		{Charge::Unit::ke,	{"ke",	1e-3/J_per_eV}},
		{Charge::Unit::e,	{"e",	1/J_per_eV}}
	});

	Count::conversionTable = Count::ConversionTable({
		{Count::Unit::pcs,	{"pcs",	1}},
		{Count::Unit::mol,	{"mol",	1/pcs_per_mol}},
	});

	Energy::conversionTable = Energy::ConversionTable({
		{Energy::Unit::GeV,	{"GeV",	1e-9}},
		{Energy::Unit::MeV,	{"MeV",	1e-6}},
		{Energy::Unit::keV,	{"keV",	1e-3}},
		{Energy::Unit::eV,	{"eV",	1}},
		{Energy::Unit::meV,	{"meV",	1e3}},
		{Energy::Unit::ueV,	{"ueV",	1e6}},
		{Energy::Unit::J,	{"J",	J_per_eV}}
	});

	Length::conversionTable = Length::ConversionTable({
		{Length::Unit::m,	{"m",	1}},
		{Length::Unit::mm,	{"mm",	1e3}},
		{Length::Unit::um,	{"um",	1e6}},
		{Length::Unit::nm,	{"nm",	1e9}},
		{Length::Unit::pm,	{"pm",	1e12}},
		{Length::Unit::fm,	{"fm",	1e15}},
		{Length::Unit::am,	{"am",	1e18}},
		{Length::Unit::Ao,	{"Ao",	1e10}}
	});

	Temperature::conversionTable = Temperature::ConversionTable({
		{Temperature::Unit::kK,	{"kK",	1e-3}},
		{Temperature::Unit::K,	{"K",	1}},
		{Temperature::Unit::mK,	{"mK",	1e3}},
		{Temperature::Unit::uK,	{"uK",	1e6}},
		{Temperature::Unit::nK,	{"nK",	1e9}}
	});

	Time::conversionTable = Time::ConversionTable({
		{Time::Unit::s,		{"s",	1}},
		{Time::Unit::ms,	{"ms",	1e3}},
		{Time::Unit::us,	{"us",	1e6}},
		{Time::Unit::ns,	{"ns",	1e9}},
		{Time::Unit::ps,	{"ps",	1e12}},
		{Time::Unit::fs,	{"fs",	1e15}},
		{Time::Unit::as,	{"as",	1e18}}
	});
}

};	//End of namespace Quantity
};	//End of namespace TBTK
