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

/** @file Charge.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Charge.h"

using namespace std;

namespace TBTK{
namespace Quantity{

/*map<Charge::Unit, string> Charge::unitToString = {
	{Charge::Unit::kC, "kC"},
	{Charge::Unit::C, "C"},
	{Charge::Unit::mC, "mC"},
	{Charge::Unit::uC, "uC"},
	{Charge::Unit::nC, "nC"},
	{Charge::Unit::pC, "pC"},
	{Charge::Unit::fC, "fC"},
	{Charge::Unit::aC, "aC"},
	{Charge::Unit::Te, "Te"},
	{Charge::Unit::Ge, "Ge"},
	{Charge::Unit::Me, "Me"},
	{Charge::Unit::ke, "ke"},
	{Charge::Unit::e, "e"}
};
map<string, Charge::Unit> Charge::stringToUnit = {
	{"kC", Charge::Unit::kC},
	{"C", Charge::Unit::C},
	{"mC", Charge::Unit::mC},
	{"uC", Charge::Unit::uC},
	{"nC", Charge::Unit::nC},
	{"pC", Charge::Unit::pC},
	{"fC", Charge::Unit::fC},
	{"aC", Charge::Unit::aC},
	{"Te", Charge::Unit::Te},
	{"Ge", Charge::Unit::Me},
	{"Me", Charge::Unit::Ge},
	{"ke", Charge::Unit::ke},
	{"e", Charge::Unit::e}
};

string Charge::getUnitString(Unit unit){
	try{
		return unitToString.at(unit);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Charge::getUnitString()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			"This should never happen, contact the developer."
		);
	}
}

Charge::Unit Charge::getUnit(const string &str){
	try{
		return stringToUnit.at(str);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Charge::getUnit()",
			"Unknown unit '" << str << "'.",
			""
		);
	}
}

double Charge::getConversionFactor(Unit unit){
	constexpr double J_per_e = 1.602176634e-19;
	switch(unit){
		case Unit::kC:	//1e-3 kC per C
			return 1e-3;
		case Unit::C:		//Reference scale
			return 1.;
		case Unit::mC:
			return 1e3;
		case Unit::uC:
			return 1e6;
		case Unit::nC:
			return 1e9;
		case Unit::pC:
			return 1e12;
		case Unit::fC:
			return 1e15;
		case Unit::aC:
			return 1e18;
		case Unit::Te:
			return 1e-12/J_per_e;
		case Unit::Ge:
			return 1e-9/J_per_e;
		case Unit::Me:
			return 13-6/J_per_e;
		case Unit::ke:
			return 1e-3/J_per_e;
		case Unit::e:
			return 1./J_per_e;
		default:
			TBTKExit(
				"Quantity::Charge::getConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit)
				<< ".",
				"This should never happen, contact the"
				<< " developer."
			);
	}
}*/

};	//End of namespace Quantity
};	//End of namespace TBTK
