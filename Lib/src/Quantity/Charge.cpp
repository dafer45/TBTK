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

map<Charge::Unit, string> Charge::unitToString = {
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

std::string Charge::getUnitString(Unit unit){
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

};	//End of namespace Quantity
};	//End of namespace TBTK
