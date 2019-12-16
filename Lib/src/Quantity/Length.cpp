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

/** @file Length.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Length.h"

using namespace std;

namespace TBTK{
namespace Quantity{

/*map<Length::Unit, string> Length::unitToString = {
	{Length::Unit::m, "m"},
	{Length::Unit::mm, "mm"},
	{Length::Unit::um, "um"},
	{Length::Unit::nm, "nm"},
	{Length::Unit::pm, "pm"},
	{Length::Unit::fm, "fm"},
	{Length::Unit::am, "am"},
	{Length::Unit::Ao, "Ao"}
};
map<string, Length::Unit> Length::stringToUnit = {
	{"m", Length::Unit::m},
	{"mm", Length::Unit::mm},
	{"um", Length::Unit::um},
	{"nm", Length::Unit::nm},
	{"pm", Length::Unit::pm},
	{"fm", Length::Unit::fm},
	{"am", Length::Unit::am},
	{"Ao", Length::Unit::Ao}
};

string Length::getUnitString(Unit unit){
	try{
		return unitToString.at(unit);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Length::getUnitString()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			"This should never happen, contact the developer."
		);
	}
}

Length::Unit Length::getUnit(const string &str){
	try{
		return stringToUnit.at(str);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Length::getUnit()",
			"Unknown unit '" << str << "'.",
			""
		);
	}
}

double Length::getConversionFactor(Unit unit){
	switch(unit){
		case Unit::m:	//Reference scale
			return 1.;
		case Unit::mm:	//1e3 mm per m
			return 1e3;
		case Unit::um:
			return 1e6;
		case Unit::nm:
			return 1e9;
		case Unit::pm:
			return 1e12;
		case Unit::fm:
			return 1e15;
		case Unit::am:
			return 1e18;
		case Unit::Ao:
			return 1e10;
		default:
			TBTKExit(
				"Quantity::Length::getConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit)
				<< ".",
				"This should never happen, contact the"
				<< " developer."
			);
	}
}*/

};	//End of namespace Quantity
};	//End of namespace TBTK
