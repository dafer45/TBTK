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

/** @file Temperature.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Temperature.h"

using namespace std;

namespace TBTK{
namespace Quantity{

/*map<Temperature::Unit, string> Temperature::unitToString = {
	{Temperature::Unit::kK, "kK"},
	{Temperature::Unit::K, "K"},
	{Temperature::Unit::mK, "mK"},
	{Temperature::Unit::uK, "uK"},
	{Temperature::Unit::nK, "nK"}
};
map<string, Temperature::Unit> Temperature::stringToUnit = {
	{"kK", Temperature::Unit::kK},
	{"K", Temperature::Unit::K},
	{"mK", Temperature::Unit::mK},
	{"uK", Temperature::Unit::uK},
	{"nK", Temperature::Unit::nK}
};

std::string Temperature::getUnitString(Unit unit){
	try{
		return unitToString.at(unit);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Temperature::getUnitString()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			"This should never happen, contact the developer."
		);
	}
}

Temperature::Unit Temperature::getUnit(const string &str){
	try{
		return stringToUnit.at(str);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Temperature::getUnit()",
			"Unknown unit '" << str << "'.",
			""
		);
	}
}

double Temperature::getConversionFactor(Unit unit){
	switch(unit){
		case Unit::kK:	//1e-3 kK per K
			return 1e-3;
		case Unit::K:	//Reference scale
			return 1.;
		case Unit::mK:
			return 1e3;
		case Unit::uK:
			return 1e6;
		case Unit::nK:
			return 1e9;
		default:
			TBTKExit(
				"Quantity::Temperature::getConversionUnit()",
				"Unknown unit - " << static_cast<int>(unit)
				<< ".",
				"This should never happen, contact the"
				<< " developer."
			);
	}
}*/

};	//End of namespace Quantity
};	//End of namespace TBTK
