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

/** @file Count.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Count.h"

using namespace std;

namespace TBTK{
namespace Quantity{

map<Count::Unit, string> Count::unitToString = {
	{Count::Unit::pcs, "pcs"},
	{Count::Unit::mol, "mol"}
};
map<string, Count::Unit> Count::stringToUnit = {
	{"pcs", Count::Unit::pcs},
	{"mol", Count::Unit::mol}
};

string Count::getUnitString(Unit unit){
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

Count::Unit Count::getUnit(const string &str){
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

double Count::getConversionFactor(Unit unit){
	constexpr double pcs_per_mol = 6.02214076e23;
	switch(unit){
		case Unit::pcs:
			return 1.;	//Reference scale
		case Unit::mol:	//1/N_A mol per pcs
			return 1./pcs_per_mol;
		default:
			TBTKExit(
				"Quantity::Count::getConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit)
				<< ".",
				"This should never happen, contact the"
				<< " developer."
			);
	}
}

};	//End of namespace Quantity
};	//End of namespace TBTK
