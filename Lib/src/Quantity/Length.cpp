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

map<Length::Unit, string> Length::unitToString = {
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

std::string Length::getUnitString(Unit unit){
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

};	//End of namespace Quantity
};	//End of namespace TBTK
