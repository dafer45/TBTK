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

/** @file Time.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Time.h"

using namespace std;

namespace TBTK{
namespace Quantity{

map<Time::Unit, string> Time::unitToString = {
	{Time::Unit::s, "s"},
	{Time::Unit::ms, "ms"},
	{Time::Unit::us, "us"},
	{Time::Unit::ns, "ns"},
	{Time::Unit::ps, "ps"},
	{Time::Unit::fs, "fs"},
	{Time::Unit::as, "as"}
};
map<string, Time::Unit> Time::stringToUnit = {
	{"", Time::Unit::s},
	{"", Time::Unit::ms},
	{"", Time::Unit::us},
	{"", Time::Unit::ns},
	{"", Time::Unit::ps},
	{"", Time::Unit::fs},
	{"", Time::Unit::as}
};

std::string Time::getUnitString(Unit unit){
	try{
		return unitToString.at(unit);
	}
	catch(std::out_of_range e){
		TBTKExit(
			"Quantity::Time::getUnitString()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace Quantity
};	//End of namespace TBTK
