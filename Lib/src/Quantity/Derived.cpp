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

/** @file Derived.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Derived.h"

using namespace std;

namespace TBTK{
namespace Quantity{

constexpr double kg_per_baseMass = 1.602176634e-19;
constexpr double u_per_baseMass = 2.99792458e8*2.99792458e8/9.31494095e8;
template<>
Mass::ConversionTable Mass::conversionTable({
	{Mass::Unit::kg,	{"kg",	kg_per_baseMass}},
	{Mass::Unit::g,		{"g",	kg_per_baseMass*1e3}},
	{Mass::Unit::mg,	{"mg",	kg_per_baseMass*1e6}},
	{Mass::Unit::ug,	{"ug",	kg_per_baseMass*1e9}},
	{Mass::Unit::ng,	{"ng",	kg_per_baseMass*1e12}},
	{Mass::Unit::pg,	{"pg",	kg_per_baseMass*1e15}},
	{Mass::Unit::fg,	{"fg",	kg_per_baseMass*1e18}},
	{Mass::Unit::ag,	{"ag",	kg_per_baseMass*1e21}},
	{Mass::Unit::ag,	{"u",	u_per_baseMass*1e21}}
});

};	//End of namespace Quantity
};	//End of namespace TBTK
