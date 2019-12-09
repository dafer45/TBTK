/* Copyright 2018 Kristofer Björnson
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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file Calculator.h
 *  @brief Base class Calculators
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CALCULATOR_CALCULATOR
#define COM_DAFER45_TBTK_CALCULATOR_CALCULATOR

namespace TBTK{
namespace Calculator{

/** The Calculator is a base class for derived Calculator that provides
 *  interfaces for calculate common physical properties such Density, LDOS,
 *  etc. for a single Index. */
template<typename>
class Calculator{
public:
	void calculate();
};

};	//End of namespace Calculator
};	//End of namespace TBTK

#endif
/// @endcond
