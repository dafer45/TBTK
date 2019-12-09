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
 *  @file KineticOperator.h
 *  @brief Kinetic energy operator.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_KINETIC_OPERATOR
#define COM_DAFER45_TBTK_KINETIC_OPERATOR

#include "TBTK/AbstractOperator.h"

namespace TBTK{

class KineticOperator : public AbstractOperator{
public:
	/** Constructor. */
	KineticOperator(double mass);

	/** Get the mass.
	 *
	 *  @return The mass that should enter into the prefactor of the
	 *  operator. */
	double getMass() const;
private:
	/** The mass. */
	double mass;
};

inline KineticOperator::KineticOperator(
	double mass
) :
	AbstractOperator(OperatorID::Kinetic)
{
	this->mass = mass;
}

inline double KineticOperator::getMass() const{
	return mass;
}

};	//End of namespace TBTK

#endif
/// @endcond
