/* Copyright 2016 Kristofer Björnson
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

/** @package TBTKcalc
 *  @file FockStateRule.h
 *  @brief FockStateRule.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FOCK_STATE_RULE
#define COM_DAFER45_TBTK_FOCK_STATE_RULE

#include "FockState.h"
#include "BitRegister.h"
#include "ExtensiveBitRegister.h"

namespace TBTK{
	template<typename BIT_REGISTER>
	class FockSpace;
namespace FockStateRule{

class FockStateRule{
public:
	/** Constructor */
	FockStateRule();

	/** Destructor. */
	virtual ~FockStateRule();

	/** Clone FockStateRule. */
	virtual FockStateRule* clone() const = 0;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<BitRegister> &fockSpace,
		const FockState<BitRegister> &fockState
	) const = 0;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<ExtensiveBitRegister> &fockSpace,
		const FockState<ExtensiveBitRegister> &fockState
	) const = 0;
private:
};

};	//End of namespace FockSpaceRule
};	//End of namespace TBTK

#endif