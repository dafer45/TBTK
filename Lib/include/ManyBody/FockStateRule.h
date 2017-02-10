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

#include "FockSpace.h"
#include "FockState.h"

namespace TBTK{

class FockStateRule{
public:
	/** Constructor */
	FockStateRule();

	/** Destructor. */
	~FockStateRult();

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	template<typename BIT_REGISTER>
	virtual bool isFullfilled(
		const FockSpace<BIT_REGISTER> &fockSpace,
		const FockState<BIT_REGISTER> &fockState
	) = 0;
private:
};

};	//End of namespace TBTK

#endif
