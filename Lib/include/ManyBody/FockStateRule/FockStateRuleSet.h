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
 *  @file FockStateRuleSet.h
 *  @brief FockStateRuleSet.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FOCK_STATE_RULE_SET
#define COM_DAFER45_TBTK_FOCK_STATE_RULE_SET

#include "FockStateRule.h"
#include "WrapperRule.h"

#include <vector>

namespace TBTK{

class FockStateRuleSet{
public:
	/** Constructor */
	FockStateRuleSet();

	/** Destructor. */
	~FockStateRuleSet();

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	bool isSatisfied(
		const FockSpace<BitRegister> &fockSpace,
		const FockState<BitRegister> &fockState
	) const;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	bool isSatisfied(
		const FockSpace<ExtensiveBitRegister> &fockSpace,
		const FockState<ExtensiveBitRegister> &fockState
	) const;

	/** Add FockStateRule. */
	void addFockStateRule(const FockStateRule::WrapperRule &fockStateRule);

	/** Get size. */
	unsigned int getSize() const;

	/** Comparison operator. */
	bool operator==(const FockStateRuleSet &rhs) const;
private:
	/** FockStateRules. */
	std::vector<FockStateRule::WrapperRule> fockStateRules;
};

inline void FockStateRuleSet::addFockStateRule(
	const FockStateRule::WrapperRule &fockStateRule
){
	for(unsigned int n = 0; n < fockStateRules.size(); n++)
		if(fockStateRules.at(n) == fockStateRule)
			return;

	fockStateRules.push_back(fockStateRule);
}

inline unsigned int FockStateRuleSet::getSize() const{
	return fockStateRules.size();
}

};	//End of namespace TBTK

#endif
