/* Copyright 2017 Kristofer Björnson
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

/** @file WrapperRule.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{
namespace FockStateRule{

WrapperRule::WrapperRule(
	const FockStateRule &fockStateRule
) :
	FockStateRule(FockStateRuleID::WrapperRule)
{
	this->fockStateRule = fockStateRule.clone();
}

WrapperRule::WrapperRule(
	const WrapperRule &wrapperRule
) :
	FockStateRule(FockStateRuleID::WrapperRule)
{
	this->fockStateRule = (FockStateRule*)wrapperRule.clone();
}

WrapperRule::~WrapperRule(){
	delete fockStateRule;
}

WrapperRule* WrapperRule::clone() const{
	return new WrapperRule(*fockStateRule);
}

WrapperRule& WrapperRule::operator=(const WrapperRule &rhs){
	if(this != &rhs)
		this->fockStateRule = (FockStateRule*)rhs.clone();

	return *this;
}

bool WrapperRule::isSatisfied(
	const FockSpace<BitRegister> &fockSpace,
	const FockState<BitRegister> &fockState
) const{
	return fockStateRule->isSatisfied(fockSpace, fockState);
}

bool WrapperRule::isSatisfied(
	const FockSpace<ExtensiveBitRegister> &fockSpace,
	const FockState<ExtensiveBitRegister> &fockState
) const{
	return fockStateRule->isSatisfied(fockSpace, fockState);
}

bool WrapperRule::operator==(const FockStateRule& rhs) const{
	//Note the order is important here. If rhs is moved to the left, an
	//infinite recursion will occur if the rhs is a WrapperRule.
	return (rhs == *fockStateRule);
}

};	//End of namespace FockSpaceRule
};	//End of namespace TBTK
