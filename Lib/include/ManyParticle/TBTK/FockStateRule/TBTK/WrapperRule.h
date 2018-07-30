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
 *  @file WrapperRule.h
 *  @brief WrapperRule.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WRAPPER_RULE
#define COM_DAFER45_TBTK_WRAPPER_RULE

#include "TBTK/FockStateRule.h"

namespace TBTK{
namespace FockStateRule{

class WrapperRule : public FockStateRule{
public:
	/** Constructor */
	WrapperRule(const FockStateRule &fockStateRule);

	/** Copy constructor */
	WrapperRule(const WrapperRule &wrapperRule);

	/** Destructor. */
	virtual ~WrapperRule();

	/** Clone WrapperRule. */
	virtual WrapperRule* clone() const;

	/** Implements FockStateRule::createNewRule(). */
	virtual WrapperRule createNewRule(
		const LadderOperator<BitRegister> &ladderOperator
	) const;

	/** Implements FockStateRule::createNewRule(). */
	virtual WrapperRule createNewRule(
		const LadderOperator<ExtensiveBitRegister> &ladderOperator
	) const;

	/** Asignment operator. */
	WrapperRule& operator=(const WrapperRule &wrapperRule);

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<BitRegister> &fockSpace,
		const FockState<BitRegister> &fockState
	) const;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<ExtensiveBitRegister> &fockSpace,
		const FockState<ExtensiveBitRegister> &fockState
	) const;

	/** Comparison operator. */
	virtual bool operator==(const FockStateRule &rhs) const;

	/** Implements FockStateRile::print(). */
	virtual void print() const;
private:
	FockStateRule *fockStateRule;
};

inline WrapperRule WrapperRule::createNewRule(
	const LadderOperator<BitRegister> &ladderOperator
) const{
	return WrapperRule(fockStateRule->createNewRule(ladderOperator));
}

inline WrapperRule WrapperRule::createNewRule(
	const LadderOperator<ExtensiveBitRegister> &ladderOperator
) const{
	return WrapperRule(fockStateRule->createNewRule(ladderOperator));
}

inline void WrapperRule::print() const{
	fockStateRule->print();
}

//Note: Declared in FockStateRule.h
inline WrapperRule operator*(
	const LadderOperator<BitRegister> &ladderOperator,
	const FockStateRule &fockStateRule
){
	return fockStateRule.createNewRule(ladderOperator);
}

//Note: Declared in FockStateRule.h
inline WrapperRule operator*(
	const LadderOperator<ExtensiveBitRegister> &ladderOperator,
	const FockStateRule &fockStateRule
){
	return fockStateRule.createNewRule(ladderOperator);
}

};	//End of namespace FockSpaceRule
};	//End of namespace TBTK

#endif
