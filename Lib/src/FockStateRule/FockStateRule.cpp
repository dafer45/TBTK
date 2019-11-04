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

/** @file FockStateRule.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateRule/FockStateRule.h"

namespace TBTK{
namespace FockStateRule{

FockStateRule::FockStateRule(FockStateRuleID fockStateRuleID){
	this->fockStateRuleID = fockStateRuleID;
}

FockStateRule::~FockStateRule(){
}

};	//End of namespace FockSpaceRule
};	//End of namespace TBTK
