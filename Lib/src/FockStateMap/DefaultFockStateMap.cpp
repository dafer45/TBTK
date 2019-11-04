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

/** @file DefaultFockStateMap.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateMap/DefaultMap.h"

namespace TBTK{
namespace FockStateMap{

template<>
unsigned int DefaultMap<BitRegister>::getBasisIndex(
	const FockState<BitRegister> &fockState
) const{
	const BitRegister& bitRegister = fockState.getBitRegister();
	return bitRegister.getValues();
}

template<>
unsigned int DefaultMap<ExtensiveBitRegister>::getBasisIndex(
	const FockState<ExtensiveBitRegister> &fockState
) const{
	TBTKExit(
		"DefaultFockStateMap<ExtensiveBitRegister>::getBasisIndex()",
		"Function not supported for ExtensiveBitRegister.",
		"Use BitRegister instead, or change StateMap."
        );
}

template<>
FockState<BitRegister> DefaultMap<BitRegister>::getFockState(
	unsigned int state
) const{
	FockState<BitRegister> result(getExponentialDimension()+1);
	result.getBitRegister().setValues(state);

	return result;
}

template<>
FockState<ExtensiveBitRegister> DefaultMap<ExtensiveBitRegister>::getFockState(
	unsigned int state
) const{
	TBTKExit(
		"DefaultFockStateMap<ExtensiveBitRegister>::getFockState()",
		"Function not supported for ExtensiveBitRegister.",
		"Use BitRegister instead, or change StateMap."
	);
}

};	//End of namesapce FockStateMap
};	//End of namespace TBTK
