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

/** @package TBTKcalc
 *  @file DefaultFockStateMap.h
 *  @brief DefaultFockStateMap.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DEFAULT_FOCK_STATE_MAP
#define COM_DAFER45_TBTK_DEFAULT_FOCK_STATE_MAP

#include "FockStateMap.h"

namespace TBTK{

template<typename BIT_REGISTER>
class DefaultFockStateMap : public FockStateMap<BIT_REGISTER>{
public:
	/** Constructor. */
	DefaultFockStateMap(unsigned int exponentialDimension);

	/** Destructor. */
	~DefaultFockStateMap();

	/** Get many-body Hilbert space index for corresponding FockState. */
	virtual unsigned int getBasisIndex(const FockState<BIT_REGISTER> &fockState) const;

	/** Get FockState for corresponding many-body Hilbert space index. */
	virtual FockState<BIT_REGISTER> getFockState(unsigned int index) const;
private:
};

template<typename BIT_REGISTER>
DefaultFockStateMap<BIT_REGISTER>::DefaultFockStateMap(
	unsigned int exponentialDimension
) :
	FockStateMap<BIT_REGISTER>(exponentialDimension)
{
}

template<typename BIT_REGISTER>
DefaultFockStateMap<BIT_REGISTER>::~DefaultFockStateMap(){
}

template<>
unsigned int DefaultFockStateMap<BitRegister>::getBasisIndex(
	const FockState<BitRegister> &fockState
) const{
	const BitRegister& bitRegister = fockState.getBitRegister();
	return bitRegister.getValues();
}

template<>
unsigned int DefaultFockStateMap<ExtensiveBitRegister>::getBasisIndex(
	const FockState<ExtensiveBitRegister> &fockState
) const{
	TBTKExit(
		"DefaultFockStateMap<ExtensiveBitRegister>::getBasisIndex()",
		"Function not supported for ExtensiveBitRegister.",
		"Use BitRegister instead, or change StateMap."
	);
}

template<>
FockState<BitRegister> DefaultFockStateMap<BitRegister>::getFockState(
	unsigned int state
) const{
	FockState<BitRegister> result(getExponentialDimension()+1);
	result.getBitRegister().setValues(state);

	return result;
}

template<>
FockState<ExtensiveBitRegister> DefaultFockStateMap<ExtensiveBitRegister>::getFockState(
	unsigned int state
) const{
	TBTKExit(
		"DefaultFockStateMap<ExtensiveBitRegister>::getFockState()",
		"Function not supported for ExtensiveBitRegister.",
		"Use BitRegister instead, or change StateMap."
	);
}

};	//End of namespace TBTK

#endif
