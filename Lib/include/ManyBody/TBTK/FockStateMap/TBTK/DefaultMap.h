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
 *  @file DefaultMap.h
 *  @brief DefaultMap.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DEFAULT_MAP
#define COM_DAFER45_TBTK_DEFAULT_MAP

#include "TBTK/FockStateMap.h"
#include "TBTK/BitRegister.h"
#include "TBTK/ExtensiveBitRegister.h"

namespace TBTK{
namespace FockStateMap{

template<typename BIT_REGISTER>
class DefaultMap : public FockStateMap<BIT_REGISTER>{
public:
	/** Constructor. */
	DefaultMap(unsigned int exponentialDimension);

	/** Destructor. */
	virtual ~DefaultMap();

	/** Get many-body Hilbert space size. */
	virtual unsigned int getBasisSize() const;

	/** Get many-body Hilbert space index for corresponding FockState. */
	virtual unsigned int getBasisIndex(const FockState<BIT_REGISTER> &fockState) const;

	/** Get FockState for corresponding many-body Hilbert space index. */
	virtual FockState<BIT_REGISTER> getFockState(unsigned int index) const;
private:
};

template<typename BIT_REGISTER>
DefaultMap<BIT_REGISTER>::DefaultMap(
	unsigned int exponentialDimension
) :
	FockStateMap<BIT_REGISTER>(exponentialDimension)
{
}

template<typename BIT_REGISTER>
DefaultMap<BIT_REGISTER>::~DefaultMap(){
}

template<typename BIT_REGISTER>
unsigned int DefaultMap<BIT_REGISTER>::getBasisSize() const{
	return (1 << FockStateMap<BIT_REGISTER>::getExponentialDimension());
}

};	//End of namespace FockStateMap
};	//End of namespace TBTK

#endif
