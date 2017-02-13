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
 *  @file FockSpaceWrapper.h
 *  @brief Fock space wrapper.
 *
 *  @author Kristofer Björnson
 */


#ifndef COM_DAFER45_TBTK_FOCK_SPACE_WRAPPER
#define COM_DAFER45_TBTK_FOCK_SPACE_WRAPPER

#include "BitRegister.h"
#include "ExtensiveBitRegister.h"
#include "FockSpace.h"

namespace TBTK{

class FockSpaceWrapper{
public:
	/** Constructor. */
	FockSpaceWrapper(FockSpace<BitRegister> *fockSpace);

	/** Constructor. */
	FockSpaceWrapper(FockSpace<ExtensiveBitRegister> *fockSpace);

	/** Destructor. */
	~FockSpaceWrapper();
private:
	/** Pointer to FockSpace using BitRegsiter. */
	FockSpace<BitRegister> *brFockSpace;

	/** Pointer to FockSpace using ExtensiveBitRegister. */
	FockSpace<ExtensiveBitRegister> *ebrFockSpace;
};

};	//End of namespace TBTK

#endif
