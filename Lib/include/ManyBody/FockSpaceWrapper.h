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

#include <memory>

namespace TBTK{

class FockSpaceWrapper{
public:
	/** Constructor. The FockSpaceWrapper assumes ownership of the
	 *  FockSpace and will destroy it at destrucion. */
	FockSpaceWrapper(FockSpace<BitRegister> *fockSpace);

	/** Constructor. The FockSpaceWrapper assumes ownership of the
	 *  FockSpace and will destroy it at destrucion. */
	FockSpaceWrapper(FockSpace<ExtensiveBitRegister> *fockSpace);

	/** Destructor. */
	~FockSpaceWrapper();

	/** Returns true if the wrapped FockState is of type BitRegister. */
	bool wrapsBitRegister();

	/** Returns true if the wrapped FockState is of type
	 *  ExtensiveBitRegister. */
	bool wrapsExtensiveBitRegister();

	/** Returns a pointer to the FockState<BitRegister> contained by the
	 *  wrapper. */
	FockSpace<BitRegister>* getFockSpaceBitRegister();

	/** Returns a pointer to the FockState<ExtensiveBitRegister> contained
	 *  by the wrapper. */
	FockSpace<ExtensiveBitRegister>* getFockSpaceExtensiveBitRegister();
private:
	/** Pointer to FockSpace using BitRegsiter. */
	std::shared_ptr<FockSpace<BitRegister>> brFockSpace;

	/** Pointer to FockSpace using ExtensiveBitRegister. */
	std::shared_ptr<FockSpace<ExtensiveBitRegister>> ebrFockSpace;
};

inline bool FockSpaceWrapper::wrapsBitRegister(){
	if(brFockSpace.get() != NULL)
		return true;
	else
		return false;
}

inline bool FockSpaceWrapper::wrapsExtensiveBitRegister(){
	if(ebrFockSpace.get() != NULL)
		return true;
	else
		return false;
}

inline FockSpace<BitRegister>* FockSpaceWrapper::getFockSpaceBitRegister(){
	return brFockSpace.get();
}

inline FockSpace<ExtensiveBitRegister>* FockSpaceWrapper::getFockSpaceExtensiveBitRegister(){
	return ebrFockSpace.get();
}

};	//End of namespace TBTK

#endif
