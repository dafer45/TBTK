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
 *  @file FockState.h
 *  @brief FockState.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FOCK_STATE
#define COM_DAFER45_TBTK_FOCK_STATE

#include "Streams.h"

namespace TBTK{

template<typename BIT_REGISTER>
class FockSpace;

template<typename BIT_REGISTER>
class LadderOperator;

template<typename BIT_REGISTER>
class FockState{
public:
	/** Constructor. */
	FockState(unsigned int exponentialDimension);

	/** Copy constructor. */
	FockState(const FockState &fockState);

	/** Destructor. */
	~FockState();

	/** Returns true if the vector is the state is the null vector. */
	bool isNull() const;

	/** Returns the BIT_REGISTER. */
	const BIT_REGISTER& getBitRegister() const;

	/** Returns the BIT_REGISTER. */
	BIT_REGISTER& getBitRegister();

	/** Get prefactor. */
	int getPrefactor() const;

	/** Get number of particles. */
//	unsigned int getNumFermions() const;

	/** Print. */
	void print() const;
private:
	/** Allow the FockSpace to immediatly access the internal storage. */
	friend class FockSpace<BIT_REGISTER>;

	/** Allow operators to operate immediately on the internal storage. */
	friend class LadderOperator<BIT_REGISTER>;

	/** Bit register used to store occupation numbers. */
	BIT_REGISTER bitRegister;

	/** Prefactor containing the sign and amplitude of the state a|psi>.
	 *  For efficiency sign(a)a^2 is stored rather than a. */
	int prefactor;
};

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>::FockState(unsigned int exponentialDimension
) :
	bitRegister(exponentialDimension+1)
{
	bitRegister.clear();
	prefactor = 1;
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>::FockState(const FockState &fockState
) :
	bitRegister(fockState.bitRegister)
{
	prefactor = fockState.prefactor;
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>::~FockState(){
}

template<typename BIT_REGISTER>
bool FockState<BIT_REGISTER>::isNull() const{
	return bitRegister.getMostSignificantBit();
}

template<typename BIT_REGISTER>
const BIT_REGISTER& FockState<BIT_REGISTER>::getBitRegister() const{
	return bitRegister;
}

template<typename BIT_REGISTER>
BIT_REGISTER& FockState<BIT_REGISTER>::getBitRegister(){
	return bitRegister;
}

template<typename BIT_REGISTER>
int FockState<BIT_REGISTER>::getPrefactor() const{
	return prefactor;
}

/*template<typename BIT_REGISTER>
unsigned int FockState<BIT_REGISTER>::getNumFermions() const{
	return bitRegister.getNumOneBits();
}*/

template<typename BIT_REGISTER>
void FockState<BIT_REGISTER>::print() const{
	Streams::out << prefactor << "|";
	for(int n = bitRegister.getNumBits()-1; n >= 0; n--){
		Streams::out << bitRegister.getBit(n);
		if(n%8 == 0 && n != 0)
			Streams::out << " ";
	}
	Streams::out << ">\n";
}

};	//End of namespace TBTK

#endif
