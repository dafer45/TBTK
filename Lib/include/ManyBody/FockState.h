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

namespace TBTK{

template<typename BIT_REGISTER>
class LadderOperator;

template<typename BIT_REGISTER>
class FockState{
public:
	/** Constructor. */
	FockState(unsigned int exponentialDimension);

	/** Destructor. */
	~FockState();

	/** Returns true if the vector is the state is the null vector. */
	bool isNull() const;

	/** Print. */
	void print() const;
private:
	friend class LadderOperator<BIT_REGISTER>;

	BIT_REGISTER bitRegister;
};

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>::FockState(unsigned int exponentialDimension
) :
	bitRegister(exponentialDimension+1)
{
	bitRegister.clear();
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>::~FockState(){
}

template<typename BIT_REGISTER>
bool FockState<BIT_REGISTER>::isNull() const{
	return bitRegister.getMostSignificantBit();
}

template<typename BIT_REGISTER>
void FockState<BIT_REGISTER>::print() const{
	bitRegister.print();
}

};	//End of namespace TBTK

#endif
