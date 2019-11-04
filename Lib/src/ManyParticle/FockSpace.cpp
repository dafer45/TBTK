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

/** @file FockSpace.cpp
 *
 *  @author Kristofer Björnson
 */


#include "TBTK/FockSpace.h"
#include "TBTK/FockStateMap/LookupTableMap.h"

namespace TBTK{

template<>
FockSpace<BitRegister>::FockSpace(
	const HoppingAmplitudeSet *hoppingAmplitudeSet,
	Statistics statistics,
	unsigned int maxParticlesPerState
){
	this->hoppingAmplitudeSet = hoppingAmplitudeSet;
	this->statistics = statistics;

//	unsigned int maxParticlesPerState;
	switch(statistics){
	case Statistics::FermiDirac:
		maxParticlesPerState = 1;
		break;
	case Statistics::BoseEinstein:
//		maxParticlesPerState = maxParticlesPerState;
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	int numBitsPerState = 0;
	for(int n = maxParticlesPerState; n != 0; n /= 2)
		numBitsPerState++;

	exponentialDimension = numBitsPerState*hoppingAmplitudeSet->getBasisSize();

	TBTKAssert(
		exponentialDimension < BitRegister().getNumBits(),
		"FockSpace::FockSpace()",
		"The Hilbert space is too big to be contained in a BitRegister.",
		"Use ExtensiveBitRegister instead."
	);

	vacuumState = new FockState<BitRegister>(BitRegister().getNumBits());

	BitRegister fermionMask;
	fermionMask.clear();
	switch(statistics){
	case Statistics::FermiDirac:
		for(unsigned int n = 0; n < exponentialDimension; n++)
			fermionMask.setBit(n, true);
		break;
	case Statistics::BoseEinstein:
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	operators = new LadderOperator<BitRegister>*[hoppingAmplitudeSet->getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
/*		operators[n] = new LadderOperator<BitRegister>[2]{
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Creation,
				statistics,
				hoppingAmplitudeSet,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			),
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Annihilation,
				statistics,
				hoppingAmplitudeSet,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			)
		};*/
		operators[n] = new LadderOperator<BitRegister>[2];
		operators[n][0] = LadderOperator<BitRegister>(
			LadderOperator<BitRegister>::Type::Creation,
			statistics,
			hoppingAmplitudeSet,
			n,
			numBitsPerState,
			maxParticlesPerState,
			*vacuumState,
			fermionMask
		);
		operators[n][1] = LadderOperator<BitRegister>(
			LadderOperator<BitRegister>::Type::Annihilation,
			statistics,
			hoppingAmplitudeSet,
			n,
			numBitsPerState,
			maxParticlesPerState,
			*vacuumState,
			fermionMask
		);
	}

/*	if(numParticles < 0){
		fockStateMap = new DefaultFockStateMap<BitRegister>(
			exponentialDimension
		);
	}
	else{
		fockStateMap = new LookupTableFockStateMap<BitRegister>(
			exponentialDimension
		);

		FockState<BitRegister> fockState = getVacuumState();
		for(unsigned int n = 0; n < (unsigned int)(1 << exponentialDimension); n++){
			if(fockState.getBitRegister().getNumOneBits() == (unsigned int)numParticles)
				((LookupTableFockStateMap<BitRegister>*)fockStateMap)->addState(fockState);

			fockState.getBitRegister()++;
		}
	}*/
}

template<>
FockSpace<ExtensiveBitRegister>::FockSpace(
	const HoppingAmplitudeSet *hoppingAmplitudeSet,
	Statistics statistics,
	unsigned int maxParticlesPerState
){
	this->hoppingAmplitudeSet = hoppingAmplitudeSet;
	this->statistics = statistics;

//	unsigned int maxParticlesPerState;
	switch(statistics){
	case Statistics::FermiDirac:
		maxParticlesPerState = 1;
		break;
	case Statistics::BoseEinstein:
//		maxParticlesPerState = maxParticlesPerState;
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	int numBitsPerState = 0;
	for(int n = maxParticlesPerState; n != 0; n /= 2)
		numBitsPerState++;

	exponentialDimension = numBitsPerState*hoppingAmplitudeSet->getBasisSize();

	vacuumState = new FockState<ExtensiveBitRegister>(
		exponentialDimension + 1
	);

	ExtensiveBitRegister fermionMask(exponentialDimension+1);
	fermionMask.clear();
	switch(statistics){
	case Statistics::FermiDirac:
		for(unsigned int n = 0; n < exponentialDimension; n++)
			fermionMask.setBit(n, true);
		break;
	case Statistics::BoseEinstein:
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	operators = new LadderOperator<ExtensiveBitRegister>*[hoppingAmplitudeSet->getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
/*		operators[n] = new LadderOperator<ExtensiveBitRegister>[2]{
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Creation,
				statistics,
				hoppingAmplitudeSet,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			),
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
				statistics,
				hoppingAmplitudeSet,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			)
		};*/
		operators[n] = new LadderOperator<ExtensiveBitRegister>[2];
		operators[n][0] = LadderOperator<ExtensiveBitRegister>(
			LadderOperator<ExtensiveBitRegister>::Type::Creation,
			statistics,
			hoppingAmplitudeSet,
			n,
			numBitsPerState,
			maxParticlesPerState,
			*vacuumState,
			fermionMask
		);
		operators[n][1] = LadderOperator<ExtensiveBitRegister>(
			LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
			statistics,
			hoppingAmplitudeSet,
			n,
			numBitsPerState,
			maxParticlesPerState,
			*vacuumState,
			fermionMask
		);
	}

/*	fockStateMap = new DefaultFockStateMap<ExtensiveBitRegister>(
		exponentialDimension
	);*/
}

};	//End of namespace TBTK
