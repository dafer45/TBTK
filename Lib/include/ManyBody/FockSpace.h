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
 *  @file FockSpcae.h
 *  @brief Fock space.
 *
 *  @author Kristofer Björnson
 */


#ifndef COM_DAFER45_TBTK_FOCK_SPACE
#define COM_DAFER45_TBTK_FOCK_SPACE

#include "AmplitudeSet.h"
#include "BitRegister.h"
#include "ExtensiveBitRegister.h"
#include "FockState.h"
#include "LadderOperator.h"
#include "Model.h"
#include "FockStateMap.h"
#include "DefaultMap.h"
#include "LookupTableMap.h"

namespace TBTK{

template<typename BIT_REGISTER>
class FockSpace{
public:
	/** Constructor. */
	FockSpace(
		AmplitudeSet *amplitudeSet,
		Model::Statistics statistics,
		int numParticles
	);

	/** Destructor. */
	~FockSpace();

	/** Get operators. */
	LadderOperator<BIT_REGISTER>** getOperators() const;

	/** Get the vacuum state. */
	FockState<BIT_REGISTER> getVacuumState() const;

	/** Returns the number of fermions in the state. */
	unsigned int getNumFermions(
		const FockState<BIT_REGISTER> &fockState
	) const;

	/** Get number of particles in the single particle state with given
	 *  index for the given FockState. */
	unsigned int getNumParticles(
		const FockState<BIT_REGISTER> &fockState,
		const Index &index
	) const;

	/** Get number of particles in the single particle states that
	 *  satisfies the given index pattern for the given FockState. */
	unsigned int getSumParticles(
		const FockState<BIT_REGISTER> &fockState,
		const Index &pattern
	) const;

	/** Create FockSpaceMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockSpaceMap(
		int numParticles
	) const;

	/** Returns the many-body Hilbert space index corresponding to the
	 *  given FockState. */
/*	unsigned int getBasisIndex(
		const FockState<BIT_REGISTER> &fockState
	) const;*/

	/** Returns the FockState corresponding to the given many-body Hilbert
	 *  space index. */
//	FockState<BIT_REGISTER> getFockState(unsigned int state) const;

	/** Get the many-body Hilbert space size. */
//	unsigned int getBasisSize() const;
private:
	/** Particle number. If positive, only the Fock space is restricted to
	 *  the subsapce with numParticle particles. If numParticles is
	 *  negative, the Fock space is restricted to the subspace with up to
	 *  -numParticles particles. */
//	unsigned int numParticles;

	/** Maximum number of particles per state. Is 1 for fermions, and
	 *  |numParticles| for bosons. */
//	unsigned int maxParticlesPerState;

	Model::Statistics statistics;

	/** Number of bits needed to encode all states. */
	unsigned int exponentialDimension;

	/** AmplitudeSet holding the single particle representation. */
	AmplitudeSet *amplitudeSet;

	/** Vacuum state used as template when creating new states. */
	FockState<BIT_REGISTER> *vacuumState;

	/** Operators. */
	LadderOperator<BIT_REGISTER> **operators;

	/** Converts a FockState to a many-body Hilbert space index. */
	unsigned int (*stateMapCallback)(
		const FockState<BIT_REGISTER> &fockState
	);

	/** Fock state map for mapping FockStates to many-body Hilbert space
	 *  indices, and vice versa. */
	FockStateMap::FockStateMap<BIT_REGISTER> *fockStateMap;
};

template<typename BIT_REGISTER>
FockSpace<BIT_REGISTER>::~FockSpace(){
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		delete [] operators[n];
	delete [] operators;
}

template<typename BIT_REGISTER>
LadderOperator<BIT_REGISTER>** FockSpace<BIT_REGISTER>::getOperators() const{
	return operators;
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER> FockSpace<BIT_REGISTER>::getVacuumState() const{
	return *vacuumState;
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getNumFermions(const FockState<BIT_REGISTER> &fockState) const{
	switch(statistics){
	case Model::Statistics::FermiDirac:
		return fockState.bitRegister.getNumOneBits();
	case Model::Statistics::BoseEinstein:
		return 0;
	default:
		TBTKExit(
			"FockSpace<BIT_REGISTER>::getNumFermions()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getNumParticles(
	const FockState<BIT_REGISTER> &fockState,
	const Index &index
) const{
	return operators[amplitudeSet->getBasisIndex(index)][0].getNumParticles(fockState);
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getSumParticles(
	const FockState<BIT_REGISTER> &fockState,
	const Index &pattern
) const{
	if(pattern.isPatternIndex()){
		std::vector<Index> indexList = amplitudeSet->getIndexList(pattern);

		unsigned int numParticles = 0;
		for(unsigned int n = 0; n < indexList.size(); n++){
			numParticles += getNumParticles(
				fockState,
				indexList.at(n)
			);
		}

		return numParticles;
	}
	else{
		return getNumParticles(fockState, pattern);
	}
}

template<typename BIT_REGISTER>
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockSpaceMap(int numParticles) const{
	if(numParticles < 0){
		FockStateMap::DefaultMap<BIT_REGISTER> *fockStateMap = new FockStateMap::DefaultMap<BIT_REGISTER>(
			exponentialDimension
		);

		return fockStateMap;
	}
	else{
		FockStateMap::LookupTableMap<BIT_REGISTER> *fockStateMap = new FockStateMap::LookupTableMap<BIT_REGISTER>(
			exponentialDimension
		);

		FockState<BIT_REGISTER> fockState = getVacuumState();
		for(unsigned int n = 0; n < (unsigned int)(1 << exponentialDimension); n++){
			if(fockState.getBitRegister().getNumOneBits() == (unsigned int)numParticles)
				fockStateMap->addState(fockState);

			fockState.getBitRegister()++;
		}

		return fockStateMap;
	}
}

/*template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getBasisIndex(const FockState<BIT_REGISTER> &fockState) const{
	return fockStateMap->getBasisIndex(fockState);
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER> FockSpace<BIT_REGISTER>::getFockState(unsigned int state) const{
	return fockStateMap->getFockState(state);
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getBasisSize() const{
	return fockStateMap->getBasisSize();
}*/

};	//End of namespace TBTK

#endif
