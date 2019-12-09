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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file FockSpcae.h
 *  @brief Fock space.
 *
 *  @author Kristofer Björnson
 */


#ifndef COM_DAFER45_TBTK_FOCK_SPACE
#define COM_DAFER45_TBTK_FOCK_SPACE

#include "TBTK/BitRegister.h"
#include "TBTK/ExtensiveBitRegister.h"
#include "TBTK/FockState.h"
#include "TBTK/FockStateMap/DefaultMap.h"
#include "TBTK/FockStateMap/FockStateMap.h"
#include "TBTK/FockStateMap/LookupTableMap.h"
#include "TBTK/FockStateRuleSet.h"
#include "TBTK/FockStateRule/FockStateRule.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/LadderOperator.h"
//#include "Model.h"
#include "TBTK/Statistics.h"

namespace TBTK{

template<typename BIT_REGISTER>
class FockSpace{
public:
	/** Constructor. */
	FockSpace();

	/** Constructor. */
	FockSpace(
		const HoppingAmplitudeSet *hoppingAmplitudeSet,
		Statistics statistics,
		unsigned int maxParticlesPerState
	);

	/** Copy constructor. */
	FockSpace(const FockSpace &fockSpace);

	/** Destructor. */
	~FockSpace();

	/** Assignment operator. */
	FockSpace& operator=(const FockSpace &rhs);

	/** Get operators. */
	LadderOperator<BIT_REGISTER> const* const* getOperators() const;

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

	/** Create FockStateMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockStateMap(
		int numParticles
	) const;

	/** Create FockStateMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockStateMap(
		const FockStateRule::FockStateRule &rule
	) const;

	/** Create FockStateMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockStateMap(
		std::initializer_list<const FockStateRule::WrapperRule> rules
	) const;

	/** Create FockStateMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockStateMap(
		std::vector<FockStateRule::WrapperRule> rules
	) const;

	/** Create FockStateMap. */
	FockStateMap::FockStateMap<BIT_REGISTER>* createFockStateMap(
		const FockStateRuleSet &rules
	) const;

	/** Get amplitude set. */
	const HoppingAmplitudeSet* getHoppingAmplitudeSet() const;
private:
	/** Statistics. */
	Statistics statistics;

	/** Number of bits needed to encode all states. */
	unsigned int exponentialDimension;

	/** HoppingAmplitudeSet holding the single particle representation. */
	const HoppingAmplitudeSet *hoppingAmplitudeSet;

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
//	FockStateMap::FockStateMap<BIT_REGISTER> *fockStateMap;
};

template<typename BIT_REGISTER>
FockSpace<BIT_REGISTER>::FockSpace(){
	hoppingAmplitudeSet = nullptr;
	vacuumState = nullptr;
	operators = nullptr;
//	fockStateMap = nullptr;
}

template<typename BIT_REGISTER>
FockSpace<BIT_REGISTER>::FockSpace(const FockSpace &fockSpace){
	statistics = fockSpace.statistics;
	exponentialDimension = fockSpace.exponentialDimension;
	hoppingAmplitudeSet = fockSpace.hoppingAmplitudeSet;
	if(fockSpace.vacuumState == nullptr)
		vacuumState = nullptr;
	else
		vacuumState = new FockState<BIT_REGISTER>(*fockSpace.vacuumState);
	if(fockSpace.operators == nullptr){
		operators = nullptr;
	}
	else{
		operators = new LadderOperator<BIT_REGISTER>*[
			hoppingAmplitudeSet->getBasisSize()
		];
		for(
			int n = 0;
			n < hoppingAmplitudeSet->getBasisSize();
			n++
		){
			operators[n] = new LadderOperator<BIT_REGISTER>[2];
			for(unsigned int c = 0; c < 2; c++)
				operators[n][c] = fockSpace.operators[n][c];
		}
	}
}

template<typename BIT_REGISTER>
FockSpace<BIT_REGISTER>::~FockSpace(){
	if(operators != nullptr){
		for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
			delete [] operators[n];
		delete [] operators;
	}
}

template<typename BIT_REGISTER>
FockSpace<BIT_REGISTER>& FockSpace<BIT_REGISTER>::operator=(
	const FockSpace &rhs
){
	if(this != &rhs){
		statistics = rhs.statistics;
		exponentialDimension = rhs.exponentialDimension;
		if(vacuumState != nullptr)
			delete vacuumState;
		if(rhs.vacuumState == nullptr){
			vacuumState = nullptr;
		}
		else{
			vacuumState
				= new FockState<BIT_REGISTER>(*rhs.vacuumState);
		}

		if(operators != nullptr){
			for(
				int n = 0;
				n < hoppingAmplitudeSet->getBasisSize();
				n++
			){
				delete operators[n];
			}
			delete operators;
		}
		hoppingAmplitudeSet = rhs.hoppingAmplitudeSet;
		if(rhs.operators == nullptr){
			operators = nullptr;
		}
		else{
			operators = new LadderOperator<BIT_REGISTER>*[
				hoppingAmplitudeSet->getBasisSize()
			];
			for(
				int n = 0;
				n < hoppingAmplitudeSet->getBasisSize();
				n++
			){
				operators[n]
					= new LadderOperator<BIT_REGISTER>[2];
				for(unsigned int c = 0; c < 2; c++)
					operators[n][c] = rhs.operators[n][c];
			}
		}
	}

	return *this;
}

template<typename BIT_REGISTER>
LadderOperator<BIT_REGISTER> const* const* FockSpace<BIT_REGISTER>::getOperators(
) const{
	return operators;
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER> FockSpace<BIT_REGISTER>::getVacuumState() const{
	return *vacuumState;
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getNumFermions(const FockState<BIT_REGISTER> &fockState) const{
	switch(statistics){
	case Statistics::FermiDirac:
		return fockState.bitRegister.getNumOneBits();
	case Statistics::BoseEinstein:
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
	return operators[hoppingAmplitudeSet->getBasisIndex(index)][0].getNumParticles(fockState);
}

template<typename BIT_REGISTER>
unsigned int FockSpace<BIT_REGISTER>::getSumParticles(
	const FockState<BIT_REGISTER> &fockState,
	const Index &pattern
) const{
	if(pattern.isPatternIndex()){
		std::vector<Index> indexList = hoppingAmplitudeSet->getIndexList(pattern);

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
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockStateMap(int numParticles) const{
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

template<typename BIT_REGISTER>
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockStateMap(const FockStateRule::FockStateRule &rule) const{
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(rule);
	return createFockStateMap(fockStateRuleSet);
}

template<typename BIT_REGISTER>
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockStateMap(
	std::initializer_list<const FockStateRule::WrapperRule> rules
) const{
	FockStateRuleSet fockStateRuleSet;
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(*(rules.begin()+n));
	return createFockStateMap(fockStateRuleSet);
}

template<typename BIT_REGISTER>
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockStateMap(
	std::vector<FockStateRule::WrapperRule> rules
) const{
	FockStateRuleSet fockStateRuleSet;
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(rules.at(n));
	return createFockStateMap(fockStateRuleSet);
}

template<typename BIT_REGISTER>
FockStateMap::FockStateMap<BIT_REGISTER>* FockSpace<BIT_REGISTER>::createFockStateMap(
	const FockStateRuleSet &rules
) const{
	FockStateMap::LookupTableMap<BIT_REGISTER> *fockStateMap = new FockStateMap::LookupTableMap<BIT_REGISTER>(
		exponentialDimension
	);

	if(rules.getSize() == 0){
		FockStateMap::DefaultMap<BIT_REGISTER> *fockStateMap = new FockStateMap::DefaultMap<BIT_REGISTER>(
			exponentialDimension
		);

		return fockStateMap;
	}
	else{
		if(exponentialDimension > 31){
			//See comment bellow
			TBTKExit(
				"FockSpace::createFockStateMap()",
				"FockSpaces with more than 31 states not yet supported using lookup table.",
				""
			);
		}

		//This loop is very slow for large exponential dimension and a
		//better method should be implemented that can take advantage
		//of the FockStateRules more directly.
		FockState<BIT_REGISTER> fockState = getVacuumState();
		for(unsigned int n = 0; n < (unsigned int)(1 << exponentialDimension); n++){
			if(rules.isSatisfied(*this, fockState))
				fockStateMap->addState(fockState);

			fockState.getBitRegister()++;
		}
	}

	return fockStateMap;
}

template<typename BIT_REGISTER>
const HoppingAmplitudeSet* FockSpace<BIT_REGISTER>::getHoppingAmplitudeSet() const{
	return hoppingAmplitudeSet;
}

};	//End of namespace TBTK

#endif
/// @endcond
