#ifndef COM_DAFER45_TBTK_FOCK_SPACE
#define COM_DAFER45_TBTK_FOCK_SPACE

#include "AmplitudeSet.h"
#include "BitRegister.h"
#include "ExtensiveBitRegister.h"
#include "FockState.h"
#include "LadderOperator.h"
#include "Model.h"
#include "FockStateMap.h"
#include "DefaultFockStateMap.h"

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

	/** Returns the many-body Hilbert space index corresponding to the
	 *  given FockState. */
	unsigned int getBasisIndex(
		const FockState<BIT_REGISTER> &fockState
	) const;

	/** Returns the FockState corresponding to the given many-body Hilbert
	 *  space index. */
	FockState<BIT_REGISTER> getFockState(unsigned int state) const;
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
	FockStateMap<BIT_REGISTER> *fockStateMap;
};

template<>
FockSpace<BitRegister>::FockSpace(
	AmplitudeSet *amplitudeSet,
	Model::Statistics statistics,
	int numParticles
){
//	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;
	this->statistics = statistics;

	unsigned int maxParticlesPerState;
	switch(statistics){
	case Model::Statistics::FermiDirac:
		maxParticlesPerState = 1;
		break;
	case Model::Statistics::BoseEinstein:
		maxParticlesPerState = abs(numParticles);
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

	exponentialDimension = numBitsPerState*amplitudeSet->getBasisSize();

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
	case Model::Statistics::FermiDirac:
		for(int n = 0; n < exponentialDimension; n++)
			fermionMask.setBit(n, true);
		break;
	case Model::Statistics::BoseEinstein:
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	operators = new LadderOperator<BitRegister>*[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		operators[n] = new LadderOperator<BitRegister>[2]{
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Creation,
				statistics,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			),
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Annihilation,
				statistics,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			)
		};
	}

	fockStateMap = new DefaultFockStateMap<BitRegister>(
		exponentialDimension
	);
}

template<>
FockSpace<ExtensiveBitRegister>::FockSpace(
	AmplitudeSet *amplitudeSet,
	Model::Statistics statistics,
	int numParticles
){
//	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;
	this->statistics = statistics;

	unsigned int maxParticlesPerState;
	switch(statistics){
	case Model::Statistics::FermiDirac:
		maxParticlesPerState = 1;
		break;
	case Model::Statistics::BoseEinstein:
		maxParticlesPerState = abs(numParticles);
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

	exponentialDimension = numBitsPerState*amplitudeSet->getBasisSize();

	vacuumState = new FockState<ExtensiveBitRegister>(
		exponentialDimension + 1
	);

	ExtensiveBitRegister fermionMask(exponentialDimension+1);
	fermionMask.clear();
	switch(statistics){
	case Model::Statistics::FermiDirac:
		for(int n = 0; n < exponentialDimension; n++)
			fermionMask.setBit(n, true);
		break;
	case Model::Statistics::BoseEinstein:
		break;
	default:
		TBTKExit(
			"FockSpace::FockSpace()",
			"Unknown statistics.",
			"This should never happen, contact the developer."
		);
	}

	operators = new LadderOperator<ExtensiveBitRegister>*[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		operators[n] = new LadderOperator<ExtensiveBitRegister>[2]{
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Creation,
				statistics,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			),
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
				statistics,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState,
				fermionMask
			)
		};
	}

	fockStateMap = new DefaultFockStateMap<ExtensiveBitRegister>(
		exponentialDimension
	);
}

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
unsigned int FockSpace<BIT_REGISTER>::getBasisIndex(const FockState<BIT_REGISTER> &fockState) const{
	return fockStateMap->getBasisIndex(fockState);
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER> FockSpace<BIT_REGISTER>::getFockState(unsigned int state) const{
	return fockStateMap->getFockState(state);
}

};	//End of namespace TBTK

#endif
