#ifndef COM_DAFER45_TBTK_FOCK_SPACE
#define COM_DAFER45_TBTK_FOCK_SPACE

#include "AmplitudeSet.h"
#include "BitRegister.h"
#include "ExtensiveBitRegister.h"
#include "LadderOperator.h"
#include "Model.h"

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
private:
	/** Particle number. If positive, only the Fock space is restricted to
	 *  the subsapce with numParticle particles. If numParticles is
	 *  negative, the Fock space is restricted to the subspace with up to
	 *  -numParticles particles. */
//	unsigned int numParticles;

	/** Maximum number of particles per state. Is 1 for fermions, and
	 *  |numParticles| for bosons. */
//	unsigned int maxParticlesPerState;

	/** Number of bits needed to encode all states. */
	unsigned int exponentialDimension;

	/** AmplitudeSet holding the single particle representation. */
	AmplitudeSet *amplitudeSet;

	/** Vacuum state used as template when creating new states. */
	FockState<BIT_REGISTER> *vacuumState;

	/** Operators. */
	LadderOperator<BIT_REGISTER> **operators;

	/** Fock state hash callback. */
	unsigned int hashCallback(const FockState<BIT_REGISTER> &fockState);
};

template<>
FockSpace<BitRegister>::FockSpace(
	AmplitudeSet *amplitudeSet,
	Model::Statistics statistics,
	int numParticles
){
//	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;

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

	operators = new LadderOperator<BitRegister>*[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		operators[n] = new LadderOperator<BitRegister>[2]{
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Creation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState
			),
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Annihilation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState
			)
		};
	}
}

template<>
FockSpace<ExtensiveBitRegister>::FockSpace(
	AmplitudeSet *amplitudeSet,
	Model::Statistics statistics,
	int numParticles
){
//	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;

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

	operators = new LadderOperator<ExtensiveBitRegister>*[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		operators[n] = new LadderOperator<ExtensiveBitRegister>[2]{
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Creation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState
			),
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*vacuumState
			)
		};
	}
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

};	//End of namespace TBTK

#endif
