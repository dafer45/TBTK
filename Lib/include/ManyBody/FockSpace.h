#ifndef COM_DAFER45_TBTK_FOCK_SPACE
#define COM_DAFER45_TBTK_FOCK_SPACE

#include "AmplitudeSet.h"
#include "BitRegister.h"
#include "ExtensiveBitRegister.h"
#include "LadderOperator.h"
#include "Model.h"

namespace TBTK{

template<typename T>
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
	LadderOperator<T>** getOperators() const;

	/** Get the vacuum state. */
	FockState<T> getVacuumState() const;
private:
	/** Particle number. If positive, only the Fock space is restricted to
	 *  the subsapce with numParticle particles. If numParticles is
	 *  negative, the Fock space is restricted to the subspace with up to
	 *  -numParticles particles. */
	unsigned int numParticles;

	/** Maximum number of particles per state. Is 1 for fermions, and
	 *  |numParticles| for bosons. */
	unsigned int maxParticlesPerState;

	/** Number of bits needed to encode all states. */
	unsigned int exponentialDimension;

	/** AmplitudeSet holding the single particle representation. */
	AmplitudeSet *amplitudeSet;

	/** Template state used to create new states. */
	FockState<T> *templateState;

	/** Operators. */
	LadderOperator<T> **operators;
};

template<>
FockSpace<BitRegister>::FockSpace(
	AmplitudeSet *amplitudeSet,
	Model::Statistics statistics,
	int numParticles
){
	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;

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

	templateState = new FockState<BitRegister>(BitRegister().getNumBits());

	operators = new LadderOperator<BitRegister>*[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		operators[n] = new LadderOperator<BitRegister>[2]{
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Creation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*templateState
			),
			LadderOperator<BitRegister>(
				LadderOperator<BitRegister>::Type::Annihilation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*templateState
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
	this->numParticles = numParticles;
	this->amplitudeSet = amplitudeSet;

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

	templateState = new FockState<ExtensiveBitRegister>(
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
				*templateState
			),
			LadderOperator<ExtensiveBitRegister>(
				LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
				n,
				numBitsPerState,
				maxParticlesPerState,
				*templateState
			)
		};
	}
}

template<typename T>
FockSpace<T>::~FockSpace(){
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		delete [] operators[n];
	delete [] operators;
}

template<typename T>
LadderOperator<T>** FockSpace<T>::getOperators() const{
	return operators;
}

template<typename T>
FockState<T> FockSpace<T>::getVacuumState() const{
	return FockState<T>(exponentialDimension + 1);
//	return templateState.cloneStructure().clear();
}

};	//End of namespace TBTK

#endif
