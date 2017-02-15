#ifndef COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER
#define COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER

#include "DiagonalizationSolver.h"
#include "InteractionAmplitudeSet.h"
#include "Model.h"
#include "FockSpaceWrapper.h"
#include "WrapperRule.h"

#include <initializer_list>

namespace TBTK{

class ExactDiagonalizationSolver{
public:
	/** Constructor. */
	ExactDiagonalizationSolver(
		Model *singleParticleModel,
		InteractionAmplitudeSet *interactionAmplitudeSet,
		FockSpaceWrapper fockSpaceWrapper
	);

	/** Destructor. */
	~ExactDiagonalizationSolver();

	/** Add FockStateRule. */
	unsigned int addSubspace(std::initializer_list<const FockStateRule::WrapperRule> rules);

	/** Run calculation. */
	void run(unsigned int subspace);

	/** Get eigen values. */
	const double* getEigenValues(unsigned int subspace);

	/** Get eigen value. */
	const double getEigenValue(unsigned int subspace, int state);

	/** Get amplitude for a given eigenvector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$
	 *  @param subspace Subspace identifier.
	 *  @param state Eigenstate number.
	 *  @param index Physical index \f$x\fx. */
	const std::complex<double> getAmplitude(
		unsigned int subspace,
		int state,
		const Index &index
	);
private:
	/** Model to work on. */
	Model *singleParticleModel;

	/** Interaction amplitude set. */
	InteractionAmplitudeSet *interactionAmplitudeSet;

	/** Fock space wrapper. */
	FockSpaceWrapper fockSpaceWrapper;

	/** Subspace context containing rules, a many-body model, and a
	 *  diagonalization solver for a specific subspace. */
	class SubspaceContext{
	public:
		/** Constructor. */
		SubspaceContext(
			std::initializer_list<const FockStateRule::WrapperRule> rules
		);

		/** Destructor. */
		~SubspaceContext();

		/** Subspace rules. */
		std::vector<FockStateRule::WrapperRule> rules;

		/** Pointer to many-body model. */
		Model *manyBodyModel;

		/** Pointer to diagonalization solver. */
		DiagonalizationSolver *dSolver;
	private:
	};

	/** Subspace contexts. */
	std::vector<SubspaceContext> subspaceContexts;

	/** Setup many-body mapping. */
	void setupManyBodyModel(unsigned int subspace);

	/** Setup many-body model. */
	template<typename BIT_REGISTER>
	void setupManyBodyModel(unsigned int subspace);
};

inline const double* ExactDiagonalizationSolver::getEigenValues(unsigned int subspace){
	return subspaceContexts.at(subspace).dSolver->getEigenValues();
}

inline const double ExactDiagonalizationSolver::getEigenValue(
	unsigned int subspace,
	int state
){
	return subspaceContexts.at(subspace).dSolver->getEigenValue(state);
}

inline const std::complex<double> ExactDiagonalizationSolver::getAmplitude(
	unsigned int subspace,
	int state,
	const Index &index
){
	return subspaceContexts.at(subspace).dSolver->getAmplitude(
		state,
		index
	);
}

};	//End of namespace TBTK

#endif
