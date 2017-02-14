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

	/** Rules for constructing subspaces. */
//	std::vector<std::vector<FockStateRule::WrapperRule>> subspaceRules;

	/** Many-body model. */
//	std::vector<Model*> manyBodyModels;

	/** Many-body model. */
//	std::vector<DiagonalizationSolver*> solvers;

	/** Setup many-body mapping. */
	void setupManyBodyModel(unsigned int subspace);

	/** Setup many-body model. */
	template<typename BIT_REGISTER>
	void setupManyBodyModel(unsigned int subspace);
};

};	//End of namespace TBTK

#endif
