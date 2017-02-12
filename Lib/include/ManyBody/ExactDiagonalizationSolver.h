#ifndef COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER
#define COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER

#include "InteractionAmplitudeSet.h"
#include "Model.h"
#include "WrapperRule.h"

#include <initializer_list>

namespace TBTK{

class ExactDiagonalizationSolver{
public:
	/** Constructor. */
	ExactDiagonalizationSolver(Model *singleParticleModel, InteractionAmplitudeSet *interactionAmplitudeSet);

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

	/** Rules for constructing subspaces. */
	std::vector<std::vector<FockStateRule::WrapperRule>> subspaceRules;

	/** Many-body model. */
	std::vector<Model*> manyBodyModels;

	/** Setup many-body mapping. */
	void setupManyBodyModel(unsigned int subspace);

	/** Setup many-body model. */
	template<typename BIT_REGISTER>
	void setupManyBodyModel(unsigned int subspace);
};

};	//End of namespace TBTK

#endif
