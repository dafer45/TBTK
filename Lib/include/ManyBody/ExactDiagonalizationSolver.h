#ifndef COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER
#define COM_DAFER45_TBTK_EXACT_DIAGONALIZATION_SOLVER

#include "InteractionAmplitudeSet.h"
#include "Model.h"

namespace TBTK{

class ExactDiagonalizationSolver{
public:
	/** Constructor. */
	ExactDiagonalizationSolver(Model *model, InteractionAmplitudeSet *interactionAmplitudeSet);

	/** Destructor. */
	~ExactDiagonalizationSolver();

	/** Run calculation. */
	void run();
private:
	/** Model to work on. */
	Model *model;

	/** Interaction amplitude set. */
	InteractionAmplitudeSet *interactionAmplitudeSet;

	/** Many-body model. */
	Model *manyBodyModel;

	/** Setup many-body mapping. */
	void setupManyBodyModel();

	/** Setup many-body model. */
	template<typename BIT_REGISTER>
	void setupManyBodyModel();
};

};	//End of namespace TBTK

#endif
