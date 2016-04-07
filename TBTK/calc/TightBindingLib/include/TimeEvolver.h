/** @package TBTKcalc
 *  @file TimeEvolver.h
 *  @brief TimerEvolver for time evolution of ground state in respons to
 *  external field.
 *  
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TIME_EVOLVER
#define COM_DAFER45_TBTK_TIME_EVOLVER

#include "Model.h"
#include "DiagonalizationSolver.h"
#include <vector>

namespace TBTK{

class TimeEvolver{
public:
	/** Constructor. */
	TimeEvolver(TBTK::Model *model);

	/** Destructor. */
	~TimeEvolver();

	/** Run calculation. First self-consistently finds the ground state,
	 *  and then time evolves the state. The callback set by setCallback
	 *  is called at each iteration of both the self-consistent loop and
	 *  time stepping loop. */
	void run();

	/** Set callback used to update the Hamiltonian. */
	void setCallback(bool (*callback)(TimeEvolver *timeEvolver));

	/** Set max iterations for the self-consistent loop. */
	void setMaxSCIterations(int maxIterations);

	/** Set number of time steps for time evolution. */
	void setNumTimeSteps(int numTimeSteps);

	/** Set length of time step used for time evolution. */
	void setTimeStep(double dt);

	/** May change. Set whether the energies should be updated or not. */
	void setAdiabatic(bool isAdiabatic);

	/** Get the DiagonalizationSolver, which contains the eigenvectors,
	 *  energies, etc. */
	DiagonalizationSolver *getDiagonalizationSolver();

	/** Get current time step. Returns -1 while in the self-consistent
	 *  loop. */
	int getCurrentTimeStep();
private:
	/** Model to work on. */
	Model *model;

	/** DiagonalizationSolver which is used to find the ground state, and
	 *  which also acts as a container for the eigenvectors and energies
	 *  during the time evolution. */
	DiagonalizationSolver dSolver;

	/** Callback that is called at each iteration of the self-consistent
	 *  and time iteration loop. */
	bool (*callback)(TimeEvolver *timeEvolver);

	/** Total number of time steps to evolve. */
	int numTimeSteps;

	/** Size of time step. */
	double dt;

	/** May change. Flag indicating whether energies should be updated or
	 *  not. */
	bool isAdiabatic;

	/** Current time step. */
	int currentTimeStep;

	/** List of timeEvolvers. Used by scCallback to redirect the
	 *  self-consistency callback of the dSolver to the correct
	 *  timeEvolver. */
	static std::vector<TimeEvolver*> timeEvolvers;

	/** List of DiagonalizationSolvers. Used by scCallback to redirect the
	 *  self-consistency callback of the dSolver to the correct
	 *  timeEvolver. */
	static std::vector<DiagonalizationSolver*> dSolvers;

	/** Self-consistency callback for the DiagonalizationSolver to call.
	 *  The Diagonalization solver can not call an ordinary member
	 *  function because it does not know about the TimeEvolver. This
	 *  static callback-function is therefore used in order to redirect the
	 *  call to the correct TimeEvolver*/
	static bool scCallback(DiagonalizationSolver *dSolver);
};

inline void TimeEvolver::setCallback(bool (*callback)(TimeEvolver *timeEvolver)){
	this->callback = callback;
}

inline void TimeEvolver::setMaxSCIterations(int maxIterations){
	dSolver.setMaxIterations(maxIterations);
}

inline void TimeEvolver::setNumTimeSteps(int numTimeSteps){
	this->numTimeSteps = numTimeSteps;
}

inline void TimeEvolver::setTimeStep(double dt){
	this->dt = dt;
}

inline void TimeEvolver::setAdiabatic(bool isAdiabatic){
	this->isAdiabatic = isAdiabatic;
}

inline DiagonalizationSolver* TimeEvolver::getDiagonalizationSolver(){
	return &dSolver;
}

inline int TimeEvolver::getCurrentTimeStep(){
	return currentTimeStep;
}

}; //End of namespace TBTK

#endif
