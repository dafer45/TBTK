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

namespace TBTK{

class TimeEvolver{
public:
	TimeEvolver(TBTK::Model *model);
	~TimeEvolver();

	void run();

	void setSCCallback(bool (*scCallback)(TBTK::DiagonalizationSolver *dSolver));
	void setMaxIterations(int maxIterations);
	void setTimeStepCallback(void (*tsCallback)());

	void setNumTimeSteps(int numTimeSteps);
	void setTimeStep(double dt);
	void setAdiabatic(bool isAdiabatic);
private:
	TBTK::Model *model;
	TBTK::DiagonalizationSolver dSolver;
	bool (*scCallback)(TBTK::DiagonalizationSolver *dSolver);
	void (*tsCallback)();
	int numTimeSteps;
	double dt;
	bool isAdiabatic;
};

inline void TimeEvolver::setSCCallback(bool (*scCallback)(TBTK::DiagonalizationSolver *dSolver)){
	this->scCallback = scCallback;
}

inline void TimeEvolver::setMaxIterations(int maxIterations){
	dSolver.setMaxIterations(maxIterations);
}

inline void TimeEvolver::setTimeStepCallback(void (*tsCallback)()){
	this->tsCallback = tsCallback;
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

}; //End of namespace TBTK

#endif
