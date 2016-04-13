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
#include "UnitHandler.h"
#include <vector>
#include <complex>

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

	/** Set number of particles.
	 *
	 *  @param Number of occupied particles. If set to a negative number,
	 *  the number of particles is determined by the Fermi level. For fixed
	 *  particle number the Fermi level at the first time step sets the
	 *  number of paticles, for non-fixed particle number the Fermi level
	 *  determines the number of particles in each time step. */
	void setNumberOfParticles(int numberOfParticles);

	/** Get number of particles. */
	int getNumberOfParticles();

	/** Fix particle number. If set to true, the number of particles remain
	 *  the same throughout the calculation. */
	void fixParticleNumber(bool particleNumberIsFixed);

	/** Get occupancy of state. */
	double getOccupancy(int state);

	/** Get the DiagonalizationSolver, which contains the eigenvectors,
	 *  energies, etc. */
	DiagonalizationSolver *getDiagonalizationSolver();

	/** Get eigenvalue. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\fn and physical index
	 *  \f$x\f$: \Psi_{n}(x).
	 *  @param state Eigenstate number \f$n\f$.
	 *  @param index Physical index \f$x\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);

	/** Get model. */
	Model* getModel();

	/** Decay modes:
	 *	None - The states continuously connected to the originally
	 *		occupied states at t=0 are occupied.<br/>
	 *	Instantly - The instantaneous ground state is allways occupied.
	 *	Interpolate - Experimental: The occupation of each state below
	 *		the Fermi level is increased by a constant factor until
	 *		it reaches 1, while every state above the Fermi lvel is
	 *		decreased by the same constant factor until it reaches
	 *		0.
	 */
	enum class DecayMode{None, Instantly, Interpolate};

	/** Set decay mode. */
	void setDecayMode(DecayMode decayMode);

	/** Get current time step. Returns -1 while in the self-consistent
	 *  loop. */
	int getCurrentTimeStep();

	/** Set number of time steps between orthogonality checks. Zero
	 *  corresponds to no checks. The orthogonality check is
	 *  computationally heavy O(n^3). */
	void setOrthogonalityCheckInterval(int orthogonalityCheckInterval);

	/** Get orthogonalityError. */
	double getOrthogonalityError();
private:
	/** Model to work on. */
	Model *model;

	/** DiagonalizationSolver which is used to find the ground state, and
	 *  which also acts as a container for the eigenvectors and energies
	 *  during the time evolution. */
	DiagonalizationSolver dSolver;

	/** Pointer to array containing eigenvalues. */
	double *eigenValues;

	/** Pointer to array containing eigenvectors. */
	std::complex<double> *eigenVectors;

	/** Pointer to eigenvector pointers. Array used to to index into
	 *  eigenVectors array, avoids the need to sort eigenVectors according
	 *  to energy. eigenVectorsMap is sorted instead, and should be used to
	 *  access eigenVectors. */
	std::complex<double> **eigenVectorsMap;

	/** Pointer to array storing occupation numbers. */
	double* occupancy;

	/** Number of particles. Negative: Fermi level determines occupancy. */
	int numberOfParticles;

	/** Flag indicating whether or not the number of particles is fixed or
	 *  not. */
	bool particleNumberIsFixed;

	/** Decay mode. */
	DecayMode decayMode;

	/** Callback that is called at each iteration of the self-consistent
	 *  and time iteration loop. */
	bool (*callback)(TimeEvolver *timeEvolver);

	/** Total number of time steps to evolve. */
	int numTimeSteps;

	/** Size of time step. */
	double dt;

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

	/** Member function to call when diagonalization is finished. Called by
	 *  scCallback(). */
	void onDiagonalizationFinished();

	/** Sort eigenvalues, eigenVectorsMap, and occupancy according to
	 *  energy (eigenvalues). */
	void sort();

	/** Update occupancy. */
	void updateOccupancy();

	/** Execute instantaneous decay. */
	void decayInstantly();

	/** Experimental: Execute interpolated decay. */
	void decayInterpolate();

	/** Parameter calculated during time steping to check for failure to
	 *  keep basis orthogonal. */
	double orthogonalityError;

	/** Number of time steps between the orthogonalityErro should be
	 *  updated. Zero corresponds to no check. */
	int orthogonalityCheckInterval;

	/** Calculate orthogonality error. */
	void calculateOrthogonalityError();
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

inline void TimeEvolver::setNumberOfParticles(int numberOfParticles){
	this->numberOfParticles = numberOfParticles;
}

inline int TimeEvolver::getNumberOfParticles(){
	return numberOfParticles;
}

inline void TimeEvolver::fixParticleNumber(bool particleNumberIsFixed){
	this->particleNumberIsFixed = particleNumberIsFixed;
}

inline DiagonalizationSolver* TimeEvolver::getDiagonalizationSolver(){
	return &dSolver;
}

inline double TimeEvolver::getEigenValue(int state){
	return eigenValues[state];
}

inline double TimeEvolver::getOccupancy(int state){
	return occupancy[state];
}

inline const std::complex<double> TimeEvolver::getAmplitude(int state, const Index &index){
	return eigenVectorsMap[state][model->getBasisIndex(index)];
}

inline Model* TimeEvolver::getModel(){
	return model;
}

inline void TimeEvolver::setDecayMode(DecayMode decayMode){
	this->decayMode = decayMode;
}

inline int TimeEvolver::getCurrentTimeStep(){
	return currentTimeStep;
}

inline void TimeEvolver::setOrthogonalityCheckInterval(int orthogonalityCheckInterval){
	this->orthogonalityCheckInterval = orthogonalityCheckInterval;
}

inline double TimeEvolver::getOrthogonalityError(){
	return orthogonalityError;
}

}; //End of namespace TBTK

#endif
