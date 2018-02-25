/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file TimeEvolver.h
 *  @brief Time evolves a ground state.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TIME_EVOLVER
#define COM_DAFER45_TBTK_TIME_EVOLVER

#include "Diagonalizer.h"
#include "Model.h"
#include "UnitHandler.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Solver{

/** @brief Time evloves a ground state. */
class TimeEvolver : public Solver{
public:
	/** Constructor. */
	TimeEvolver();

	/** Destructor. */
	virtual ~TimeEvolver();

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

	/** Get the Diagonalizer, which contains the eigenvectors,
	 *  energies, etc. */
	Diagonalizer *getDiagonalizer();

	/** Get eigenvalue. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\fn and physical index
	 *  \f$x\f$: \Psi_{n}(x).
	 *  @param state Eigenstate number \f$n\f$.
	 *  @param index Physical index \f$x\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);

	/** Decay modes:
	 *	None - The states continuously connected to the originally
	 *		occupied states at t=0 are occupied.<br/>
	 *	Instantly - The instantaneous ground state is allways occupied.
	 *	Interpolate - Experimental: The occupation of each state below
	 *		the Fermi level is increased by a constant factor until
	 *		it reaches 1, while every state above the Fermi lvel is
	 *		decreased by the same constant factor until it reaches
	 *		0.
	 *	Custom - Experimental: Uses a TimeEvolver::DecayHandler to
	 *		handle the decay. A decay handler has to be specified as
	 *		well using setDecayHandler().
	 */
	enum class DecayMode{None, Instantly, Interpolate, Custom};

	/** Set decay mode. */
	void setDecayMode(DecayMode decayMode);

	class DecayHandler{
	public:
		virtual void decay(
			TimeEvolver *timeEvolver,
			double *occupancy,
			double *eigenValues,
			std::complex<double> **eigenVectorsMap
		) = 0;
	private:
	};

	/** Set DecayHandler. */
	void setDecayHandler(DecayHandler *decayHandler);

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
	/** Diagonalizer which is used to find the ground state, and which also
	 *  acts as a container for the eigenvectors and energies during the
	 *  time evolution. */
	Diagonalizer dSolver;

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

	/** DecayHandler. */
	DecayHandler *decayHandler;

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

	/** List of Diagonalizer. Used by scCallback to redirect the
	 *  self-consistency callback of the dSolver to the correct
	 *  timeEvolver. */
	static std::vector<Diagonalizer*> dSolvers;

	/** Self-consistency callback for the Diagonalizer to call. The
	 *  Diagonalizer solver can not call an ordinary member function
	 *  because it does not know about the TimeEvolver. This static
	 *  callback-function is therefore used in order to redirect the
	 *  call to the correct TimeEvolver*/
	static bool selfConsistencyCallback(Diagonalizer &dSolver);

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

inline void TimeEvolver::setCallback(
	bool (*callback)(TimeEvolver *timeEvolver)
){
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

inline Diagonalizer* TimeEvolver::getDiagonalizer(){
	return &dSolver;
}

inline double TimeEvolver::getEigenValue(int state){
	return eigenValues[state];
}

inline double TimeEvolver::getOccupancy(int state){
	return occupancy[state];
}

inline const std::complex<double> TimeEvolver::getAmplitude(
	int state,
	const Index &index
){
	return eigenVectorsMap[state][getModel().getBasisIndex(index)];
}

inline void TimeEvolver::setDecayMode(DecayMode decayMode){
	this->decayMode = decayMode;
}

inline void TimeEvolver::setDecayHandler(DecayHandler *decayHandler){
	this->decayHandler = decayHandler;
}

inline int TimeEvolver::getCurrentTimeStep(){
	return currentTimeStep;
}

inline void TimeEvolver::setOrthogonalityCheckInterval(
	int orthogonalityCheckInterval
){
	this->orthogonalityCheckInterval = orthogonalityCheckInterval;
}

inline double TimeEvolver::getOrthogonalityError(){
	return orthogonalityError;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
