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
 *  @file Diagonalizer.h
 *  @brief Solves a Model using diagonalization.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_DIAGONALIZATION
#define COM_DAFER45_TBTK_SOLVER_DIAGONALIZATION

#include "TBTK/CArray.h"
#include "TBTK/Communicator.h"
#include "TBTK/Model.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Solves a Model using diagonalization.
 *
 *  Solves a Model by Diagonalizing the Hamiltonian. Use
 *  PropertyExtractor::Diagonalizer to extract @link Property::AbstractProperty
 *  Properties@endlink.
 *
 *  <b>Scaling behavior:</b><br />
 *  Time: \f$O(h^3)\f$<br />
 *  Space: \f$O(h^2)\f$
 *
 *  Here \f$h\f$ is the size of the Hilbert space basis.
 *
 *  # Example
 *  \snippet Solver/Diagonalizer.cpp Diagonalizer
 *  ## Output
 *  \snippet output/Solver/Diagonalizer.txt Diagonalizer */
class Diagonalizer : public Solver, public Communicator{
public:
	/** Abstract base class for self-consistency callbacks. */
	class SelfConsistencyCallback{
	public:
		/** Function that is called after the Hamiltonian have been
		 *  diagonalized and is responsible for carrying out the
		 *  self-consistency step. The function should return true if
		 *  the result has converged and false otherwise.
		 *
		 *  @param diagonalizer The solver that calls the
		 *  self-consistency callback-
		 *
		 *  @return True if the solution has converged, otherwise false. */
		virtual bool selfConsistencyCallback(Diagonalizer &diagonalizer) = 0;
	};

	/** Constructs a Solver::Diagonalizer. */
	Diagonalizer();

	/** Set SelfConsistencyCallback. If never called, the self-consistency
	 *  loop will not be run.
	 *
	 *  @param selfConsistencyCallback A SelfConsistencyCallback that will
	 *  be called after the Model has been diagonalized. The callback
	 *  should calculate relevant quantities, modify the Model if
	 *  necessary, and return false if further iteration is necessary. If
	 *  true is returned, self-consistency is considered to be reached and
	 *  the iteration stops. */
	void setSelfConsistencyCallback(
		SelfConsistencyCallback &selfConsistencyCallback
	);

	/** Set the maximum number of iterations for the self-consistency loop.
	 *  Only used if Diagonalizer::setSelfConsistencyCallback() has been
	 *  called with a non-nullptr argument. If the self-consistency
	 *  callback does not return true, maxIterations determines the maximum
	 *  number of times it is called.
	 *
	 *  @param maxIterations Maximum number of iterations to use in a
	 *  self-consistent callculation. */
	void setMaxIterations(int maxIterations);

	/** Set if you want to use GPU acceleration provided by CUDA routines.
	 *
	 *  @param useGPUAcceleration Turns GPU acceleration for the Diagonalizer 
	 *  solver off or on. */
	void setUseGPUAcceleration(bool useGPUAcceleration);

	/** Run calculations. Diagonalizes ones if no self-consistency callback
	 *  have been set, or otherwise multiple times until self-consistencey
	 *  or maximum number of iterations has been reached. */
	void run();

	/** Get eigenvalues. Eigenvalues are ordered in accending order.
	 *
	 *  @return A pointer to the internal storage for the eigenvalues. */
	const CArray<double>& getEigenValues();

	/** Get eigenvalues. Eigenvalues are ordered in accending order. Same
	 *  as getEigenValues(), but with write access. Use with caution.
	 *
	 *  @return A pointer to the internal storage for the eigenvalues. */
	CArray<double>& getEigenValuesRW();

	/** Get eigenvectors. The eigenvectors are stored successively in
	 *  memory, with the eigenvector corresponding to the smallest
	 *  eigenvalue occupying the 'basisSize' first positions, the second
	 *  occupying the next 'basisSize' elements, and so forth, where
	 *  'basisSize' is the basis size of the Model.
	 *
	 *  @return A pointer to the internal storage for the eigenvectors. **/
	const CArray<std::complex<double>>& getEigenVectors();

	/** Get eigenvectors. The eigenvectors are stored successively in
	 *  memory, with the eigenvector corresponding to the smallest
	 *  eigenvalue occupying the 'basisSize' first positions, the second
	 *  occupying the next 'basisSize' elements, and so forth, where
	 *  'basisSize' is the basis size of the Model. Same as
	 *  getEigenVectors(), but with write access. Use with caution.
	 *
	 *  @return A pointer to the internal storage for the eigenvectors. **/
	CArray<std::complex<double>>& getEigenVectorsRW();

	/** Get eigenvalue for a specific state.
	 *
	 *  @param state The state number, ordered in accending order.
	 *
	 *  @return The eigenvalue for the given state. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\f$ and physical index
	 * \f$x\f$: \f$\Psi_{n}(x)\f$.
	 *
	 *  @param state Eigenstate number \f$n\f$.
	 *  @param index Physical index \f$x\f$.
	 *
	 *  @return The amplitude \f$\Psi_{n}(x)\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);
private:
	/** pointer to array containing Hamiltonian. */
	CArray<std::complex<double>> hamiltonian;

	/** Pointer to array containing eigenvalues.*/
	CArray<double> eigenValues;

	/** Pointer to array containing eigenvectors. */
	CArray<std::complex<double>> eigenVectors;

	/** Pointer to array containing the basis transformation. Only used for
	 *  non-orthonormal bases.*/
	CArray<std::complex<double>> basisTransformation;

	/** Maximum number of iterations in the self-consistency loop. */
	int maxIterations;

	/** Enables GPU acceleration for the solver. */
	bool useGPUAcceleration;

	/** SelfConsistencyCallback to call each time a diagonalization has
	 *  been completed. */
	SelfConsistencyCallback *selfConsistencyCallback;

	/** Allocates space for Hamiltonian etc. */
	void init();

	/** Updates Hamiltonian. */
	void update();

	/** Diagonalizes the Hamiltonian. */
	void solve();

	/** Diagonalizes the Hamiltonian using the GPU. */
	void solveGPU();

	/** Setup the basis transformation. */
	void setupBasisTransformation();

	/** Transform the Hamiltonian to an orthonormal basis. */
	void transformToOrthonormalBasis();

	/** Transform the eigen vectors to the original basis. */
	void transformToOriginalBasis();
};

inline void Diagonalizer::setSelfConsistencyCallback(
	SelfConsistencyCallback &selfConsistencyCallback
){
	this->selfConsistencyCallback = &selfConsistencyCallback;
}

inline void Diagonalizer::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

inline const CArray<double>& Diagonalizer::getEigenValues(){
	return eigenValues;
}

inline CArray<double>& Diagonalizer::getEigenValuesRW(){
	return eigenValues;
}

inline const CArray<std::complex<double>>& Diagonalizer::getEigenVectors(){
	return eigenVectors;
}

inline CArray<std::complex<double>>& Diagonalizer::getEigenVectorsRW(){
	return eigenVectors;
}

inline const std::complex<double> Diagonalizer::getAmplitude(
	int state,
	const Index &index
){
	const Model &model = getModel();
	return eigenVectors[model.getBasisSize()*state + model.getBasisIndex(index)];
}

inline double Diagonalizer::getEigenValue(int state){
	return eigenValues[state];
}

inline void Diagonalizer::setUseGPUAcceleration(bool useGPUAcceleration){
	this->useGPUAcceleration = useGPUAcceleration;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
