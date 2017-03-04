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
 *  @file ArnoldiSolver
 *  @brief Solves a Model using the Arnoldi method.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_ARNOLDI_SOLVER
#define COM_DAFER45_ARNOLDI_SOLVER

#include "Model.h"
#include "Solver.h"

#include "slu_zdefs.h"

#include <complex>

namespace TBTK{

/** The ArnoldiSolver can be used to calculate a few eigenvalues and
 *  eigenvectors around a given energy.
 */
class ArnoldiSolver : public Solver{
public:
	/** Constructor. */
	ArnoldiSolver();

	/** Destructor. */
	virtual ~ArnoldiSolver();

	/** Enum class describing the different modes of operation.
	 *
	 *  Normal:
	 *      Lanczos like iteration for finding extreme eigenvalues and
	 *      corresponding eigen vectors.
	 *
	 *  ShiftAndInvert:
	 *      Shift and invert iteration for finding eigen values around a
	 *      given value and the corresponding eigen vectors. */
	enum class Mode {Normal, ShiftAndInvert};

	/** Set mode of operation. */
	void setMode(Mode mode);

	/** Get mode of operation. */

	/** Set the number of eigenvalues to calculate. */
	void setNumEigenValues(int numEigenValues);

	/** Get number of eigenvalues. */
	int getNumEigenValues() const;

	/** Set wether eigen vectors should be calculated. */
	void setCalculateEigenVectors(bool calculateEigenVectors);

	/** Get wether eigen vectors are calculated or not. */
	bool getCalculateEigenVectors() const;

	/** Set the number of Lanczos vectors to use. (Dimension of the Krylov
	 *  space). */
	void setNumLanczosVectors(int numLanczosVectors);

	/** Set the accpeted tolerance. */
	void setTolerance(double tolerance);

	/** Set maimum number of iterations in the implicitly restarted Arnoldi
	 *  algorithm. */
	void setMaxIterations(int maxIterations);

	/** Set shift. */
	void setCentralValue(double centralValue);

	/** Run the implicitly restarted Arnoldi algorithm. */
	void run();

	/** Get eigenValues. */
	const std::complex<double>* getEigenValues() const;

	/** Get eigen value. */
	const double getEigenValue(int state) const;

	/** Get amplitude for given eigen vector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$.
	 *  @param state Eigen state number \f$n\f$.
	 *  @param index Physical index \f$\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);
private:
	/** Mode of operation. */
	Mode mode;

	/** Number of eigenvalues to calculate (Arnoldi variable). */
	int numEigenValues;

	/** Flag indicating whether eigenvectors should be calculated. (Arnoldi
	 *  variable). */
	bool calculateEigenVectors;

	/** Number of Lanczos vectors to use, i.e. the dimension of the Krylov
	 *  space. (Arnoldi variable). */
	int numLanczosVectors;

	/** Energy around which the eigenvalues and eigenvectors are
	 *  calculated. (Arnoldi variable). */
	double shift;

	/** Accepted tolerance. Machine tolerance is used if tolerance <= 0.
	 *  (Arnoldi variable). */
	double tolerance;

	/** Maximum number of iterations in the implicitly restarted Arnoldi
	 *  algorithm. (Arnoldi variable). */
	int maxIterations;

	/** Residuals. (Arnoldi variable). */
	std::complex<double> *residuals;

	/** Eigen values. (Arnoldi variable). */
	std::complex<double> *eigenValues;

	/** Eigen vectors. (Arnoldi variable). */
	std::complex<double> *eigenVectors;

	/** Hamiltonian. (SuperLU variable). */
	SuperMatrix *hamiltonian;

	/** Vector used to represent both x and b in Ax = b. (SuperLU
	 *  variable). */
	SuperMatrix *vector;

	/** Lower triangular matrix L in LU decomposition of the Hamiltonian.
	 *  (SuperLU variable). */
	SuperMatrix *lowerTriangular;

	/** Upper triangular matrix U in LU decomposition of the Hamiltonian.
	 *  (SuperLU variable). */
	SuperMatrix *upperTriangular;

	/** Row permutation matrix Pr in LU decomposition LU = PrHPc.
	 *  Permutations are performed to enhance performance and increase
	 *  numerical stability. (SuperLU variable). */
	int *rowPermutations;

	/** Column permutation matrix Pc in LU decomposition LU = PrHPc.
	 *  Permutations are performed to enhance performance and increase
	 *  numerical stability. (SuperLU variable). */
	int *colPermutations;

	/** Options for LU decomposition. (SuperLU variable). */
	superlu_options_t *options;

	/** Status for LU decomposition. (SuperLU variable). */
	SuperLUStat_t *stat;

	/** Initialize solver for normal mode. Setting up SuperLU. (SuperLU
	 *  routine). */
	void initNormal();

	/** Initialize solver for shift and invert mode. Setting up SuperLU.
	 *  (SuperLU routine). */
	void initShiftAndInvert();

	/** Perform LU decomposition of Hamiltonian. (SuperLU routine). */
	void performLUFactorization();

	/** Run implicitly restarted Arnoldi loop using normal mode. (Arnoldi
	 *  routine). */
	void arnoldiLoopNormal();

	/** Run implicitly restarted Arnoldi loop using shift and invert mode.
	 *  (Arnoldi routine). */
	void arnoldiLoopShiftAndInvert();

	/** Sort eigen values and eigen vectors in accending order according to
	 *  the real part of the eigen values. */
	void sort();

	/** Merge sort helper function for ArnoldiSolver::sort(). */
	void mergeSortSplit(
		std::complex<double> *dataIn,
		std::complex<double> *dataOut,
		int *orderIn,
		int *orderOut,
		int first,
		int end
	);

	/** Merge sort helper function for ArnoldiSolver::sort(). */
	void mergeSortMerge(
		std::complex<double> *dataIn,
		std::complex<double> *dataOut,
		int *orderIn,
		int *orderOut,
		int first,
		int middle,
		int end
	);
};

inline void ArnoldiSolver::setMode(Mode mode){
	this->mode = mode;
}

inline void ArnoldiSolver::setNumEigenValues(int numEigenValues){
	this->numEigenValues = numEigenValues;
}

inline int ArnoldiSolver::getNumEigenValues() const{
	return numEigenValues;
}

inline void ArnoldiSolver::setCalculateEigenVectors(bool calculateEigenVectors){
	this->calculateEigenVectors = calculateEigenVectors;
}

inline bool ArnoldiSolver::getCalculateEigenVectors() const{
	return calculateEigenVectors;
}

inline void ArnoldiSolver::setNumLanczosVectors(int numLanczosVectors){
	this->numLanczosVectors = numLanczosVectors;
}

inline void ArnoldiSolver::setTolerance(double tolerance){
	this->tolerance = tolerance;
}

inline void ArnoldiSolver::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

inline void ArnoldiSolver::setCentralValue(double centralValue){
	shift = centralValue;
}

inline const std::complex<double>* ArnoldiSolver::getEigenValues() const{
	return eigenValues;
}

inline const double ArnoldiSolver::getEigenValue(int state) const{
	return real(eigenValues[state]);
}

inline const std::complex<double> ArnoldiSolver::getAmplitude(
	int state,
	const Index &index
){
	Model *model = getModel();
	return eigenVectors[model->getBasisSize()*state + model->getBasisIndex(index)];
}

};	//End of namesapce TBTK

#endif
