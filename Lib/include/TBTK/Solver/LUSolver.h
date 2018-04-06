/* Copyright 2017 Kristofer Björnson
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
 *  @file LUSolver.h
 *  @brief Solves Mx = b for x, where M is a SparseMatrix.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LU_SOLVER
#define COM_DAFER45_TBTK_LU_SOLVER

#include "TBTK/Communicator.h"
#include "TBTK/Matrix.h"
#include "TBTK/SparseMatrix.h"

#include <complex>

#include "slu_zdefs.h"

namespace TBTK{

/** @brief Solves Mx = b for x, where M is a SparseMatrix. */
class LUSolver : public Communicator{
public:
	/** Enum class for specifying the data type of the matrix. Used since
	 *  complex matrices that are purely real are converted to double type
	 *  for optimization. */
	enum class DataType {None, Double, ComplexDouble};

	/** Constructs a Solver::LUSolver. */
	LUSolver();

	/** Destructor. */
	~LUSolver();

	/** Set matrix.
	 *
	 *  @param sparseMatrix Matrix \f$M\f$ to use in the equation \f$Mx =
	 *  b\f$. */
	void setMatrix(const SparseMatrix<double> &sparseMatrix);

	/** Set matrix.
	 *
	 *  @param sparseMatrix Matrix \f$H\f$ to use in the equation \f$Mx =
	 *  b\f$. */
	void setMatrix(const SparseMatrix<std::complex<double>> &sparseMatrix);

	/** Get matrix data type. Is None before a matrix has been set, Double
	 *  if a real matrix has been set, and ComplexDouble if a complex
	 *  matrix has been set. Note that a SparseMatrix<std::complex<double>>
	 *  is stored as Double internally if the matrix is real.
	 *
	 *  @return The data type that is used to store the data. */
	DataType getMatrixDataType() const;

	/** Solve for \f$x\f$ in the equation \f$Mx = b\f$.
	 *
	 *  @param b The vector \f$b\f$. Contains the answer \f$x\f$ when
	 *  finished. */
	void solve(Matrix<double> &b);

	/** Solve for \f$x\f$ in the equation \f$Mx = b\f$.
	 *
	 *  @param b The vector \f$b\f$. Contains the answer \f$x\f$ when
	 *  finished. */
	void solve(Matrix<std::complex<double>> &b);
private:
	/** Pointer to lower triangular matrix. */
	SuperMatrix *L;

	/** Pointer to upper triangular matrix. */
	SuperMatrix *U;

	/** Row permutations. */
	int *rowPermutations;

	/** Column permutations. */
	int *columnPermutations;

	/** SuperLU statistics. */
	SuperLUStat_t *statistics;

	/** Get matrix data type. */
	DataType matrixDataType;

	/** Allocate permutation matrices. */
	void allocatePermutationMatrices(
		unsigned int numRows,
		unsigned int numColumns
	);

	/** Initialize statistics. */
	void initStatistics();

	/** Allocate LU matrices. */
	void allocateLUMatrices();

	/** Initialize SuperLU options and permutation matrices. */
	void initOptionsAndPermutationMatrices(
		superlu_options_t &options,
		SuperMatrix &matrix
	);

	/** Perform LU factorization. */
	void performLUFactorization(SuperMatrix &matrix);

	/** Check assertments for solve(). */
	void checkSolveAssert(unsigned int numRows);

	/** Check for Xgstrf errors. */
	void checkXgstrfErrors(
		int info,
		std::string functionName,
		int numColumns
	);

	/** Check for Xgstrs errors. */
	void checkXgstrsErrors(int info, std::string functionName);
};

inline LUSolver::DataType LUSolver::getMatrixDataType() const{
	return matrixDataType;
}

};	//End of namespace TBTK

#endif
