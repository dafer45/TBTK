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
 *  @brief Solves Hx = y for x, where H is a SparseMatrix.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LU_SOLVER
#define COM_DAFER45_TBTK_LU_SOLVER

#include "Communicator.h"
#include "Matrix.h"
#include "SparseMatrix.h"

#include <complex>

#include "slu_zdefs.h"

namespace TBTK{

/** @brief Solves Hx = y for x, where H is a SparseMatrix. */
class LUSolver : public Communicator{
public:
	/** Enum class for specifying the data type of the matrix. Used since
	 *  complex matrices that are purely real are converted to double type
	 *  for optimization. */
	enum class DataType {None, Double, ComplexDouble};

	/** Constructor */
	LUSolver();

	/** Destructor. */
	~LUSolver();

	/** Set matrix. */
	void setMatrix(const SparseMatrix<double> &sparseMatrix);

	/** Set matrix. */
	void setMatrix(const SparseMatrix<std::complex<double>> &sparseMatrix);

	/** Get matrix data type. */
	DataType getMatrixDataType() const;

	/** Solve. */
	void solve(Matrix<double> &b);

	/** Solve. */
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
