/* Copyright 2018 Kristofer Björnson
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

/** @file PadeApproximator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PadeApproximator.h"

using namespace std;

namespace TBTK{

vector<
	Polynomial<complex<double>, complex<double>, int>
> PadeApproximator::approximate(
	const vector<complex<double>> &values,
	const vector<complex<double>> &arguments
){
	TBTKAssert(
		values.size() == arguments.size(),
		"PadeApproximator::approximate()",
		"Incompatible sizes. The size of 'values' (" << values.size()
		<< ") must be the same as the size of 'arguments' ("
		<< arguments.size() << ").",
		""
	);
	TBTKAssert(
		values.size() > numeratorDegree + denumeratorDegree,
		"PadeApproximator::approximate()",
		"The number of values and arguments (" << values.size() << ")"
		<< " must be larger than 'numeratorDegree + denumeratorDegree="
		<< numeratorDegree + denumeratorDegree << "'.",
		""
	);

	unsigned int numRows = arguments.size();
	unsigned int numColumns = 1 + numeratorDegree + denumeratorDegree;

	complex<double> *matrix = new complex<double>[numRows*numColumns];
	complex<double> *vector = new complex<double>[numRows];

	for(unsigned int row = 0; row < numRows; row++){
		for(unsigned int column = 0; column < numColumns; column++){
			if(column == 0){
				matrix[row + numRows*column] = 1.;
			}
			else if(column < numeratorDegree+1){
				matrix[row + numRows*column]
					= pow(arguments[row], column);
			}
			else if(column == numeratorDegree+1){
				matrix[row + numRows*column] = -values[row];
			}
			else{
				matrix[row + numRows*column]
					= -values[row]*pow(
						arguments[row],
						column - numeratorDegree - 1
					);
			}
		}

		vector[row] = -values[row]*pow(
			arguments[row],
			denumeratorDegree + 1
		);
	}

	executeLeastSquare(matrix, vector, numRows, numColumns);

	std::vector<Polynomial<complex<double>, complex<double>, int>> polynomials;
	polynomials.push_back(
		Polynomial<complex<double>, complex<double>, int>(1)
	);
	polynomials.push_back(
		Polynomial<complex<double>, complex<double>, int>(1)
	);
	for(unsigned int n = 0; n < numeratorDegree+1; n++)
		polynomials[0].addTerm(vector[n], {(int)n});
	for(unsigned int n = 0; n < denumeratorDegree; n++)
		polynomials[1].addTerm(vector[n+numeratorDegree+1], {(int)n});

	delete [] matrix;
	delete [] vector;

	return polynomials;
}

extern "C" int ilaenv_(
	int *ISPEC,
	char *NAME,
	char *OPTS,
	int *N1,
	int *N2,
	int *N3,
	int *N4
);

extern "C" void zgelsd_(
	int *M,
	int *N,
	int *NRHS,
	complex<double> *A,
	int *LDA,
	complex<double> *B,
	int *LDB,
	double *S,
	double *RCOND,
	int *RANK,
	complex<double> *WORK,
	int *LWORK,
	double *RWORK,
	int *IWORK,
	int *INFO
);

void PadeApproximator::executeLeastSquare(
	complex<double> *matrix,
	complex<double> *vector,
	unsigned int numRows,
	unsigned int numColumns
){
	//ILAENV parameters.
	int ISPEC = 9;
	char NAME[] = "zgelsd";
	char OPTS[] = "";
	int N1 = numRows;
	int N2 = numColumns;
	int N3 = 1;
	int N4 = -1;
	int SMLSIZ = ilaenv_(&ISPEC, NAME, OPTS, &N1, &N2, &N3, &N4);
	if(SMLSIZ < 0){
		TBTKExit(
			"PadeApproximator::executeLeastSquare()",
			"Invalid argument to ilaenv_() at position '"
			<< -SMLSIZ << ".",
			"This should never happen, contact the developer."
		);
	}

	int M = numRows;
	int N = numColumns;
	int NRHS = 1;
	int LDA = numRows;
	double *S = new double[min(numRows, numColumns)];
	double RCOND;
	int RANK;
	int LWORK = max(numRows, numColumns)*(2 + NRHS);
	complex<double> *WORK = new complex<double>[max(1, LWORK)];
	int NLVL = max(0, (int)log2(min(numRows, numColumns)/(SMLSIZ + 1)) + 1);
	int LRWORK;
	if(numRows >= numColumns){
		LRWORK = 10*numColumns
			+ 2*numColumns*SMLSIZ
			+ 8*numColumns*NLVL
			+ 3*SMLSIZ*NRHS
			+ pow(SMLSIZ + 1, 2);
	}
	else{
		LRWORK = 10*numRows
			+ 2*numRows*SMLSIZ
			+ 8*numRows*NLVL
			+ 3*SMLSIZ*NRHS
			+ pow(SMLSIZ + 1, 2);
	}
	double *RWORK = new double[max(1, LRWORK)];
	int LIWORK = max(
		1,
		(int)(
			3*min(numRows, numColumns)*NLVL
			+ 11*min(numRows, numColumns)
		)
	);
	int *IWORK = new int[max(1, LIWORK)];
	int INFO;

	Streams::out << M << "\n";
	zgelsd_(
		&M,
		&N,
		&NRHS,
		matrix,
		&LDA,
		vector,
		&M,
		S,
		&RCOND,
		&RANK,
		WORK,
		&LWORK,
		RWORK,
		IWORK,
		&INFO
	);

	delete [] S;
	delete [] WORK;
	delete [] RWORK;
	delete [] IWORK;
}

};	//End of namespace TBTK
