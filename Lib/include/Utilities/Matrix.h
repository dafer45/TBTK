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
 *  @file Matrix.h
 *  @brief Custom matrix.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MATRIX
#define COM_DAFER45_TBTK_MATRIX

#include "TBTKMacros.h"

#include <complex>

namespace TBTK{

template<typename DataType, unsigned int ROWS = 0, unsigned int COLS = 0>
class Matrix{
public:
	/** Constructor. */
	Matrix();

	/** Copy constructor. */
	Matrix(const Matrix<DataType, ROWS, COLS> &matrix);

	/** Move constructor. */
	Matrix(Matrix<DataType, ROWS, COLS> &&matrix);

	/** Destructor. */
	~Matrix();

	/** Assignment operator. */
	Matrix<DataType, ROWS, COLS>& operator=(
		const Matrix<DataType, ROWS, COLS> &rhs
	);

	/** Move assignment operator. */
	Matrix<DataType, ROWS, COLS>& operator=(
		Matrix<DataType, ROWS, COLS> &&rhs
	);

	/** Returns a constant reference to the data element. */
	const DataType& at(unsigned int row, unsigned int col) const;

	/** Returns a reference to the data element. */
	DataType& at(unsigned int row, unsigned int col);

	/** Get number of rows. */
	unsigned int getNumRows() const;

	/** Get number of columns. */
	unsigned int getNumCols() const;
private:
	/** Data. */
	DataType data[ROWS*COLS];
};

template<typename DataType>
class Matrix<DataType, 0, 0>{
public:
	/** Constructor. */
	Matrix(unsigned int rows, unsigned int cols);

	/** Copy constructor. */
	Matrix(const Matrix<DataType, 0, 0> &matrix);

	/** Move constructor. */
	Matrix(Matrix<DataType, 0, 0> &&matrix);

	/** Destructor. */
	~Matrix();

	/** Assignment operator. */
	Matrix<DataType, 0, 0>& operator=(const Matrix<DataType, 0, 0> &rhs);

	/** Move assignment operator. */
	Matrix<DataType, 0, 0>& operator=(Matrix<DataType, 0, 0> &&rhs);

	/** Returns a constant reference to the data element. */
	const DataType& at(unsigned int row, unsigned int col) const;

	/** Returns a reference to the data element. */
	DataType& at(unsigned int row, unsigned int col);

	/** Get number of rows. */
	unsigned int getNumRows() const;

	/** Get number of columns. */
	unsigned int getNumCols() const;

	/** Multiplication operator. */
	const Matrix<DataType, 0, 0> operator*(
		const Matrix<DataType, 0, 0> &rhs
	) const;
private:
	/** Data. */
	DataType *data;

	/** Number of rows. */
	unsigned int rows;

	/** Number of columns. */
	unsigned int cols;
};

template<>
class Matrix<std::complex<double>, 0, 0>{
public:
	/** Constructor. */
	Matrix(unsigned int rows, unsigned int cols);

	/** Copy constructor. */
	Matrix(const Matrix<std::complex<double>, 0, 0> &matrix);

	/** Move constructor. */
	Matrix(Matrix<std::complex<double>, 0, 0> &&matrix);

	/** Destructor. */
	~Matrix();

	/** Assignment operator. */
	Matrix<std::complex<double>, 0, 0>& operator=(
		const Matrix<std::complex<double>, 0, 0> &rhs
	);

	/** Move assignment operator. */
	Matrix<std::complex<double>, 0, 0>& operator=(
		Matrix<std::complex<double>, 0, 0> &&rhs
	);

	/** Returns a constant reference to the data element. */
	const std::complex<double>& at(
		unsigned int row,
		unsigned int col
	) const;

	/** Returns a reference to the data element. */
	std::complex<double>& at(unsigned int row, unsigned int col);

	/** Get number of rows. */
	unsigned int getNumRows() const;

	/** Get number of columns. */
	unsigned int getNumCols() const;

	/** Multiplication operator. */
	const Matrix<std::complex<double>, 0, 0> operator*(
		const Matrix<std::complex<double>, 0, 0> &rhs
	) const;

	/** Invert. */
	void invert();

	/** Determinant. */
	std::complex<double> determinant();
private:
	/** Data. */
	std::complex<double> *data;

	/** Number of rows. */
	unsigned int rows;

	/** Number of columns. */
	unsigned int cols;
};

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::Matrix(){
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::Matrix(const Matrix<DataType, ROWS, COLS> &matrix){
	for(unsigned int n = 0; n < ROWS*COLS; n++)
		data[n] = matrix.data[n];
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::Matrix(Matrix<DataType, ROWS, COLS> &&matrix){
	for(unsigned int n = 0; n < ROWS*COLS; n++)
		data[n] = matrix.data[n];
}

template<typename DataType>
Matrix<DataType, 0, 0>::Matrix(unsigned int rows, unsigned int cols){
	this->rows = rows;
	this->cols = cols;
	data = new DataType[rows*cols];
}

template<typename DataType>
Matrix<DataType, 0, 0>::Matrix(const Matrix<DataType, 0, 0> &matrix){
	rows = matrix.rows;
	cols = matrix.cols;
	data = new DataType[rows*cols];
	for(unsigned int n = 0; n < rows*cols; n++)
		data[n] = matrix.data[n];
}

template<typename DataType>
Matrix<DataType, 0, 0>::Matrix(Matrix<DataType, 0, 0> &&matrix){
	rows = matrix.rows;
	cols = matrix.cols;
	data = matrix.data;
	matrix.data = nullptr;
}

inline Matrix<std::complex<double>, 0, 0>::Matrix(unsigned int rows, unsigned int cols){
	this->rows = rows;
	this->cols = cols;
	data = new std::complex<double>[rows*cols];
}

inline Matrix<std::complex<double>, 0, 0>::Matrix(
	const Matrix<std::complex<double>, 0, 0> &matrix
){
	rows = matrix.rows;
	cols = matrix.cols;
	data = new std::complex<double>[rows*cols];
	for(unsigned int n = 0; n < rows*cols; n++)
		data[n] = matrix.data[n];
}

inline Matrix<std::complex<double>, 0, 0>::Matrix(
	Matrix<std::complex<double>, 0, 0> &&matrix
){
	rows = matrix.rows;
	cols = matrix.cols;
	data = matrix.data;
	matrix.data = nullptr;
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::~Matrix(){
}

template<typename DataType>
Matrix<DataType, 0, 0>::~Matrix(){
	if(data != nullptr)
		delete [] data;
}

inline Matrix<std::complex<double>, 0, 0>::~Matrix(){
	if(data != nullptr)
		delete [] data;
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>& Matrix<DataType, ROWS, COLS>::operator=(
	const Matrix<DataType, ROWS, COLS> &rhs
){
	if(this != &rhs){
		for(unsigned int n = 0; n < ROWS*COLS; n++)
			data[n] = rhs.data[n];
	}

	return *this;
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>& Matrix<DataType, ROWS, COLS>::operator=(
	Matrix<DataType, ROWS, COLS> &&rhs
){
	if(this != &rhs){
		for(unsigned int n = 0; n < ROWS*COLS; n++)
			data[n] = rhs.data[n];
	}

	return *this;
}

template<typename DataType>
Matrix<DataType, 0, 0>& Matrix<DataType, 0, 0>::operator=(
	const Matrix<DataType, 0, 0> &rhs
){
	if(this != &rhs){
		rows = rhs.rows;
		cols = rhs.cols;

		if(data != nullptr)
			delete [] data;

		data = new DataType[rows*cols];
		for(unsigned int n = 0; n < rows*cols; n++)
			data[n] = rhs.data[n];
	}
}

template<typename DataType>
Matrix<DataType, 0, 0>& Matrix<DataType, 0, 0>::operator=(
	Matrix<DataType, 0, 0> &&rhs
){
	if(this != &rhs){
		rows = rhs.rows;
		cols = rhs.cols;

		if(data != nullptr)
			delete [] data;

		data = rhs.data;
		rhs.data = nullptr;
	}
}

inline Matrix<std::complex<double>, 0, 0>& Matrix<std::complex<double>, 0, 0>::operator=(
	const Matrix<std::complex<double>, 0, 0> &rhs
){
	if(this != &rhs){
		rows = rhs.rows;
		cols = rhs.cols;

		if(data != nullptr)
			delete [] data;

		data = new std::complex<double>[rows*cols];
		for(unsigned int n = 0; n < rows*cols; n++)
			data[n] = rhs.data[n];
	}
}

inline Matrix<std::complex<double>, 0, 0>& Matrix<std::complex<double>, 0, 0>::operator=(
	Matrix<std::complex<double>, 0, 0> &&rhs
){
	if(this != &rhs){
		rows = rhs.rows;
		cols = rhs.cols;

		if(data != nullptr)
			delete [] data;

		data = rhs.data;
		rhs.data = nullptr;
	}
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
const DataType& Matrix<DataType, ROWS, COLS>::at(
	unsigned int row,
	unsigned int col
) const{
	return data[row + ROWS*col];
}

template<typename DataType>
const DataType& Matrix<DataType, 0, 0>::at(
	unsigned int row,
	unsigned int col
) const{
	return data[row + rows*col];
}

inline const std::complex<double>& Matrix<std::complex<double>, 0, 0>::at(
	unsigned int row,
	unsigned int col
) const{
	return data[row + rows*col];
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
DataType& Matrix<DataType, ROWS, COLS>::at(
	unsigned int row,
	unsigned int col
){
	return data[row + ROWS*col];
}

template<typename DataType>
DataType& Matrix<DataType, 0, 0>::at(
	unsigned int row,
	unsigned int col
){
	return data[row + rows*col];
}

inline std::complex<double>& Matrix<std::complex<double>, 0, 0>::at(
	unsigned int row,
	unsigned int col
){
	return data[row + rows*col];
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
unsigned int Matrix<DataType, ROWS, COLS>::getNumRows() const{
	return ROWS;
}

template<typename DataType>
unsigned int Matrix<DataType, 0, 0>::getNumRows() const{
	return rows;
}

inline unsigned int Matrix<std::complex<double>, 0, 0>::getNumRows() const{
	return rows;
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
unsigned int Matrix<DataType, ROWS, COLS>::getNumCols() const{
	return COLS;
}

template<typename DataType>
unsigned int Matrix<DataType, 0, 0>::getNumCols() const{
	return cols;
}

inline unsigned int Matrix<std::complex<double>, 0, 0>::getNumCols() const{
	return cols;
}

template<typename DataType>
inline const Matrix<DataType, 0, 0> Matrix<DataType, 0, 0>::operator*(
	const Matrix<DataType, 0, 0> &rhs
) const{
	TBTKAssert(
		cols == rhs.rows,
		"Matrix::operator*()",
		"Incompatible matrix dimensions.",
		"The matrix dimensions are " << rows << "x" << cols << " and "
		<< rhs.rows << "x" << rhs.cols << "\n"
	);

	Matrix<DataType> result(rows, rhs.cols);
	for(unsigned int row = 0; row < rows; row++){
		for(unsigned int col = 0; col < rhs.cols; col++){
			result.at(row, col) = 0.;

			for(unsigned int n = 0; n < cols; n++){
				result.at(row, col)
					+= at(row, n)*rhs.at(n, col);
			}
		}
	}

	return result;
}

inline const Matrix<std::complex<double>, 0, 0> Matrix<std::complex<double>, 0, 0>::operator*(
	const Matrix<std::complex<double>, 0, 0> &rhs
) const{
	TBTKAssert(
		cols == rhs.rows,
		"Matrix::operator*()",
		"Incompatible matrix dimensions.",
		"The matrix dimensions are " << rows << "x" << cols << " and "
		<< rhs.rows << "x" << rhs.cols << "\n"
	);

	Matrix<std::complex<double>> result(rows, rhs.cols);
	for(unsigned int row = 0; row < rows; row++){
		for(unsigned int col = 0; col < rhs.cols; col++){
			result.at(row, col) = 0.;

			for(unsigned int n = 0; n < cols; n++){
				result.at(row, col)
					+= at(row, n)*rhs.at(n, col);
			}
		}
	}

	return result;
}

extern "C"{
	void zgetrf_(
		int *M,
		int *N,
		std::complex<double> *A,
		int *lda,
		int *ipiv,
		int *info
	);
	void zgetri_(
		int *N,
		std::complex<double> *A,
		int *lda,
		int *ipiv,
		std::complex<double> *work,
		int *lwork,
		int *info
	);
};

inline void Matrix<std::complex<double>, 0, 0>::invert(){
	TBTKAssert(
		rows == cols,
		"Matrix::invert()",
		"Invalid matrix dimension. Only square matrices can be"
		<< " inverted, but the matrix has size " << rows << "x" << cols
		<< "\n",
		""
	);

	int *ipiv = new int[std::min(rows, cols)];
	int lwork = rows*cols;
	std::complex<double> *work = new std::complex<double>[lwork];
	int info;

	zgetrf_((int*)&rows, (int*)&cols, data, (int*)&rows, ipiv, &info);

	if(info < 0){
		TBTKExit(
			"Matrix::invert()",
			"Argument '" << -info << "' to zgetrf_() is invlid.",
			"This should never happen, contact the developer."
		);
	}
	else if(info > 0){
		TBTKExit(
			"Matrix::invert()",
			"Unable to invert matrix since it is signular.",
			""
		);
	}

	zgetri_((int*)&rows, data, (int*)&rows, ipiv, work, &lwork, &info);
	TBTKAssert(
		info == 0,
		"Matrix::invert()",
		"Inversion failed with error code 'INFO = " << info << "'.",
		"See the documentation for the lapack function zgetri_() for"
		<< " further information."
	);

	delete [] ipiv;
	delete [] work;
}

inline std::complex<double> Matrix<std::complex<double>, 0, 0>::determinant(){
	TBTKAssert(
		rows == cols,
		"Matrix::determinant()",
		"Invalid matrix dimension. The determinant can only be"
		<< " calculated for square matrices, but the matrix has size "
		<< rows << "x" << cols << "\n",
		""
	);

	std::complex<double> *copy = new std::complex<double>[rows*cols];
	for(unsigned int n = 0; n < rows*cols; n++)
		copy[n] = data[n];

	int *ipiv = new int[std::min(rows, cols)];
//	int lwork = rows*cols;
//	std::complex<double> *work = new std::complex<double>[lwork];
	int info;

	zgetrf_((int*)&rows, (int*)&cols, copy, (int*)&rows, ipiv, &info);

	if(info < 0){
		TBTKExit(
			"Matrix::determinant()",
			"Argument '" << -info << "' to zgetrf_() is invlid.",
			"This should never happen, contact the developer."
		);
	}
	else if(info > 0){
		TBTKExit(
			"Matrix::determinant()",
			"Unable to invert matrix since it is signular.",
			""
		);
	}

	std::complex<double> det = 1.;
	for(unsigned int n = 0; n < rows; n++){
		Streams::out << ipiv[n] << "\n";
		if(ipiv[n]-1 == n)
			det *= copy[rows*n + n];
		else
			det *= -copy[rows*n + n];
	}

	delete [] copy;
	delete [] ipiv;

	return det;
}

};	//End namespace TBTK

#endif
