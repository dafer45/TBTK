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

namespace TBTK{

template<typename DataType, unsigned int ROWS, unsigned int COLS>
class Matrix{
public:
	/** Constructor. */
	Matrix();

	/** Destructor. */
	~Matrix();

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

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::Matrix(){
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
Matrix<DataType, ROWS, COLS>::~Matrix(){
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
const DataType& Matrix<DataType, ROWS, COLS>::at(
	unsigned int row,
	unsigned int col
) const{
	return data[row + ROWS*col];
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
DataType& Matrix<DataType, ROWS, COLS>::at(
	unsigned int row,
	unsigned int col
){
	return data[row + ROWS*col];
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
unsigned int Matrix<DataType, ROWS, COLS>::getNumRows() const{
	return ROWS;
}

template<typename DataType, unsigned int ROWS, unsigned int COLS>
unsigned int Matrix<DataType, ROWS, COLS>::getNumCols() const{
	return COLS;
}

};	//End namespace TBTK

#endif
