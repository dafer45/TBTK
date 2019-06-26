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
 *  @file SparseMatrix.h
 *  @brief Sparse matrix.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SPARSE_MATRIX
#define COM_DAFER45_TBTK_SPARSE_MATRIX

#include "TBTK/TBTKMacros.h"

#include <algorithm>
#include <tuple>
#include <vector>

namespace TBTK{

template<typename DataType>
class SparseMatrix{
public:
	/** Enum class for determining the storage format. */
	enum class StorageFormat {CSR, CSC};

	/** Constructor. */
	SparseMatrix();

	/** Constructor. */
	SparseMatrix(StorageFormat);

	/** Copy constructor. */
	SparseMatrix(const SparseMatrix &sparseMatrix);

	/** Copy constructor. */
	SparseMatrix(SparseMatrix &&sparseMatrix);

	/** Constructor. */
	SparseMatrix(
		StorageFormat storageFormat,
		unsigned int numRows,
		unsigned int numCols
	);

	/** Assignment operator. */
	SparseMatrix& operator=(const SparseMatrix &sparseMatrix);

	/** Move assignment operator. */
	SparseMatrix& operator=(SparseMatrix &&sparseMatrix);

	/** Destructor. */
	~SparseMatrix();

	/** Add matrix element. */
	void add(unsigned int row, unsigned int col, const DataType &value);

	/** Set StorageFormat. */
	void setStorageFormat(StorageFormat storageFormat);

	/** Get number of rows. */
	unsigned int getNumRows() const;

	/** Get number of columns. */
	unsigned int getNumColumns() const;

	/** Get number of CSR matrix elements. */
	unsigned int getCSRNumMatrixElements() const;

	/** Get number of CSC matrix elements. */
	unsigned int getCSCNumMatrixElements() const;

	/** Get CSR row pointers. */
	const unsigned int* getCSRRowPointers() const;

	/** Get CSC column pointers. */
	const unsigned int* getCSCColumnPointers() const;

	/** Get CSR columns. */
	const unsigned int* getCSRColumns() const;

	/** Get CSC columns. */
	const unsigned int* getCSCRows() const;

	/** Get CSR values. */
	const DataType* getCSRValues() const;

	/** Get CSC values. */
	const DataType* getCSCValues() const;

	/** Construct the sparse matrix. */
	void construct();

	/** Addition assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the right hand side has been added.
	 */
	SparseMatrix& operator+=(const SparseMatrix &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new SparseMatrix that is the sum of this SparseMatrix and
	 *  the right hand side. */
	SparseMatrix operator+(const SparseMatrix &rhs) const;

	/** Subtraction assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the right hand side has been
	 *  subtracted. */
	SparseMatrix& operator-=(const SparseMatrix &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new SparseMatrix that is the difference between this
	 *  SparseMatrix and the right hand side. */
	SparseMatrix operator-(const SparseMatrix &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new SparseMatrix that is the product of this SparseMatrix
	 *  and the right hand side. */
	SparseMatrix operator*(const SparseMatrix &rhs) const;

	/** Print. */
	void print() const;
private:
	/** Dictionary of keys (DOK). Used to allow for incremental
	 *  construction of the matrix. */
	std::vector<
		std::tuple<unsigned int, unsigned int, DataType>
	> dictionaryOfKeys;

	/** Number of rows and columns. */
	int numRows, numCols;

	/** Flag indicating whether row and column numbers should be allowed to
	 *  change dynamically. */
	bool allowDynamicDimensions;

	/** Storage format. */
	StorageFormat storageFormat;

	/** Row/columns pointer for compressed sparse row/column format
	 *  (CSR/CSC). */
	unsigned int *csxXPointers;

	/** Columns/rows for compressed sparse row/column format (CSR/CSC). */
	unsigned int *csxY;

	/** Values for compressed sparse row/column format (CSR/CSC). */
	DataType *csxValues;

	/** Number of matrix elements for compressed row/columns format
	 *  (CSR/CSC). Is set to -1 when not initialized. */
	int csxNumMatrixElements;
public:
	/** Construct list of lists (LIL) from dictionaryOfKeys. */
	std::vector<
		std::vector<std::tuple<unsigned int, DataType>>
	> constructLIL();

	/** Sort list of lists (LIL). */
	void sortLIL(
		std::vector<
			std::vector<std::tuple<unsigned int, DataType>>
		> &listOfLists
	) const;

	/** Merge list of lists (LIL) entries with the same row and column
	 *  indices. */
	void mergeLIL(
		std::vector<
			std::vector<std::tuple<unsigned int, DataType>>
		> &listOfLists
	) const;

	/** Construct matrix on compressed sparse row format (CSR). */
	void constructCSX();

	/** Adds all matrix elements constructed on CSR/CSC format to LIL.
	 *  Used when a matrix is reconstructed either because new elements has
	 *  been added, or because the format of the matrix is being changed.
	 */
	void convertCSXToLIL();

	/** Helper function for operator*() that performs the actual matrix
	 *  multiplication.
	 *
	 *  @param rhs The left hand side in the matrix multiplication.
	 *  @param rhs The right hand side in the matrix multiplication.
	 *  @param result The SparseMatrix to store the result in. */
	static void multiply(
		const SparseMatrix &lhs,
		const SparseMatrix &rhs,
		SparseMatrix &result
	);
};

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(){
	csxXPointers = nullptr;
	csxY = nullptr;
	csxValues = nullptr;
}

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(StorageFormat storageFormat){
	this->storageFormat = storageFormat;

	numRows = -1;
	numCols = -1;
	allowDynamicDimensions = true;

	csxXPointers = nullptr;
	csxY = nullptr;
	csxValues = nullptr;
	csxNumMatrixElements = -1;
}

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(
	StorageFormat storageFormat,
	unsigned int numRows,
	unsigned int numCols
){
	this->storageFormat = storageFormat;

	this->numRows = numRows;
	this->numCols = numCols;
	allowDynamicDimensions = false;

	csxXPointers = nullptr;
	csxY = nullptr;
	csxValues = nullptr;
	csxNumMatrixElements = -1;
}

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(
	const SparseMatrix &sparseMatrix
){
	storageFormat = sparseMatrix.storageFormat;

	numRows = sparseMatrix.numRows;
	numCols = sparseMatrix.numCols;
	allowDynamicDimensions = sparseMatrix.allowDynamicDimensions;

	csxNumMatrixElements = sparseMatrix.csxNumMatrixElements;
	if(csxNumMatrixElements == -1){
		TBTKAssert(
			sparseMatrix.csxXPointers == nullptr
			&& sparseMatrix.csxY == nullptr
			&& sparseMatrix.csxValues == nullptr,
			"SparseMatrix::SparseMatrix()",
			"Invalid pointers in the original SparseMatrix.",
			"This should never happen, contact the developer."
		);

		csxXPointers = nullptr;
		csxY = nullptr;
		csxValues = nullptr;
	}
	else{
		TBTKAssert(
			sparseMatrix.csxXPointers != nullptr
			&& sparseMatrix.csxY != nullptr
			&& sparseMatrix.csxValues != nullptr,
			"SparseMatrix::SparseMatrix()",
			"Invalid pointers in the original SparseMatrix.",
			"This should never happen, contact the developer."
		);

		switch(storageFormat){
		case StorageFormat::CSR:
			csxXPointers = new unsigned int[numRows+1];
			for(int row = 0; row < numRows+1; row++){
				csxXPointers[row]
					= sparseMatrix.csxXPointers[row];
			}
			break;
		case StorageFormat::CSC:
			csxXPointers = new unsigned int[numCols+1];
			for(int col = 0; col < numCols+1; col++){
				csxXPointers[col]
					= sparseMatrix.csxXPointers[col];
			}
			break;
		default:
			TBTKExit(
				"SparseMatrix::SparseMatrix()",
				"Unknow StorageFormat.",
				"This should never happen, contact the"
				<< " developer."
			);
		}

		csxY = new unsigned int[csxNumMatrixElements];
		csxValues = new DataType[csxNumMatrixElements];
		for(int n = 0; n < csxNumMatrixElements; n++){
			csxY[n] = sparseMatrix.csxY[n];
			csxValues[n] = sparseMatrix.csxValues[n];
		}
	}
}

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(
	SparseMatrix &&sparseMatrix
){
	storageFormat = sparseMatrix.storageFormat;

	numRows = sparseMatrix.numRows;
	numCols = sparseMatrix.numCols;
	allowDynamicDimensions = sparseMatrix.allowDynamicDimensions;

	csxNumMatrixElements = sparseMatrix.csxNumMatrixElements;
	if(csxNumMatrixElements == -1){
		TBTKAssert(
			sparseMatrix.csxXPointers == nullptr
			&& sparseMatrix.csxY == nullptr
			&& sparseMatrix.csxValues == nullptr,
			"SparseMatrix::SparseMatrix()",
			"Invalid pointers in the original SparseMatrix.",
			"This should never happen, contact the developer."
		);

		csxXPointers = nullptr;
		csxY = nullptr;
		csxValues = nullptr;
	}
	else{
		TBTKAssert(
			sparseMatrix.csxXPointers != nullptr
			&& sparseMatrix.csxY != nullptr
			&& sparseMatrix.csxValues != nullptr,
			"SparseMatrix::SparseMatrix()",
			"Invalid pointers in the original SparseMatrix.",
			"This should never happen, contact the developer."
		);

		csxXPointers = sparseMatrix.csxXPointers;
		sparseMatrix.csxXPointers = nullptr;

		csxY = sparseMatrix.csxY;
		sparseMatrix.csxY = nullptr;

		csxValues = sparseMatrix.csxValues;
		sparseMatrix.csxValues = nullptr;
	}
}

template<typename DataType>
inline SparseMatrix<DataType>::~SparseMatrix(){
	if(csxXPointers != nullptr)
		delete [] csxXPointers;
	if(csxY != nullptr)
		delete [] csxY;
	if(csxValues != nullptr)
		delete [] csxValues;
}

template<typename DataType>
inline SparseMatrix<DataType>& SparseMatrix<DataType>::operator=(
	const SparseMatrix &rhs
){
	if(this != &rhs){
		storageFormat = rhs.storageFormat;

		numRows = rhs.numRows;
		numCols = rhs.numCols;
		allowDynamicDimensions = rhs.allowDynamicDimensions;

		if(csxXPointers != nullptr)
			delete [] csxXPointers;
		if(csxY != nullptr)
			delete [] csxY;
		if(csxValues != nullptr)
			delete [] csxValues;

		csxNumMatrixElements = rhs.csxNumMatrixElements;
		if(csxNumMatrixElements == -1){
			TBTKAssert(
				rhs.csxXPointers == nullptr
				&& rhs.csxY == nullptr
				&& rhs.csxValues == nullptr,
				"SparseMatrix::operator=()",
				"Invalid pointers in the original SparseMatrix.",
				"This should never happen, contact the developer."
			);

			csxXPointers = nullptr;
			csxY = nullptr;
			csxValues = nullptr;
		}
		else{
			TBTKAssert(
				rhs.csxXPointers != nullptr
				&& rhs.csxY != nullptr
				&& rhs.csxValues != nullptr,
				"SparseMatrix::SparseMatrix()",
				"Invalid pointers in the original SparseMatrix.",
				"This should never happen, contact the developer."
			);

			switch(storageFormat){
			case StorageFormat::CSR:
				csxXPointers = new unsigned int[numRows+1];
				for(unsigned int row = 0; row < numRows+1; row++){
					csxXPointers[row]
						= rhs.csxXPointers[row];
				}
				break;
			case StorageFormat::CSC:
				csxXPointers = new unsigned int[numCols+1];
				for(unsigned int col = 0; col < numCols+1; col++){
					csxXPointers[col]
						= rhs.csxXPointers[col];
				}
				break;
			default:
				TBTKExit(
					"SparseMatrix::SparseMatrix()",
					"Unknow StorageFormat.",
					"This should never happen, contact the"
					<< " developer."
				);
			}

			csxY = new unsigned int[csxNumMatrixElements];
			csxValues = new DataType[csxNumMatrixElements];
			for(unsigned int n = 0; n < csxNumMatrixElements; n++){
				csxY[n] = rhs.csxY[n];
				csxValues[n] = rhs.csxValues[n];
			}
		}
	}

	return *this;
}

template<typename DataType>
inline SparseMatrix<DataType>& SparseMatrix<DataType>::operator=(
	SparseMatrix &&rhs
){
	if(this != &rhs){
		storageFormat = rhs.storageFormat;

		numRows = rhs.numRows;
		numCols = rhs.numCols;
		allowDynamicDimensions = rhs.allowDynamicDimensions;

		if(csxXPointers != nullptr)
			delete [] csxXPointers;
		if(csxY != nullptr)
			delete [] csxY;
		if(csxValues != nullptr)
			delete [] csxValues;

		csxNumMatrixElements = rhs.csxNumMatrixElements;
		if(csxNumMatrixElements == -1){
			TBTKAssert(
				rhs.csxXPointers == nullptr
				&& rhs.csxY == nullptr
				&& rhs.csxValues == nullptr,
				"SparseMatrix::operator=()",
				"Invalid pointers in the original SparseMatrix.",
				"This should never happen, contact the developer."
			);

			csxXPointers = nullptr;
			csxY = nullptr;
			csxValues = nullptr;
		}
		else{
			TBTKAssert(
				rhs.csxXPointers != nullptr
				&& rhs.csxY != nullptr
				&& rhs.csxValues != nullptr,
				"SparseMatrix::SparseMatrix()",
				"Invalid pointers in the original SparseMatrix.",
				"This should never happen, contact the developer."
			);

			csxXPointers = rhs.csxXPointers;
			rhs.csxXPointers = nullptr;

			csxY = rhs.csxY;
			rhs.csxY = nullptr;

			csxValues = rhs.csxValues;
			rhs.csxValues = nullptr;
		}
	}

	return *this;
}

template<typename DataType>
inline void SparseMatrix<DataType>::add(
	unsigned int row,
	unsigned int col,
	const DataType &value
){
	if(!allowDynamicDimensions){
		TBTKAssert(
			(int)row < numRows && (int)col < numCols,
			"SparseMatrix::add()",
			"Invalid matrix entry. The matrix was constructed"
			" specifiying that the matrix dimension is '"
			<< numRows << "x" << numCols << " but tried to add"
			<< " matrix element with index '(" << row << ", "
			<< col << ")'",
			"Ensure that the matrix elements are in range, or use"
			<< " a constructor which does not fix the matrix"
			<< " dimension."
		);
	}

	dictionaryOfKeys.push_back(std::make_tuple(row, col, value));
}

template<typename DataType>
inline void SparseMatrix<DataType>::setStorageFormat(
	StorageFormat storageFormat
){
	if(this->storageFormat != storageFormat){
		convertCSXToLIL();
		this->storageFormat = storageFormat;
		constructCSX();
	}
}

template<typename DataType>
inline unsigned int SparseMatrix<DataType>::getNumRows() const{
	TBTKAssert(
		numRows != -1,
		"SparseMatrix::getNumRows()",
		"Number of rows not yet determined.",
		""
	);

	return (unsigned int)numRows;
}

template<typename DataType>
inline unsigned int SparseMatrix<DataType>::getNumColumns() const{
	TBTKAssert(
		numRows != -1,
		"SparseMatrix::getNumRows()",
		"Number of rows not yet determined.",
		""
	);

	return (unsigned int)numCols;
}

template<typename DataType>
inline unsigned int SparseMatrix<DataType>::getCSRNumMatrixElements() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSR,
		"SparseMatrix::getCSRNumMatrixElements()",
		"Tried to access CSR number of matrix elements, but the matrix"
		" is not on the CSR storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	if(csxNumMatrixElements == -1)
		return 0;
	else
		return (unsigned int)csxNumMatrixElements;
}

template<typename DataType>
inline unsigned int SparseMatrix<DataType>::getCSCNumMatrixElements() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSC,
		"SparseMatrix::getCSCNumMatrixElements()",
		"Tried to access CSC number of matrix elements, but the matrix"
		" is not on the CSC storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	if(csxNumMatrixElements == -1)
		return 0;
	else
		return (unsigned int)csxNumMatrixElements;
}

template<typename DataType>
inline const unsigned int* SparseMatrix<DataType>::getCSRRowPointers() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSR,
		"SparseMatrix::getCSRRowPointers()",
		"Tried to access CSR row pointers, but the matrix is not on"
		" the CSR storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxXPointers != nullptr,
		"SparseMatrix::getCSRRowPointers()",
		"Tried to access CSR row pointers, but row pointers have not"
		<< " been constructed yet.",
		""
	);

	return csxXPointers;
}

template<typename DataType>
inline const unsigned int* SparseMatrix<DataType>::getCSCColumnPointers() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSC,
		"SparseMatrix::getCSCColumnPointers()",
		"Tried to access CSC row pointers, but the matrix is not on"
		" the CSC storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxXPointers != nullptr,
		"SparseMatrix::getCSCColumnPointers()",
		"Tried to access CSC column pointers, but column pointers have"
		<< " not been constructed yet.",
		""
	);

	return csxXPointers;
}

template<typename DataType>
inline const unsigned int* SparseMatrix<DataType>::getCSRColumns() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSR,
		"SparseMatrix::getCSRColumns()",
		"Tried to access CSR columns, but the matrix is not on the CSR"
		<< " storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxY != nullptr,
		"SparseMatrix::getCSRColumns()",
		"Tried to access CSR columns, but columns have not been"
		<< " constructed yet.",
		""
	);

	return csxY;
}

template<typename DataType>
inline const unsigned int* SparseMatrix<DataType>::getCSCRows() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSC,
		"SparseMatrix::getCSCRows()",
		"Tried to access CSC rows, but the matrix is not on the CSC"
		<< " storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxY != nullptr,
		"SparseMatrix::getCSCRows()",
		"Tried to access CSC rows, but rows have not been constructed"
		<< " yet.",
		""
	);

	return csxY;
}

template<typename DataType>
inline const DataType* SparseMatrix<DataType>::getCSRValues() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSR,
		"SparseMatrix::getCSRValues()",
		"Tried to access CSR values, but the matrix is not on the CSR"
		<< " storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxValues != nullptr,
		"SparseMatrix::getCSRValues()",
		"Tried to access CSR values, but values have not been"
		<< " constructed yet.",
		""
	);

	return csxValues;
}

template<typename DataType>
inline const DataType* SparseMatrix<DataType>::getCSCValues() const{
	TBTKAssert(
		storageFormat == StorageFormat::CSC,
		"SparseMatrix::getCSCValues()",
		"Tried to access CSC values, but the matrix is not on the CSC"
		<< " storage format.",
		"Use SparseMatrix::setFormat() to change the storage format."
	);

	TBTKAssert(
		csxValues != nullptr,
		"SparseMatrix::getCSCValues()",
		"Tried to access CSC values, but values have not been"
		<< " constructed yet.",
		""
	);

	return csxValues;
}

template<typename DataType>
inline void SparseMatrix<DataType>::construct(){
	constructCSX();
}

template<typename DataType>
inline SparseMatrix<DataType>& SparseMatrix<DataType>::operator+=(
	const SparseMatrix &rhs
){
	TBTKAssert(
		csxNumMatrixElements != -1 && rhs.csxNumMatrixElements != -1,
		"SparseMatrix::operator+=()",
		"Unable to add matrices since the matrices have not yet been"
		<< " constructed.",
		"Ensure that SparseMatrix::construct() has been called for"
		<< " both matrices."
	);
	TBTKAssert(
		storageFormat == rhs.storageFormat,
		"SparseMatrix::operator+=()",
		"The left and right hand sides must have the same storage"
		<< " format. But the left hand side has storage format '" << (
			storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "' while the right hand side has storage format '" << (
			rhs.storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "'.",
		""
	);
	TBTKAssert(
		allowDynamicDimensions == rhs.allowDynamicDimensions,
		"SparseMatrix::operator+=()",
		"The left and right hand sides must either both have dynamic"
		<< " or both not have dynamic dimensions. But the left hand"
		<< " side " << (
			allowDynamicDimensions
			? "has dynamic dimensions "
			: "does not have dynamic dimensions "
		) << " whilte the right hand side " << (
			allowDynamicDimensions
			? "has dynamic dimensions."
			: "does not have dynamic dimensions."
		),
		"Whether the SparseMatrix has dynamic dimensions or not"
		<< " depends on whether the number of rows and columns are"
		<< " passed to the SparseMatrix constructor or not."
	);

	if(!allowDynamicDimensions){
		TBTKAssert(
			numRows == rhs.numRows && numCols == rhs.numCols,
			"SparseMatrix::operator+=()",
			"The left and right hand sides must have the same"
			<< " dimensions, but the left hand side has dimension"
			<< " '" << numRows << "x" << numCols << "' while the"
			<< " right hand side has dimensions '" << rhs.numRows
			<< "x" << numCols << "'.",
			"If both matrices have dynamic dimensions their"
			<< " dimensions do not need to agree. To create"
			<< " matrices with dynamic dimensions, do not pass row"
			<< " and column numbers to the SparseMatrix"
			<< " constructor."
		);
	}

	convertCSXToLIL();

	switch(storageFormat){
	case StorageFormat::CSR:
	{
		for(int row = 0; row < rhs.numRows; row++){
			for(
				int n = rhs.csxXPointers[row];
				n < rhs.csxXPointers[row+1];
				n++
			){
				add(row, rhs.csxY[n], rhs.csxValues[n]);
			}
		}
		break;
	}
	case StorageFormat::CSC:
		for(int col = 0; col < rhs.numCols; col++){
			for(
				int n = rhs.csxXPointers[col];
				n < rhs.csxXPointers[col+1];
				n++
			){
				add(rhs.csxY[n], col, rhs.csxValues[n]);
			}
		}
		break;
	default:
		TBTKExit(
			"SparseMatrix::operator+=()",
			"Unknown storage format.",
			"This should never happen, contact the developer."
		);
	}

	construct();

	return *this;
}

template<typename DataType>
inline SparseMatrix<DataType> SparseMatrix<DataType>::operator+(
	const SparseMatrix &rhs
) const{
	SparseMatrix sparseMatrix = *this;

	return sparseMatrix += rhs;
}

template<typename DataType>
inline SparseMatrix<DataType>& SparseMatrix<DataType>::operator-=(
	const SparseMatrix &rhs
){
	TBTKAssert(
		csxNumMatrixElements != -1 && rhs.csxNumMatrixElements != -1,
		"SparseMatrix::operator-=()",
		"Unable to subtract matrices since the matrices have not yet"
		<< " been constructed.",
		"Ensure that SparseMatrix::construct() has been called for"
		<< " both matrices."
	);
	TBTKAssert(
		storageFormat == rhs.storageFormat,
		"SparseMatrix::operator-=()",
		"The left and right hand sides must have the same storage"
		<< " format. But the left hand side has storage format '" << (
			storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "' while the right hand side has storage format '" << (
			rhs.storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "'.",
		""
	);
	TBTKAssert(
		allowDynamicDimensions == rhs.allowDynamicDimensions,
		"SparseMatrix::operator-=()",
		"The left and right hand sides must either both have dynamic"
		<< " or both not have dynamic dimensions. But the left hand"
		<< " side " << (
			allowDynamicDimensions
			? "has dynamic dimensions "
			: "does not have dynamic dimensions "
		) << " whilte the right hand side " << (
			allowDynamicDimensions
			? "has dynamic dimensions."
			: "does not have dynamic dimensions."
		),
		"Whether the SparseMatrix has dynamic dimensions or not"
		<< " depends on whether the number of rows and columns are"
		<< " passed to the SparseMatrix constructor or not."
	);

	if(!allowDynamicDimensions){
		TBTKAssert(
			numRows == rhs.numRows && numCols == rhs.numCols,
			"SparseMatrix::operator-=()",
			"The left and right hand sides must have the same"
			<< " dimensions, but the left hand side has dimension"
			<< " '" << numRows << "x" << numCols << "' while the"
			<< " right hand side has dimensions '" << rhs.numRows
			<< "x" << numCols << "'.",
			"If both matrices have dynamic dimensions their"
			<< " dimensions do not need to agree. To create"
			<< " matrices with dynamic dimensions, do not pass row"
			<< " and column numbers to the SparseMatrix"
			<< " constructor."
		);
	}

	convertCSXToLIL();

	switch(storageFormat){
	case StorageFormat::CSR:
	{
		for(int row = 0; row < rhs.numRows; row++){
			for(
				int n = rhs.csxXPointers[row];
				n < rhs.csxXPointers[row+1];
				n++
			){
				add(row, rhs.csxY[n], -rhs.csxValues[n]);
			}
		}
		break;
	}
	case StorageFormat::CSC:
		for(int col = 0; col < rhs.numCols; col++){
			for(
				int n = rhs.csxXPointers[col];
				n < rhs.csxXPointers[col+1];
				n++
			){
				add(rhs.csxY[n], col, -rhs.csxValues[n]);
			}
		}
		break;
	default:
		TBTKExit(
			"SparseMatrix::operator-=()",
			"Unknown storage format.",
			"This should never happen, contact the developer."
		);
	}

	construct();

	return *this;
}

template<typename DataType>
inline SparseMatrix<DataType> SparseMatrix<DataType>::operator-(
	const SparseMatrix &rhs
) const{
	SparseMatrix sparseMatrix = *this;

	return sparseMatrix -= rhs;
}

template<typename DataType>
inline SparseMatrix<DataType> SparseMatrix<DataType>::operator*(
	const SparseMatrix &rhs
) const{
	TBTKAssert(
		csxNumMatrixElements != -1 && rhs.csxNumMatrixElements != -1,
		"SparseMatrix::operator*=()",
		"Unable to multiply matrices since the matrices have not yet"
		<< " been constructed.",
		"Ensure that SparseMatrix::construct() has been called for"
		<< " both matrices."
	);
	TBTKAssert(
		storageFormat == rhs.storageFormat,
		"SparseMatrix::operator*=()",
		"The left and right hand sides must have the same storage"
		<< " format. But the left hand side has storage format '" << (
			storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "' while the right hand side has storage format '" << (
			rhs.storageFormat == StorageFormat::CSR
			? "StorageFormat::CSR"
			: "StorageFormat::CSC"
		) << "'.",
		""
	);
	TBTKAssert(
		allowDynamicDimensions == rhs.allowDynamicDimensions,
		"SparseMatrix::operator*=()",
		"The left and right hand sides must either both have dynamic,"
		<< " or both not have dynamic dimensions. But the left hand"
		<< " side " << (
			allowDynamicDimensions
			? "has dynamic dimensions "
			: "does not have dynamic dimensions "
		) << " whilte the right hand side " << (
			allowDynamicDimensions
			? "has dynamic dimensions."
			: "does not have dynamic dimensions."
		),
		"Whether the SparseMatrix has dynamic dimensions or not"
		<< " depends on whether the number of rows and columns are"
		<< " passed to the SparseMatrix constructor or not."
	);

	if(!allowDynamicDimensions){
		TBTKAssert(
			numCols == rhs.numRows,
			"SparseMatrix::operator*=()",
			"The number of columns for the left hand side must be"
			<< " equal to the number of rows for the right hand"
			<< " side. But the left hand side has '" << numCols
			<< "' columns while the right hand side has '"
			<< rhs.numRows << "'.",
			"If both matrices have dynamic dimensions their"
			<< " dimensions do not need to agree. To create"
			<< " matrices with dynamic dimensions, do not pass row"
			<< " and column numbers to the SparseMatrix"
			<< " constructor."
		);
	}

	SparseMatrix result;
	if(allowDynamicDimensions)
		result = SparseMatrix(storageFormat);
	else
		result = SparseMatrix(storageFormat, numRows, rhs.numCols);

	//Continue here!
	switch(storageFormat){
	case StorageFormat::CSR:
	{
		SparseMatrix rhsCSC = rhs;
		rhsCSC.setStorageFormat(StorageFormat::CSC);
		multiply(*this, rhsCSC, result);
		break;
	}
	case StorageFormat::CSC:
	{
		SparseMatrix lhsCSR = *this;
		lhsCSR.setStorageFormat(StorageFormat::CSR);
		multiply(lhsCSR, rhs, result);
		break;
	}
	default:
		TBTKExit(
			"SparseMatrix::operator+=()",
			"Unknown storage format.",
			"This should never happen, contact the developer."
		);
	}

	result.construct();

	return result;
}

template<typename DataType>
inline void SparseMatrix<DataType>::print() const{
	Streams::out << "### Dictionary of Keys (DOK) ###\n";
	if(dictionaryOfKeys.size() == 0)
		Streams::out << "-\n";
	for(unsigned int n = 0; n < dictionaryOfKeys.size(); n++){
		unsigned int row = std::get<0>(dictionaryOfKeys[n]);
		unsigned int col = std::get<1>(dictionaryOfKeys[n]);
		const DataType &value = std::get<2>(dictionaryOfKeys[n]);
		Streams::out << "(" << row << ", " << col << ", " << value << ")\n";
	}

	Streams::out << "\n";
	switch(storageFormat){
	case StorageFormat::CSR:
		Streams::out << "### Compressed sparse row (CSR) ###\n";
		if(csxNumMatrixElements == -1){
			Streams::out << "-\n";
		}
		else{
			Streams::out << "Row pointers:\n";
			for(int row = 0; row < numRows+1; row++)
				Streams::out << csxXPointers[row] << "\t";
			Streams::out << "\nColumns:\n";
			for(int n = 0; n < csxNumMatrixElements; n++)
				Streams::out << csxY[n] << "\t";
			Streams::out << "\nValues:\n";
			for(int n = 0; n < csxNumMatrixElements; n++)
				Streams::out << csxValues[n] << "\t";
			Streams::out << "\n";
		}
		break;
	case StorageFormat::CSC:
		Streams::out << "### Compressed sparse column (CSC) ###\n";
		if(csxNumMatrixElements == -1){
			Streams::out << "-\n";
		}
		else{
			Streams::out << "Column pointers:\n";
			for(int col = 0; col < numCols+1; col++)
				Streams::out << csxXPointers[col] << "\t";
			Streams::out << "\nRows:\n";
			for(int n = 0; n < csxNumMatrixElements; n++)
				Streams::out << csxY[n] << "\t";
			Streams::out << "\nValues:\n";
			for(int n = 0; n < csxNumMatrixElements; n++)
				Streams::out << csxValues[n] << "\t";
			Streams::out << "\n";
		}
		break;
	default:
		TBTKExit(
			"SparseMatrix::print()",
			"Unknow StorageFormat.",
			"This should never happen, contact the developer."
		);
	}
}

template<typename DataType>
inline std::vector<
	std::vector<std::tuple<unsigned int, DataType>>
> SparseMatrix<DataType>::constructLIL(){
	unsigned int numRows = 0;
	unsigned int numCols = 0;
	for(unsigned int n = 0; n < dictionaryOfKeys.size(); n++){
		unsigned int row = std::get<0>(dictionaryOfKeys[n]);
		unsigned int col = std::get<1>(dictionaryOfKeys[n]);
		if(row >= numRows)
			numRows = row+1;
		if(col >= numCols)
			numCols = col+1;
	}

	std::vector<
		std::vector<std::tuple<unsigned int, DataType>>
	> listOfLists;

	switch(storageFormat){
	case StorageFormat::CSR:
		listOfLists.reserve(numRows);
		for(unsigned int row = 0; row < numRows; row++){
			listOfLists.push_back(
				std::vector<std::tuple<unsigned int, DataType>>()
			);
		}

		for(unsigned int n = 0; n < dictionaryOfKeys.size(); n++){
			unsigned int row = std::get<0>(dictionaryOfKeys[n]);
			unsigned int col = std::get<1>(dictionaryOfKeys[n]);
			const DataType &value = std::get<2>(dictionaryOfKeys[n]);

			listOfLists[row].push_back(std::make_tuple(col, value));
		}
		break;
	case StorageFormat::CSC:
		listOfLists.reserve(numCols);
		for(unsigned int col = 0; col < numCols; col++){
			listOfLists.push_back(
				std::vector<std::tuple<unsigned int, DataType>>()
			);
		}

		for(unsigned int n = 0; n < dictionaryOfKeys.size(); n++){
			unsigned int row = std::get<0>(dictionaryOfKeys[n]);
			unsigned int col = std::get<1>(dictionaryOfKeys[n]);
			const DataType &value = std::get<2>(dictionaryOfKeys[n]);

			listOfLists[col].push_back(std::make_tuple(row, value));
		}
		break;
	default:
		TBTKExit(
			"SparseMatrix::constructLIL()",
			"Unknow StorageFormat.",
			"This should never happen, contact the developer."
		);
	}

	sortLIL(listOfLists);
	mergeLIL(listOfLists);

	dictionaryOfKeys.clear();

	return listOfLists;
}

template<typename DataType>
inline void SparseMatrix<DataType>::sortLIL(
	std::vector<
		std::vector<std::tuple<unsigned int, DataType>>
	> &listOfLists
) const{
	for(unsigned int x = 0; x < listOfLists.size(); x++){
		std::sort(
			listOfLists[x].begin(),
			listOfLists[x].end(),
			[](
				const std::tuple<unsigned int, DataType> &t1,
				const std::tuple<unsigned int, DataType> &t2
			){
				return std::get<0>(t1) < std::get<0>(t2);
			}
		);
	}
}

template<typename DataType>
inline void SparseMatrix<DataType>::mergeLIL(
	std::vector<
		std::vector<std::tuple<unsigned int, DataType>>
	> &listOfLists
) const{
	for(unsigned int x = 0; x < listOfLists.size(); x++){
		for(int y = listOfLists[x].size()-1; y > 0; y--){
			unsigned int y1 = std::get<0>(listOfLists[x][y]);
			unsigned int y2 = std::get<0>(listOfLists[x][y-1]);

			if(y1 == y2){
				std::get<1>(listOfLists[x][y-1])
					+= std::get<1>(listOfLists[x][y]);
				listOfLists[x].erase(
					listOfLists[x].begin() + y
				);
			}
		}
	}
}

template<typename DataType>
inline void SparseMatrix<DataType>::constructCSX(){
	convertCSXToLIL();

	if(dictionaryOfKeys.size() != 0){
		std::vector<
			std::vector<std::tuple<unsigned int, DataType>>
		> listOfLists = constructLIL();

		if(allowDynamicDimensions){
			switch(storageFormat){
			case StorageFormat::CSR:
				numRows = listOfLists.size();
				numCols = 0;
				for(
					unsigned int row = 0;
					row < listOfLists.size();
					row++
				){
					if(listOfLists[row].size() == 0)
						continue;

					unsigned int maxCol = std::get<0>(
						listOfLists[row].back()
					);
					if((int)maxCol+1 > numCols)
						numCols = maxCol + 1;
				}
				break;
			case StorageFormat::CSC:
				numCols = listOfLists.size();
				numRows = 0;
				for(
					unsigned int col = 0;
					col < listOfLists.size();
					col++
				){
					if(listOfLists[col].size() == 0)
						continue;

					unsigned int maxRow = std::get<0>(
						listOfLists[col].back()
					);
					if((int)maxRow+1 > numRows)
						numRows = maxRow + 1;
				}
				break;
			default:
				TBTKExit(
					"SparseMatrix::constructCSX()",
					"Unknow StorageFormat.",
					"This should never happen, contact the"
					<< " developer."
				);
			}
		}

		csxNumMatrixElements = 0;
		for(unsigned int x = 0; x < listOfLists.size(); x++)
			csxNumMatrixElements += listOfLists[x].size();

		switch(storageFormat){
		case StorageFormat::CSR:
			csxXPointers = new unsigned int[numRows+1];
			break;
		case StorageFormat::CSC:
			csxXPointers = new unsigned int[numCols+1];
			break;
		default:
			TBTKExit(
				"SparseMatrix::constructCSX()",
				"Unknow StorageFormat.",
				"This should never happen, contact the"
				<< " developer."
			);
		}
		csxY = new unsigned int[csxNumMatrixElements];
		csxValues = new DataType[csxNumMatrixElements];

		csxXPointers[0] = 0;
		unsigned int currentMatrixElement = 0;
		for(unsigned int x = 0; x < listOfLists.size(); x++){
			csxXPointers[x+1]
				= csxXPointers[x]
				+ listOfLists[x].size();

			for(
				unsigned int y = 0;
				y < listOfLists[x].size();
				y++
			){
				csxY[currentMatrixElement]
					= std::get<0>(listOfLists[x][y]);
				csxValues[currentMatrixElement]
					= std::get<1>(listOfLists[x][y]);

				currentMatrixElement++;
			}
		}

		switch(storageFormat){
		case StorageFormat::CSR:
			for(
				int row = listOfLists.size();
				row < numRows;
				row++
			){
				csxXPointers[row+1] = csxXPointers[row];
			}
			break;
		case StorageFormat::CSC:
			for(
				int col = listOfLists.size();
				col < numCols;
				col++
			){
				csxXPointers[col+1] = csxXPointers[col];
			}
			break;
		default:
			TBTKExit(
				"SparseMatrix::constructCSX()",
				"Unknow StorageFormat.",
				"This should never happen, contact the"
				<< " developer."
			);
		}

		TBTKAssert(
			csxNumMatrixElements == (int)currentMatrixElement,
			"SparseMatrix::constructCSX()",
			"Invalid number of matrix elements.",
			"This should never happen, contact the developer."
		);
	}
}

template<typename DataType>
inline void SparseMatrix<DataType>::convertCSXToLIL(){
	if(csxNumMatrixElements != -1){
		TBTKAssert(
			csxXPointers != nullptr
			&& csxY != nullptr
			&& csxValues != nullptr,
			"SparseMatrix::convertCSXToLIL()",
			"'csxNumMatrixElements' is not -1, but a csx-pointer"
			<< " is a nullptr.",
			"This should never happen, contact the developer."
		);

		switch(storageFormat){
		case StorageFormat::CSR:
		{
			unsigned int row = 0;
			for(int n = 0; n < csxNumMatrixElements; n++){
				if((int)csxXPointers[row+1] == n){
					for(
						int r = row+1;
						r < numRows+1;
						r++
					){
						row++;
						if((int)csxXPointers[r+1] > n)
							break;
					}
				}

				add(row, csxY[n], csxValues[n]);
			}

			break;
		}
		case StorageFormat::CSC:
		{
			unsigned int col = 0;
			for(int n = 0; n < csxNumMatrixElements; n++){
				if((int)csxXPointers[col+1] == n){
					for(
						int c = col+1;
						c < numCols+1;
						c++
					){
						col++;
						if((int)csxXPointers[c+1] > n)
							break;
					}
				}

				add(csxY[n], col, csxValues[n]);
			}

			break;
		}
		default:
			TBTKExit(
				"SparseMatrix::convertCSXToLIL()",
				"Unknow StorageFormat.",
				"This should never happen, contact the"
				<< " developer."
			);
		}

		delete [] csxXPointers;
		delete [] csxY;
		delete [] csxValues;
		csxXPointers = nullptr;
		csxY = nullptr;
		csxValues = nullptr;
		csxNumMatrixElements = -1;
	}
}

template<typename DataType>
inline void SparseMatrix<DataType>::multiply(
	const SparseMatrix &lhs,
	const SparseMatrix &rhs,
	SparseMatrix &result
){
	TBTKAssert(
		(
			lhs.storageFormat == StorageFormat::CSR
			&& rhs.storageFormat == StorageFormat::CSC
		),
		"SparseMatrix::multiply()",
		"Storage format combination not supported.",
		"This should never happen, contact the developer."
		//This algorithm assumes that the left hand side is on CSR
		//format and the right hand side is on CSC format. Appropriate
		//checks should be performed in the calling function.
	);

	for(int lhsRow = 0; lhsRow < lhs.numRows; lhsRow++){
		for(int rhsCol = 0; rhsCol < rhs.numCols; rhsCol++){
			DataType scalarProduct = 0;
			int lhsElement = lhs.csxXPointers[lhsRow];
			int rhsElement = rhs.csxXPointers[rhsCol];
			int lhsTerminatingElement = lhs.csxXPointers[lhsRow+1];
			int rhsTerminatingElement = rhs.csxXPointers[rhsCol+1];
			while(true){
				if(
					lhsElement == lhsTerminatingElement
					|| rhsElement == rhsTerminatingElement
				){
					break;
				}

				int difference = lhs.csxY[lhsElement]
					- rhs.csxY[rhsElement];

				if(difference < 0){
					lhsElement++;
				}
				else if(difference > 0){
					rhsElement++;
				}
				else{
					scalarProduct += lhs.csxValues[
						lhsElement
					]*rhs.csxValues[
						rhsElement
					];

					lhsElement++;
					rhsElement++;
				}
			}

			if(scalarProduct != 0)
				result.add(lhsRow, rhsCol, scalarProduct);
		}
	}
}

}; //End of namesapce TBTK

#endif
