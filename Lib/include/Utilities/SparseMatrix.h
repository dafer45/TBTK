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

#include <tuple>
#include <vector>

namespace TBTK{

template<typename DataType>
class SparseMatrix{
public:
	/** Constructor. */
	SparseMatrix();

	/** Constructor. */
	SparseMatrix(unsigned int numRows, unsigned int numCols);

	/** Destructor. */
	~SparseMatrix();

	/** Add matrix element. */
	void add(unsigned int row, unsigned int col, const DataType &value);

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

	/** Row pointer for compressed sparse row format (CSR). */
	unsigned int *csrRowPointers;

	/** Columns for compressed sparse row format (CSR). */
	unsigned int *csrColumns;

	/** Values for compressed sparse row format (CSR). */
	DataType *csrValues;

	/** Number of matrix elements for compressed row format (CSR). Is set
	 *  to -1 when not initialized. */
	int csrNumMatrixElements;
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

	/** Construct */

	/** Construct matrix on compressed sparse row format (CSR). */
	void constructCSR();
};

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(){
	numRows = -1;
	numCols = -1;
	allowDynamicDimensions = true;

	csrRowPointers = nullptr;
	csrColumns = nullptr;
	csrValues = nullptr;
	csrNumMatrixElements = -1;
}

template<typename DataType>
inline SparseMatrix<DataType>::SparseMatrix(
	unsigned int numRows,
	unsigned int numCols
){
	this->numRows = numRows;
	this->numCols = numCols;
	allowDynamicDimensions = false;

	csrRowPointers = nullptr;
	csrColumns = nullptr;
	csrValues = nullptr;
	csrNumMatrixElements = -1;
}

template<typename DataType>
inline SparseMatrix<DataType>::~SparseMatrix(){
}

template<typename DataType>
inline void SparseMatrix<DataType>::add(
	unsigned int row,
	unsigned int col,
	const DataType &value
){
	if(!allowDynamicDimensions){
		TBTKAssert(
			row < numRows && col < numCols,
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
	Streams::out << "### Compressed sparse row (CSR) ###\n";
	if(csrNumMatrixElements == -1){
		Streams::out << "-\n";
	}
	else{
		Streams::out << "Row pointers:\n";
		for(int row = 0; row < numRows+1; row++)
			Streams::out << csrRowPointers[row] << "\t";
		Streams::out << "\nColumns:\n";
		for(int n = 0; n < csrNumMatrixElements; n++)
			Streams::out << csrColumns[n] << "\t";
		Streams::out << "\nValues:\n";
		for(int n = 0; n < csrNumMatrixElements; n++)
			Streams::out << csrValues[n] << "\t";
		Streams::out << "\n";
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
	for(unsigned int n = 0; n < listOfLists.size(); n++){
		std::sort(
			listOfLists[n].begin(),
			listOfLists[n].end(),
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
	for(unsigned int row = 0; row < listOfLists.size(); row++){
		for(int c = listOfLists[row].size()-1; c > 0; c--){
			unsigned int col1 = std::get<0>(listOfLists[row][c]);
			unsigned int col2 = std::get<0>(listOfLists[row][c-1]);

			if(col1 == col2){
				std::get<1>(listOfLists[row][c-1])
					+= std::get<1>(listOfLists[row][c]);
				listOfLists[row].erase(
					listOfLists[row].begin() + c
				);
			}
		}
	}
}

template<typename DataType>
inline void SparseMatrix<DataType>::constructCSR(){
	if(csrNumMatrixElements != -1){
		TBTKAssert(
			csrRowPointers != nullptr
			&& csrColumns != nullptr
			&& csrValues != nullptr,
			"SparseMatrix::constructCSR()",
			"'csrNumMatrixElements' is not -1, but a csr-pointer"
			<< " is a nullptr.",
			"This should never happen, contact the developer."
		);

		unsigned int row = 0;
		for(unsigned int n = 0; n < csrNumMatrixElements; n++){
			if(csrRowPointers[row+1] == n){
				for(
					unsigned int r = row+1;
					r < numRows+1;
					r++
				){
					row++;
					if(csrRowPointers[r+1] > n)
						break;
				}
			}

			add(row, csrColumns[n], csrValues[n]);
		}

		delete [] csrRowPointers;
		delete [] csrColumns;
		delete [] csrValues;
		csrNumMatrixElements = -1;
	}

	TBTKAssert(
		csrRowPointers == nullptr
		&& csrColumns == nullptr
		&& csrValues == nullptr,
		"SparseMatrix::constructCSR()",
		"Expected 'csrRowPointers', 'csrColumns', and 'csrValues' to"
		<< " be nullptr, but not all of them are.",
		"This should never happen, contact the developer."
	);

	if(dictionaryOfKeys.size() != 0){
		std::vector<
			std::vector<std::tuple<unsigned int, DataType>>
		> listOfLists = constructLIL();

		if(allowDynamicDimensions)
			numRows = listOfLists.size();
		csrNumMatrixElements = 0;
		for(unsigned int row = 0; row < listOfLists.size(); row++)
			csrNumMatrixElements += listOfLists[row].size();

		csrRowPointers = new unsigned int[numRows+1];
		csrColumns = new unsigned int[csrNumMatrixElements];
		csrValues = new DataType[csrNumMatrixElements];

		csrRowPointers[0] = 0;
		unsigned int currentMatrixElement = 0;
		for(unsigned int row = 0; row < listOfLists.size(); row++){
			csrRowPointers[row+1]
				= csrRowPointers[row]
				+ listOfLists[row].size();

			for(
				unsigned int c = 0;
				c < listOfLists[row].size();
				c++
			){
				csrColumns[currentMatrixElement]
					= std::get<0>(listOfLists[row][c]);
				csrValues[currentMatrixElement]
					= std::get<1>(listOfLists[row][c]);

				currentMatrixElement++;
			}
		}
		for(
			unsigned int row = listOfLists.size();
			row < numRows;
			row++
		){
			csrRowPointers[row+1] = csrRowPointers[row];
		}

		TBTKAssert(
			csrNumMatrixElements == currentMatrixElement,
			"SparseMatrix::constructCSR()",
			"Invalid number of matrix elements.",
			"This should never happen, contact the developer."
		);
	}
}

}; //End of namesapce TBTK

#endif
