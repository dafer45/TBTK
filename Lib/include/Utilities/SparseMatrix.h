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
	/** Enum class for determining the storage format. */
	enum class StorageFormat {CSR, CSC};

	/** Constructor. */
	SparseMatrix(StorageFormat);

	/** Constructor. */
	SparseMatrix(
		StorageFormat storageFormat,
		unsigned int numRows,
		unsigned int numCols
	);

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

	/** Construct */

	/** Construct matrix on compressed sparse row format (CSR). */
	void constructCSX();
};

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
	if(csxNumMatrixElements != -1){
		TBTKAssert(
			csxXPointers != nullptr
			&& csxY != nullptr
			&& csxValues != nullptr,
			"SparseMatrix::constructCSX()",
			"'csxNumMatrixElements' is not -1, but a csx-pointer"
			<< " is a nullptr.",
			"This should never happen, contact the developer."
		);

		switch(storageFormat){
		case StorageFormat::CSR:
		{
			unsigned int row = 0;
			for(unsigned int n = 0; n < csxNumMatrixElements; n++){
				if(csxXPointers[row+1] == n){
					for(
						unsigned int r = row+1;
						r < numRows+1;
						r++
					){
						row++;
						if(csxXPointers[r+1] > n)
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
			for(unsigned int n = 0; n < csxNumMatrixElements; n++){
				if(csxXPointers[col+1] == n){
					for(
						unsigned int c = col+1;
						c < numCols+1;
						c++
					){
						col++;
						if(csxXPointers[c+1] > n)
							break;
					}
				}

				add(csxY[n], col, csxValues[n]);
			}

			break;
		}
		default:
			TBTKExit(
				"SparseMatrix::constructCSX()",
				"Unknow StorageFormat.",
				"This should never happen, contact the"
				<< " developer."
			);
		}

		delete [] csxXPointers;
		delete [] csxY;
		delete [] csxValues;
		csxNumMatrixElements = -1;
	}

	TBTKAssert(
		csxXPointers == nullptr
		&& csxY == nullptr
		&& csxValues == nullptr,
		"SparseMatrix::constructCSX()",
		"Expected 'csxXPointers', 'csxY', and 'csrValues' to"
		<< " be nullptr, but not all of them are.",
		"This should never happen, contact the developer."
	);

	if(dictionaryOfKeys.size() != 0){
		std::vector<
			std::vector<std::tuple<unsigned int, DataType>>
		> listOfLists = constructLIL();

		if(allowDynamicDimensions){
			switch(storageFormat){
			case StorageFormat::CSR:
				numRows = listOfLists.size();
				break;
			case StorageFormat::CSC:
				numCols = listOfLists.size();
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
				unsigned int row = listOfLists.size();
				row < numRows;
				row++
			){
				csxXPointers[row+1] = csxXPointers[row];
			}
			break;
		case StorageFormat::CSC:
			for(
				unsigned int col = listOfLists.size();
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
			csxNumMatrixElements == currentMatrixElement,
			"SparseMatrix::constructCSX()",
			"Invalid number of matrix elements.",
			"This should never happen, contact the developer."
		);
	}
}

}; //End of namesapce TBTK

#endif
