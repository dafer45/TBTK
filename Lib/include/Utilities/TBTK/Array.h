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
 *  @file Array.h
 *  @brief Multi-dimensional array.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY
#define COM_DAFER45_TBTK_ARRAY

#include "TBTK/CArray.h"
#include "TBTK/Index.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

/** @brief Multi-dimensional array.
 *
 *  The Array provides a convenient interface for handling multi-dimensional
 *  array.
 *
 *  # Indexing
 *  An Array is created using
 *  ```cpp
 *    Array<DataType> array({SIZE_X, SIZE_Y, SIZE_Z});
 *  ```
 *  The curly braces determines the Array ranges. Any number of dimensions
 *  is possible. Similarly, elements can be accessed using
 *  ```cpp
 *    array[{x, y, z}] = 10;
 *    DataType value = array[{x, y, z}];
 *  ```
 *
 *  # Arithmetics
 *  It is possible to add and subtract Arrays with the same ranges.
 *  ```cpp
 *    Array<DataType> sum = array0 + array1;
 *    Array<DataType> difference = array0 - array1;
 *  ```
 *  It is also possible to multiply and divide an Array by a value with the
 *  same DataType as the Array elements.
 *  ```cpp
 *    DataType value = 10;
 *    Array<DataType> product = value*array;
 *    Array<DataType> quotient = array/value;
 *  ```
 *
 *  # Slicing
 *  Consider the code
 *  ```cpp
 *    Array<DataType> array({SIZE_X, SIZE_Y, SIZE_Z});
 *    //Fill array with some values here.
 *    //...
 *    Array<DataType> slice = array.getSlice({_a_, 2, _a_});
 *  ```
 *  Here *slice* will be an Array with ranges {SIZE_X, SIZE_Z} and satsify
 *  *slice[{x, z}] = array[{x, 2, z}].
 *
 *  *Note: If you write library code for TBTK, use IDX_ALL instead of \_a\_*.
 *
 *  # Example
 *  \snippet Utilities/Array.cpp Array
 *  ## Output
 *  \snippet output/Utilities/Array.output Array */
template<typename DataType>
class Array{
public:
	//TBTKFeature Utilities.Array.construction.1 2019-10-31
	/** Constructor. */
	Array();

	//TBTKFeature Utilities.Array.construction.2 2019-10-31
	/** Constructor.
	 *
	 *  @param ranges The ranges of the Array. */
	explicit Array(const std::initializer_list<unsigned int> &ranges);

	//TBTKFeature Utilities.Array.construction.3 2019-10-31
	/** Constructor.
	 *
	 *  @param ranges The ranges of the Array.
	 *  @param fillValue Value to fill the Array with. */
	Array(
		const std::initializer_list<unsigned int> &ranges,
		const DataType &fillValue
	);

	/** Constructs an Array from an std::vector.
	 *
	 *  @param vector The std::vector to copy from. */
	Array(const std::vector<DataType> &vector);

	/** Constructs an Array from an std::vector.
	 *
	 *  @param vector The std::vector to copy from. */
	Array(const std::vector<std::vector<DataType>> &vector);

	/** Create an array with a vector of ranges. Identical to calling the
	 *  constructor using an std::initializer_list, but using a std::vector
	 *  instead. Allows for the creation of an Arrays with dynamically
	 *  assigned ranges.
	 *
	 *  @param ranges The ranges of the Array. */
	static Array create(const std::vector<unsigned int> &ranges);

	/** Create an array with a vector of ranges. Identical to calling the
	 *  constructor using an std::initializer_list, but using a std::vector
	 *  instead. Allows for the creation of an Arrays with dynamically
	 *  assigned ranges.
	 *
	 *  @param ranges The ranges of the Array.
	 *  @param fillValue Value to fill the Array with. */
	static Array create(
		const std::vector<unsigned int> &ranges,
		const DataType &fillValue
	);

	//TBTKFeature Utilities.Array.operatorArraySubscript.1 2019-10-31
	/** Array subscript operator.
	 *
	 *  @param index Index to get the value for.
	 *
	 *  @return The value for the given index. */
	DataType& operator[](const std::vector<unsigned int> &index);

	//TBTKFeature Utilities.Array.operatorArraySubscript.2 2019-10-31
	/** Array subscript operator.
	 *
	 *  @param index Index to get the value for.
	 *
	 *  @return The value for the given index. */
	const DataType& operator[](
		const std::vector<unsigned int> &index
	) const;

	//TBTKFeature Utilities.Array.operatorArraySubscript.3 2019-10-31
	/** Array subscript operator. Imediate access for optimization. For an
	 *  array with size {SIZE_A, SIZE_B, SIZE_C}, the supplied index should
	 *  be calculated as n = SIZE_C*SIZE_B*a + SIZE_C*b + c.
	 *
	 *  @param n Entry in the array.
	 *
	 *  @return The value of entry n. */
	DataType& operator[](unsigned int n);

	//TBTKFeature Utilities.Array.opreatorArraySubscript.4 2019-10-31
	/** Array subscript operator. Imediate access for optimization. For an
	 *  array with size {SIZE_A, SIZE_B, SIZE_C}, the supplied index should
	 *  be calculated as n = SIZE_C*SIZE_B*a + SIZE_C*b + c.
	 *
	 *  @param n Entry in the array.
	 *
	 *  @return The value of entry n. */
	const DataType& operator[](unsigned int n) const;

	//TBTKFeature Utilities.Array.operatorAddition.1 2019-10-31
	//TBTKFeature Utilities.Array.operatorAddition.2 2019-10-31
	//TBTKFeature Utilities.Array.operatorAddition.3 2019-10-31
	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The sum of the left and right hand side. */
	Array operator+(const Array &rhs) const;

	//TBTKFeature Utilities.Array.operatorSubtraction.1 2019-10-31
	//TBTKFeature Utilities.Array.operatorSubtraction.2 2019-10-31
	//TBTKFeature Utilities.Array.operatorSubtraction.3 2019-10-31
	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The sum of the left and right hand side. */
	Array operator-(const Array &rhs) const;

	//TBTKFeature Utilities.Array.operatorMultiplication.1 2019-10-31
	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The product of the left and right hand side. */
	Array operator*(const DataType &rhs) const;

	//TBTKFeature Utilities.Array.operatorMultiplication.2 2019-10-31
	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side of the expression.
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The product of the left and right hand side. */
	friend Array operator*(const DataType &lhs, const Array &rhs){
		Array<DataType> result = Array<DataType>::create(rhs.ranges);
		for(unsigned int n = 0; n < rhs.getSize(); n++)
			result.data[n] = lhs*rhs.data[n];

		return result;
	}

	//TBTKFeature Utilities.Array.operatorDivision.1 2019-10-31
	/** Division operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The quotient between the left and right hand side. */
	Array operator/(const DataType &rhs) const;

	/** Comparison operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return True if the size and individual entries of the left and
	 *  right hand sides are equal, otherwise false. */
	bool operator==(const Array &rhs) const;

	//TBTKFeature Utilities.Array.getSlice.1
	/** Get a subset of the Array that results from setting one or multiple
	 *  indices equal to given values.
	 *
	 *  # Example
	 *  ```cpp
	 *    Array array({5, 5, 5});
	 *    Array slice = array.slice({_a_, 2, _a_});
	 *  ```
	 *  The result is a two-dimensional Array for which slice({x, z}) =
	 *  array({x, 2, z}).
	 *
	 *  @param index Index into the Array.
	 *
	 *  @return An Array of lower dimension. */
	Array<DataType> getSlice(const std::vector<Subindex> &index) const;

	/** Get a new Array with permuted indices.
	 *
	 *  @param permutation A list of integers from 0 to N-1, where N is the
	 *  number of indices.
	 *
	 *  @return A new Array where the nth Subindex corresponds to the
	 *  original Subindex in position permutation[n]. */
	Array<DataType> getArrayWithPermutedIndices(
		const std::vector<unsigned int> &permutation
	) const;

	/** Get a new Array with the indices in reverse order.
	 *
	 *  @return A new Array where the indices occurs in reverse order. */
	Array<DataType> getArrayWithReversedIndices() const;

	/** Get ranges.
	 *
	 *  @return The @link Array Arrays@endlink ranges. */
	const std::vector<unsigned int>& getRanges() const;

	/** ostream operator. */
	template<typename DT>
	friend std::ostream& operator<<(
		std::ostream &stream,
		const Array<DT> &array
	);

	//TBTKFeature Utilities.Array.getData.1 2019-10-31
	/** Get raw data. If the array has ranges {SIZE_X, SIZE_Y, SIZE_Z},
	 *  rawData[SIZE_Z*(SIZE_Y*x + y) + z = array[{x, y, z}].
	 *
	 *  @return The raw data as a linear CArray. */
	CArray<DataType>& getData();

	//TBTKFeature Utilities.Array.getData.2.C++ 2019-10-31
	/** Get raw data. If the array has ranges {SIZE_X, SIZE_Y, SIZE_Z},
	 *  rawData[SIZE_Z*(SIZE_Y*x + y) + z = array[{x, y, z}].
	 *
	 *  @return The raw data as a linear CArray. */
	const CArray<DataType>& getData() const;

	//TBTKFeature Utilities.Array.getSize.1 2019-10-31
	/** Get the number of elements in the Array. */
	unsigned int getSize() const;
private:
	/** Data data. */
	CArray<DataType> data;

	/** Ranges. */
	std::vector<unsigned int> ranges;

	/** Fill slice. */
	void fillSlice(
		Array &array,
		const std::vector<Subindex> &index,
		unsigned int subindex,
		unsigned int offsetSlice,
		unsigned int offsetOriginal
	) const;

	/** Checks wether the Array has the same ranges as a given Array. */
	void assertCompatibleRanges(
		const Array &array,
		std::string functionName
	) const;
};

template<typename DataType>
Array<DataType>::Array(){
}

template<typename DataType>
Array<DataType>::Array(const std::initializer_list<unsigned int> &ranges){
	this->ranges = ranges;
	unsigned int size = 1;
	for(unsigned int n = 0; n < this->ranges.size(); n++){
		TBTKAssert(
			this->ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= this->ranges[n];
	}

	data = CArray<DataType>(size);
}

template<typename DataType>
Array<DataType>::Array(
	const std::initializer_list<unsigned int> &ranges,
	const DataType &fillValue
){
	this->ranges = ranges;
	unsigned int size = 1;
	for(unsigned int n = 0; n < this->ranges.size(); n++){
		TBTKAssert(
			this->ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= this->ranges[n];
	}

	data = CArray<DataType>(size);

	for(unsigned int n = 0; n < size; n++)
		data[n] = fillValue;
}

template<typename DataType>
Array<DataType>::Array(const std::vector<DataType> &vector){
	TBTKAssert(
		vector.size() != 0,
		"Array::Array()",
		"Invalid input.",
		"Unable to create an Array from an empty vector."
	);

	ranges.push_back(vector.size());
	data = CArray<DataType>(ranges[0]);
	for(unsigned int n = 0; n < ranges[0]; n++)
		data[n] = vector[n];
}

template<typename DataType>
Array<DataType>::Array(const std::vector<std::vector<DataType>> &vector){
	TBTKAssert(
		vector.size() != 0,
		"Array::Array()",
		"Invalid input. Unable to create an Array from an empty"
		<< " vector.",
		""
	);
	TBTKAssert(
		vector[0].size() != 0,
		"Array::Array()",
		"Invalid input. Unable to create an Array from a vector with"
		<< " an empty row.",
		""
	);
	ranges.push_back(vector.size());
	ranges.push_back(vector[0].size());

	for(unsigned int n = 1; n < vector.size(); n++){
		TBTKAssert(
			vector[n].size() == vector[0].size(),
			"Array::Array()",
			"Invalid input. vector[" << n << "] has a different"
			<< " size than vector[0]",
			""
		);
	}

	data = CArray<DataType>(ranges[0]*ranges[1]);
	for(unsigned int x = 0; x < ranges[0]; x++)
		for(unsigned int y = 0; y < ranges[1]; y++)
			data[ranges[1]*x + y] = vector[x][y];
}

template<typename DataType>
Array<DataType> Array<DataType>::create(
	const std::vector<unsigned int> &ranges
){
	Array<DataType> array;
	array.ranges = ranges;
	unsigned int size = 1;
	for(unsigned int n = 0; n < array.ranges.size(); n++){
		TBTKAssert(
			array.ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= array.ranges[n];
	}

	array.data = CArray<DataType>(size);

	return array;
}

template<typename DataType>
Array<DataType> Array<DataType>::create(
	const std::vector<unsigned int> &ranges,
	const DataType &fillValue
){
	Array<DataType> array;
	array.ranges = ranges;
	unsigned int size = 1;
	for(unsigned int n = 0; n < array.ranges.size(); n++){
		TBTKAssert(
			array.ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= array.ranges[n];
	}

	array.data = CArray<DataType>(size);

	for(unsigned int n = 0; n < size; n++)
		array.data[n] = fillValue;

	return array;
}

template<typename DataType>
inline DataType& Array<DataType>::operator[](
	const std::vector<unsigned int> &index
){
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += index[n];
	}

	return data[idx];
}

template<typename DataType>
inline const DataType& Array<DataType>::operator[](
	const std::vector<unsigned int> &index
) const{
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += index[n];
	}

	return data[idx];
}

template<typename DataType>
inline DataType& Array<DataType>::operator[](unsigned int n){
	return data[n];
}

template<typename DataType>
inline const DataType& Array<DataType>::operator[](unsigned int n) const{
	return data[n];
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator+(
	const Array<DataType> &rhs
) const{
	assertCompatibleRanges(rhs, "operator+()");

	Array<DataType> result = Array<DataType>::create(ranges);
	for(unsigned int n = 0; n < data.getSize(); n++)
		result.data[n] = data[n] + rhs.data[n];

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator-(
	const Array<DataType> &rhs
) const{
	assertCompatibleRanges(rhs, "operator+()");

	Array<DataType> result = Array<DataType>::create(ranges);
	for(unsigned int n = 0; n < data.getSize(); n++)
		result.data[n] = data[n] - rhs.data[n];

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator*(
	const DataType &rhs
) const{
	Array<DataType> result = Array<DataType>::create(ranges);
	for(unsigned int n = 0; n < data.getSize(); n++)
		result.data[n] = data[n]*rhs;

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator/(const DataType &rhs) const{
	Array<DataType> result = Array<DataType>::create(ranges);
	for(unsigned int n = 0; n < data.getSize(); n++)
		result.data[n] = data[n]/rhs;

	return result;
}

template<typename DataType>
inline bool Array<DataType>::operator==(const Array<DataType> &rhs) const{
	if(ranges.size() != rhs.ranges.size())
		return false;
	for(unsigned int n = 0; n < ranges.size(); n++)
		if(ranges[n] != rhs.ranges[n])
			return false;

	for(unsigned int n = 0; n < getSize(); n++)
		if(data[n] != rhs.data[n])
			return false;

	return true;
}

template<typename DataType>
Array<DataType> Array<DataType>::getSlice(const std::vector<Subindex> &index) const{
	TBTKAssert(
		ranges.size() == index.size(),
		"Array::getSlice()",
		"Incompatible ranges.",
		"'index' must have the same number of dimensions as 'ranges'."
	);

	std::vector<unsigned int> newRanges;
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			index[n] < (int)ranges[n],
			"Array::getSlice()",
			"'index' out of range.",
			""
		);
		if(index[n] < 0){
			TBTKAssert(
				index[n].isWildcard(),
				"Array::getSlice()",
				"Invalid symbol.",
				"'index' can only contain positive numbers or"
				<< " 'IDX_ALL'."
			);
			newRanges.push_back(ranges[n]);
		}
	}

	Array array = Array::create(newRanges);

	fillSlice(array, index, 0, 0, 0);

	return array;
}

template<typename DataType>
Array<DataType> Array<DataType>::getArrayWithPermutedIndices(
	const std::vector<unsigned int> &permutation
) const{
	TBTKAssert(
		permutation.size() == ranges.size(),
		"Array::getArrayWithPermutedIndices()",
		"The number of permutation indices '" << permutation.size()
		<< "' must be the same a the number of ranges '"
		<< ranges.size() << "'.",
		""
	);

	std::vector<bool> indexIncluded(permutation.size(), false);
	for(unsigned int n = 0; n < permutation.size(); n++){
		TBTKAssert(
			permutation[n] >= 0
			&& permutation[n] < permutation.size(),
			"Array::getArrayWithPermutedIndices()",
			"Invalid permutation values 'permutation[" << n << "]"
			<< " = " << permutation[n] << "'. Must be a number"
			<< " between 0 and N-1, where N is the number of"
			<< " ranges.",
			""
		);
		indexIncluded[permutation[n]] = true;
	}
	for(unsigned int n = 0; n < indexIncluded.size(); n++){
		TBTKAssert(
			indexIncluded[n],
			"Array::getArrayWithPermutedIndices()",
			"Invalid permutation. Missing permutation index '" << n
			<< "'.",
			""
		);
	}

	std::vector<unsigned int> newRanges;
	for(unsigned int n = 0; n < ranges.size(); n++)
		newRanges.push_back(ranges[permutation[n]]);
	Array<DataType> newArray = Array<DataType>::create(newRanges);

	std::vector<unsigned int> begin = newRanges;
	std::vector<unsigned int> end = newRanges;
	std::vector<unsigned int> increment = newRanges;
	for(unsigned int n = 0; n < newRanges.size(); n++){
		begin[n] = 0;
		increment[n] = 1;
	}
	MultiCounter<unsigned int> counter(begin, end, increment);
	for(counter.reset(); !counter.done(); ++counter){
		std::vector<unsigned int> newArrayIndex = counter;
		std::vector<unsigned int> arrayIndex(newArrayIndex.size());
		for(unsigned int c = 0; c < newArrayIndex.size(); c++)
			arrayIndex[permutation[c]] = newArrayIndex[c];
		newArray[newArrayIndex] = (*this)[arrayIndex];
	}

	return newArray;
}

template<typename DataType>
Array<DataType> Array<DataType>::getArrayWithReversedIndices() const{
	std::vector<unsigned int> permutation;
	for(unsigned int n = 0; n < ranges.size(); n++)
		permutation.push_back(ranges.size() - n - 1);

	return getArrayWithPermutedIndices(permutation);
}

template<typename DataType>
void Array<DataType>::fillSlice(
	Array &array,
	const std::vector<Subindex> &index,
	unsigned int subindex,
	unsigned int offsetSlice,
	unsigned int offsetOriginal
) const{
	if(subindex == index.size()-1){
		if(index[subindex].isWildcard()){
			for(unsigned int n = 0; n < ranges[subindex]; n++){
				array.data[offsetSlice*ranges[subindex] + n]
					= data[
						offsetOriginal*ranges[subindex]
						+ n
					];
			}
		}
		else{
			array.data[offsetSlice] = data[
				offsetOriginal*ranges[subindex]
				+ index[subindex]
			];
		}
	}
	else{
		if(index[subindex].isWildcard()){
			for(unsigned int n = 0; n < ranges[subindex]; n++){
				fillSlice(
					array,
					index,
					subindex+1,
					offsetSlice*ranges[subindex] + n,
					offsetOriginal*ranges[subindex] + n
				);
			}
		}
		else{
			fillSlice(
				array,
				index,
				subindex+1,
				offsetSlice,
				offsetOriginal*ranges[subindex]
					+ index[subindex]
			);
		}
	}
}

template<typename DataType>
inline const std::vector<unsigned int>& Array<DataType>::getRanges() const{
	return ranges;
}

template<typename DataType>
inline std::ostream& operator<<(
	std::ostream &stream,
	const Array<DataType> &array
){
	switch(array.ranges.size()){
	case 1:
		stream << "[";
		for(unsigned int n = 0; n < array.ranges[0]; n++){
			if(n != 0)
				stream << ", ";
			stream << array[{n}];
		}
		stream << "]";

		break;
	case 2:
		stream << "[";
		for(unsigned int row = 0; row < array.ranges[0]; row++){
			if(row != 0)
				stream << "\n";
			stream << "[";
			for(
				unsigned int column = 0;
				column < array.ranges[1];
				column++
			){
				if(column != 0)
					stream << ", ";
				stream << array[{row, column}];
			}
			stream << "]";
		}
		stream << "]";

		break;
	default:
		TBTKExit(
			"Array::operator<<()",
			"Unable to print Array of rank '"
			<< array.ranges.size() << "'.",
			""
		);
	}

	return stream;
}

template<typename DataType>
inline CArray<DataType>& Array<DataType>::getData(){
	return data;
}

template<typename DataType>
inline const CArray<DataType>& Array<DataType>::getData() const{
	return data;
}

template<typename DataType>
inline unsigned int Array<DataType>::getSize() const{
	return data.getSize();
}

template<typename DataType>
inline void Array<DataType>::assertCompatibleRanges(
	const Array<DataType> &array,
	std::string functionName
) const{
	TBTKAssert(
		ranges.size() == array.ranges.size(),
		"Array::" + functionName,
		"Incompatible ranges.",
		"Left and right hand sides must have the same number of"
		<< " dimensions."
	);
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			ranges[n] == array.ranges[n],
			"Array::" + functionName,
			"Incompatible ranges.",
			"Left and right hand sides must have the same ranges."
		);
	}
}

}; //End of namesapce TBTK

#endif
