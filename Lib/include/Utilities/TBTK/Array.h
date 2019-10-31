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

#include "TBTK/Index.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

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
	Array(const std::vector<unsigned int> &ranges);

	//TBTKFeature Utilities.Array.construction.3 2019-10-31
	/** Constructor.
	 *
	 *  @param ranges The ranges of the Array.
	 *  @param fillValue Value to fill the Array with. */
	Array(const std::vector<unsigned int> &ranges, const DataType &fillValue);

	//TBTKFeature Utilities.Array.copy.1 2019-10-31
	/** Copy constructor.
	 *
	 *  @param array Array to copy. */
	Array(const Array &array);

	//TBTKFeature Utilities.Array.move.1 2019-10-31
	/** Move constructor.
	 *
	 *  @param array Array to move from. */
	Array(Array &&array);

	/** Destructor. */
	~Array();

	//TBTKFeature Utilities.Array.copyAssignment.1 2019-10-31
	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side of the expression after assignment has
	 *  occured. */
	Array& operator=(const Array &rhs);

	//TBTKFeature Utilities.Array.moveAssignment.1 2019-10-31
	/** Move assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side of the expression after assignment has
	 *  occured. */
	Array& operator=(Array &&rhs);

	//TBTKFeature Utilities.Array.operatorArraySubscript.1 2019-10-31
	/** Array subscript operator.
	 *
	 *  @param index Index to get the value for.
	 *
	 *  @return The value for the given index. */
	DataType& operator[](const std::initializer_list<unsigned int> &index);

	//TBTKFeature Utilities.Array.operatorArraySubscript.2 2019-10-31
	/** Array subscript operator.
	 *
	 *  @param index Index to get the value for.
	 *
	 *  @return The value for the given index. */
	const DataType& operator[](
		const std::initializer_list<unsigned int> &index
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
		Array<DataType> result(rhs.ranges);
		for(unsigned int n = 0; n < rhs.size; n++)
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

	/** Get ranges.
	 *
	 *  @return The @lin Array Arrays@endlink ranges. */
	const std::vector<unsigned int>& getRanges() const;

	/** ostream operator. */
	template<typename DT>
	friend std::ostream& operator<<(
		std::ostream &stream,
		const Array<DT> &array
	);

	/** Get data. */
	DataType* getData();

	/** Get data. */
	const DataType* getData() const;

	/** Get the number of elements in the Array. */
	unsigned int getSize() const;
private:
	/** Data data. */
	DataType *data;

	/** Data size. */
	unsigned int size;

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
	size = 0;
	data = nullptr;
}

template<typename DataType>
Array<DataType>::Array(const std::vector<unsigned int> &ranges){
	this->ranges = ranges;
	size = 1;
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= ranges[n];
	}

	data = new DataType[size];
}

template<typename DataType>
Array<DataType>::Array(
	const std::vector<unsigned int> &ranges,
	const DataType &fillValue
){
	this->ranges = ranges;
	size = 1;
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= ranges[n];
	}

	data = new DataType[size];

	for(unsigned int n = 0; n < size; n++)
		data[n] = fillValue;
}

template<typename DataType>
Array<DataType>::Array(const Array &array){
	ranges = array.ranges;
	size = array.size;
	if(size != 0){
		data = new DataType[size];
		for(unsigned int n = 0; n < size; n++)
			data[n] = array.data[n];
	}
	else{
		data = nullptr;
	}
}

template<typename DataType>
Array<DataType>::Array(Array &&array){
	ranges = std::move(array.ranges);
	size = std::move(array.size);
	data = array.data;
	array.data = nullptr;
}

template<typename DataType>
Array<DataType>::~Array(){
	if(data != nullptr)
		delete [] data;
}

template<typename DataType>
Array<DataType>& Array<DataType>::operator=(const Array &rhs){
	if(this != &rhs){
		ranges = rhs.ranges;
		size = rhs.size;
		if(data != nullptr)
			delete [] data;
		if(size != 0){
			data = new DataType[size];
			for(unsigned int n = 0; n < size; n++)
				data[n] = rhs.data[n];
		}
		else{
			data = nullptr;
		}
	}

	return *this;
}

template<typename DataType>
Array<DataType>& Array<DataType>::operator=(Array &&rhs){
	if(this != &rhs){
		ranges = std::move(rhs.ranges);
		size = std::move(rhs.size);
		if(data != nullptr)
			delete [] data;
		data = rhs.data;
		rhs.data = nullptr;
	}

	return *this;
}

template<typename DataType>
inline DataType& Array<DataType>::operator[](
	const std::initializer_list<unsigned int> &index
){
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += *(index.begin() + n);
	}

	return data[idx];
}

template<typename DataType>
inline const DataType& Array<DataType>::operator[](
	const std::initializer_list<unsigned int> &index
) const{
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += *(index.begin() + n);
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

	Array<DataType> result(ranges);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n] + rhs.data[n];

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator-(
	const Array<DataType> &rhs
) const{
	assertCompatibleRanges(rhs, "operator+()");

	Array<DataType> result(ranges);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n] - rhs.data[n];

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator*(
	const DataType &rhs
) const{
	Array<DataType> result(ranges);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n]*rhs;

	return result;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator/(
	const DataType &rhs
) const{
	Array<DataType> result(ranges);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n]/rhs;

	return result;
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

	Array array(newRanges);

	fillSlice(array, index, 0, 0, 0);

	return array;
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
inline DataType* Array<DataType>::getData(){
	return data;
}

template<typename DataType>
inline const DataType* Array<DataType>::getData() const{
	return data;
}

template<typename DataType>
inline unsigned int Array<DataType>::getSize() const{
	return size;
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
