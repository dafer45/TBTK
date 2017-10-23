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

#include "Index.h"
#include "TBTKMacros.h"

#include <vector>

namespace TBTK{

template<typename DataType>
class Array{
public:
	/** Constructor. */
	Array(const std::vector<unsigned int> &ranges);

	/** Constructor. */
	Array(const std::vector<unsigned int> &ranges, const DataType &fillValue);

	/** Copy constructor. */
	Array(const Array &array);

	/** Move constructor. */
	Array(Array &&array);

	/** Destructor. */
	~Array();

	/** Assignment operator. */
	Array& operator=(const Array &rhs);

	/** Move assignment operator. */
	Array& operator=(Array &&rhs);

	/** Array subscript operator. */
	DataType& operator[](const std::initializer_list<unsigned int> &index);

	/** Array subscript operator. */
	const DataType& operator[](
		const std::initializer_list<unsigned int> &index
	) const;

	/** Get slice. */
	Array<DataType> getSlice(const std::vector<int> &index) const;

	/** Get ranges. */
	const std::vector<unsigned int>& getRanges() const;
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
		const std::vector<int> &index,
		unsigned int subindex,
		unsigned int offsetSlice,
		unsigned int offsetOriginal
	) const;
};

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
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = array.data[n];
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
		data = new DataType[size];
		for(unsigned int n = 0; n < size; n++)
			data[n] = rhs.data[n];
	}

	return *this;
}

template<typename DataType>
Array<DataType>& Array<DataType>::operator=(Array &&rhs){
	if(this != &rhs){
		ranges = std::move(rhs.ranges);
		size = std::move(rhs.size);
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
Array<DataType> Array<DataType>::getSlice(const std::vector<int> &index) const{
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
				index[n] == IDX_ALL,
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
	const std::vector<int> &index,
	unsigned int subindex,
	unsigned int offsetSlice,
	unsigned int offsetOriginal
) const{
	if(subindex == index.size()-1){
		if(index[subindex] == IDX_ALL){
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
		if(index[subindex] == IDX_ALL){
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

}; //End of namesapce TBTK

#endif
