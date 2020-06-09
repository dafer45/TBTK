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
#include "TBTK/Range.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

//Forward declation
namespace Math{
	template<typename DataType>
	class ArrayAlgorithms;
}; //End of namespace Math

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
 *  or alternatively
 *  ```cpp
 *    Array<DataType> array({SIZE_X, SIZE_Y, SIZE_Z}, value);
 *  ```
 *  to initialize each element to 'value'.
 *
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
 *  \snippet output/Utilities/Array.txt Array */
template<typename DataType>
class Array : public Serializable{
public:
	class Modifier{
	public:
		/** Assignment operator. Assigns the right hand side to the
		 *  section of the corresponding Array that the Modifier is
		 *  Modifying.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return The Modifier itself. */
		Modifier& operator=(const DataType &rhs);

		/** Addition assignment operator. Adds the the right hand side
		 *  to the section of the corresponding Array that the Modifier
		 *  is Modifying.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return The Modifier itself. */
		Modifier& operator+=(const DataType &rhs);

		/** Subtraction assignment operator. Subtracts the right hand
		 *  side from the section of the corresponding Array that the
		 *  Modifier is Modifying.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return The Modifier itself. */
		Modifier& operator-=(const DataType &rhs);

		/** Multiplication assignment operator. Multiplies the right
		 *  hand side into the section of the corresponding Array that
		 *  the Modifier is Modifying.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return The Modifier itself. */
		Modifier& operator*=(const DataType &rhs);

		/** Division assignment operator. Divides the section of the
		 *  correpsonding Array that the Modifier is modifying by the
		 *  right hand side.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return The Modifier itself. */
		Modifier& operator/=(const DataType &rhs);
	private:
		/** The array to modify. */
		Array &array;

		/** The pattern to modify according to. */
		std::vector<Subindex> pattern;

		/** Constructor. */
		Modifier(Array &array, const std::vector<Subindex> &pattern);

		std::vector<
			std::vector<unsigned int>
		> getAllCompatibleIndices() const;

		/** Friend class. */
		friend class Array;
	};

	class Ranges : public std::vector<unsigned int>{
	public:
		/** Constructor. */
		Ranges(){};

		/** Construct a Ranges from an std::vector<unsigned int>.
		 *
		 *  @param ranges The vector to construct the ranges from. */
		Ranges(
			const std::vector<unsigned int> &ranges
		) :
			std::vector<unsigned int>(ranges)
		{
		}

		/** Construct a Ranges from an
		 *  std::initializer_list<unsigned int>.
		 *
		 *  @param ranges The initializer_list to construct the ranges
		 *  from. */
		Ranges(
			const std::initializer_list<unsigned int> &ranges
		) :
			std::vector<unsigned int>(ranges)
		{
		}

		/** Compare two Ranges.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return True if the two Ranges have the same number of
		 *  elements and each element is equal, otherwise false. */
		bool operator==(const Ranges &rhs) const;

		/** Compare two Ranges for inequality.
		 *
		 *  @param rhs The right hand side of the expression.
		 *
		 *  @return False if the two Ranges have the same number of
		 *  elements and each element is equal, otherwise true. */
		bool operator!=(const Ranges &rhs) const;
	private:
	};

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

	/** Construct an Array from a Range. The created Array has the same
	 *  number of elements as the resolution of the Range. The elements are
	 *  initialized so that the first and last elements corresponds to the
	 *  upper and lower bound of the range, and the intermediate values are
	 *  equispaced between these bounds.
	 *
	 *  @param range A Range object that specifies the number of elements
	 *  and lower and upper bound. */
	Array(const Range &range);

	/** Constructs an Array from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Array.
	 *
	 *  @param mode The mode with which the string has been serialized. */
	Array(const std::string &serialization, Mode mode);

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

	/** Type cast operator. Creates a copy of the Array with the data type
	 *  changed to the cast type.
	 *
	 *  @return A new Array with the data type of the elements changed to
	 *  CastType. */
	template<typename CastType>
	operator Array<CastType>() const;

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

	/** Function operator. Returns an Array::Modifier object that can be
	 *  used to modify multiple Array elements at once. The pattern can
	 *  contain wildcards that determine which elements to modify.
	 *
	 *  @param pattern A pattern specifying the indices to modify.
	 *
	 *  @return An Array::Modifier object that can be used to modify
	 *  multiple elements in the Array at once. */
	Modifier operator()(const std::vector<Subindex> &pattern);

	/** Addition equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after the right hand side has been added. */
	Array& operator+=(const Array &rhs);

	/** Addition equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after the right hand side has been added to each
	 *  element. */
	Array& operator+=(const DataType &rhs);

	//TBTKFeature Utilities.Array.operatorAddition.1 2019-10-31
	//TBTKFeature Utilities.Array.operatorAddition.2 2019-10-31
	//TBTKFeature Utilities.Array.operatorAddition.3 2019-10-31
	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The sum of the left and right hand side. */
	Array operator+(const Array &rhs) const{
		Array result = *this;

		return result += rhs;
	}

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new Array with the right hand side added to each element.
	 */
	Array operator+(const DataType &rhs){
		Array result = *this;

		return result += rhs;
	}

	/** Subtraction equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after the right hand side has been subtracted. */
	Array& operator-=(const Array &rhs);

	/** Subtraction equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after the right hand side has been subtracted
	 *  from each element. */
	Array& operator-=(const DataType &rhs);

	//TBTKFeature Utilities.Array.operatorSubtraction.1 2019-10-31
	//TBTKFeature Utilities.Array.operatorSubtraction.2 2019-10-31
	//TBTKFeature Utilities.Array.operatorSubtraction.3 2019-10-31
	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The sum of the left and right hand side. */
	Array operator-(const Array &rhs) const{
		Array result = *this;

		return result -= rhs;
	}

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new Array with the right hand side subtracted from each
	 *  element. */
	Array operator-(const DataType &rhs){
		Array result = *this;

		return result -= rhs;
	}
	/** Negative operator.
	 *
	 *  @return An Array with all elements the negative of the original
	 *  one. */
	Array operator-() const;

	/** Multiplication equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after multiplication by the right hand side. */
	Array& operator*=(const DataType &rhs);

	//TBTKFeature Utilities.Array.operatorMultiplication.1 2019-10-31
	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The product of the left and right hand side. */
	Array operator*(const DataType &rhs) const{
		Array result = *this;

		return result *= rhs;
	}

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

	/** Multiplication operator. Multiplies two @link Array Arrays@endlink
	 *  of rank one or two. If \f$u_i\f$ and \f$v_i\f$ are @link Array
	 *  Arrays@endlink with a single Subindex and \f$M_{ij}\f$ and
	 *  \f$N_{ij}\f$ are @link Array Arrays@endlink with two @link Subindex
	 *  Subindices@endlink, the possible products are
	 *
	 *  **Rank 1 times rank 1:**
	 *
	 *  \f$\sum_{i}u_{i}v_{i}\f$
	 *
	 *  The result is an Array with rank 1 and a single element.
	 *
	 *  **Rank 1 times rank 2:**
	 *
	 *  \f$\sum_{i}u_{i}M_{ij}\f$
	 *
	 *  The result is an Array with rank 1.
	 *
	 *  **Rank 2 times rank 1:**
	 *
	 *  \f$\sum_{j}M_{ij}u_{j}\f$
	 *
	 *  The result is an Array with rank 1.
	 *
	 *  **Rank 2 times rank 2:**
	 *
	 *  \f$\sum_{j}M_{ij}N_{jk}\f$
	 *
	 *  The result is an Array with rank 2.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return An Array containing the product of the two @link Array
	 *  Arrays@endlink. */
	Array operator*(const Array &rhs) const;

	/** Division equality operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The Array after division by the right hand side. */
	Array& operator/=(const DataType &rhs);

	//TBTKFeature Utilities.Array.operatorDivision.1 2019-10-31
	/** Division operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The quotient between the left and right hand side. */
	Array operator/(const DataType &rhs) const{
		Array result = *this;

		return result /= rhs;
	}

	/** Comparison operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return True if the size and individual entries of the left and
	 *  right hand sides are equal, otherwise false. */
	bool operator==(const Array &rhs) const;

	/** Contract two @link Array Arrays @endlink by summing over one or
	 *  more common indices.
	 *
	 *  \f$ A_{ijk} = \sum_{ab}B_{ijab}C_{akb}\f$
	 *
	 *  The function takes two @link Array Arrays
	 *  @endlink and two patterns. The patterns must have the same number
	 *  of @link Subindex Subindices @endlink as the corresponding Array
	 *  and contain either wildcards or labeled wildcards (see Subindex).
	 *
	 *  Labeled wildcards in the two patterns are identified and the
	 *  corresponding indices are summed over. The @link Subindex
	 *  Subindices@endlink of the resulting Array is ordered in the same
	 *  order as the original @link Array Arrays@endlink, with the @link
	 *  Subindex Subindices@endlink of the first Array coming before those
	 *  of the second.
	 *
	 *  If B and C are two @linkArray Arrays@endlink with four and three
	 *  @link SUbindex Subindices@endlink each, the expression above is
	 *  calculated using
	 *  ```cpp
	 *    Array<DataType> A = Array<DataType>::contract(
	 *      B,
	 *      {_a_, _a_, _aX_(0), _aX_(1)}},
	 *      C,
	 *      {_aX_(0), _a_, _aX_(1)}
	 *    );
	 *  ```
	 *
	 *  @param array0 The first array.
	 *  @param pattern0 The pattern associated with the first Array.
	 *  @param array1 The second Array.
	 *  @param pattern1 The pattern associated with the second Array.
	 *
	 *  @return A new Array resulting from contracting the labled wildcards
	 *  in the two Array. */
	static Array contract(
		const Array &array0,
		const std::vector<Subindex> &pattern0,
		const Array &array1,
		const std::vector<Subindex> &pattern1
	);

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
	const Ranges& getRanges() const;

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

	/** Implementes Serializable::serialize(). */
	std::string serialize(Mode mode) const;
private:
	/** Data data. */
	CArray<DataType> data;

	/** Ranges. */
	Ranges ranges;

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

	/** Friend class. */
	friend class Math::ArrayAlgorithms<DataType>;
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
Array<DataType>::Array(const Range &range){
	ranges.push_back(range.getResolution());
	data = CArray<DataType>(range.getResolution());
	for(unsigned int n = 0; n < range.getResolution(); n++)
		data[n] = (DataType)range[n];
}

template<typename DataType>
Array<DataType>::Array(const std::string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "Array", mode),
		"Array::Array()",
		"Unable to parse string as Array '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
	{
		try{
			nlohmann::json j
				= nlohmann::json::parse(serialization);
			data = CArray<DataType>(j.at("data"), mode);
			ranges = j.at("ranges").get<std::vector<unsigned int>>();
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"Array::Array()",
				"Unable to parse string as Array '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Array::Array()",
			"Unable to parse string as Array '" << serialization
			<< "'.",
			""
		);
	}
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
template<typename CastType>
inline Array<DataType>::operator Array<CastType>() const{
	Array<CastType> result = Array<CastType>::create(ranges);
	CArray<CastType> &resultData = result.getData();
	for(unsigned int n = 0; n < data.getSize(); n++)
		resultData[n] = (CastType)data[n];

	return result;
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
inline typename Array<DataType>::Modifier Array<DataType>::operator()(
	const std::vector<Subindex> &pattern
){
	TBTKAssert(
		pattern.size() == ranges.size(),
		"Array::operator()",
		"Invalid pattern. The pattern must have the same number of"
		" elements as the Array rank. But the pattern has '"
		<< pattern.size() << "' elements, while the Array has rank '"
		<< ranges.size() << "'.",
		""
	);
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern[n].isWildcard() || pattern[n].isLabeledWildcard())
			continue;

		TBTKAssert(
			pattern[n] >= 0,
			"Array::operator()",
			"Invalid pattern. Subindices must be wildcards, labled"
			<< " wildcards, or positive number, but '"
			<< pattern[n] << "' found in position '" << n << "'.",
			""
		);
		TBTKAssert(
			pattern[n] < (Subindex)ranges[n],
			"Arrau::operator()",
			"Invalid pattern. The value '" << pattern[n] << "' was"
			<< " found in position '" << n << "', which is larger"
			<< " or equal to the corresponding range '"
			<< ranges[n]-1 << "'.",
			""
		);
	}
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern[n].isLabeledWildcard()){
			for(unsigned int c = n+1; c < pattern.size(); c++){
				if(pattern[n] == pattern[c]){
					TBTKAssert(
						ranges[n] == ranges[c],
						"Array::operator()",
						"Invalid pattern. Found"
						<< " labeled wildcards with"
						<< " the same label in"
						<< " position '" << n << "'"
						<< " and '" << c << "', but"
						<< " the ranges are different"
						<< " for these positions.",
						""
					);
				}
			}
		}
	}

	return Modifier(*this, pattern);
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator+=(
	const Array<DataType> &rhs
){
	assertCompatibleRanges(rhs, "operator+=()");

	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] += rhs.data[n];

	return *this;
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator+=(const DataType &rhs){
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] += rhs;

	return *this;
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator-=(
	const Array<DataType> &rhs
){
	assertCompatibleRanges(rhs, "operator-=()");

	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] -= rhs.data[n];

	return *this;
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator-=(
	const DataType &rhs
){
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] -= rhs;

	return *this;
}

template<typename DataType>
inline Array<DataType> Array<DataType>::operator-() const{
	Array<DataType> result = Array<DataType>::create(ranges);
	for(unsigned int n = 0; n < data.getSize(); n++)
		result.data[n] = -data[n];

	return result;
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator*=(
	const DataType &rhs
){
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] *= rhs;

	return *this;
}

template<typename DataType>
Array<DataType> Array<DataType>::operator*(const Array<DataType> &rhs) const{
	switch(ranges.size()){
	case 1:
		switch(rhs.ranges.size()){
		case 1:
			return contract(
				*this,
				{IDX_ALL_(0)},
				rhs,
				{IDX_ALL_(0)}
			);
		case 2:
			return contract(
				*this,
				{IDX_ALL_(0)},
				rhs,
				{IDX_ALL_(0), IDX_ALL}
			);
		default:
			TBTKExit(
				"Array::operator*()",
				"Unsupported Array rank. Multiplication is"
				<< " only possible for Arrays of rank one or"
				<< " two, but the left hand side has rank '"
				<< ranges.size() << "'.",
				""
			);
		}
	case 2:
		switch(rhs.ranges.size()){
		case 1:
			return contract(
				*this,
				{IDX_ALL, IDX_ALL_(0)},
				rhs,
				{IDX_ALL_(0)}
			);
		case 2:
			return contract(
				*this,
				{IDX_ALL, IDX_ALL_(0)},
				rhs,
				{IDX_ALL_(0), IDX_ALL}
			);
		default:
			TBTKExit(
				"Array::operator*()",
				"Unsupported Array rank. Multiplication is"
				<< " only possible for Arrays of rank one or"
				<< " two, but the left hand side has rank '"
				<< ranges.size() << "'.",
				""
			);
		}
	default:
		TBTKExit(
			"Array::operator*()",
			"Unsupported Array rank. Multiplication is only"
			<< " possible for Arrays of rank one or two, but the"
			<< " left hand side has rank '" << ranges.size()
			<< "'.",
			""
		);
	}
}

template<typename DataType>
inline Array<DataType>& Array<DataType>::operator/=(const DataType &rhs){
	for(unsigned int n = 0; n < data.getSize(); n++)
		data[n] /= rhs;

	return *this;
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
Array<DataType> Array<DataType>::contract(
	const Array &array0,
	const std::vector<Subindex> &pattern0,
	const Array &array1,
	const std::vector<Subindex> &pattern1
){
	const std::vector<unsigned int> &ranges0 = array0.getRanges();
	const std::vector<unsigned int> &ranges1 = array1.getRanges();
	TBTKAssert(
		ranges0.size() == pattern0.size(),
		"Array::contract()",
		"Incompatible pattern size. The number of elements in"
		<< " 'pattern0' must be the same as the number of ranges in"
		<< " 'array0', but 'pattern0' has '" << pattern0.size() << "'"
		<< " elements while 'array0' has '"
		<< ranges0.size() << "' ranges.",
		""
	);
	TBTKAssert(
		ranges1.size() == pattern1.size(),
		"Array::contract()",
		"Incompatible pattern size. The number of elements in"
		<< " 'pattern1' must be the same as the number of ranges in"
		<< " 'array1', but 'pattern1' has '" << pattern1.size() << "'"
		<< " elements while 'array1' has '"
		<< ranges1.size() << "' ranges.",
		""
	);
	for(unsigned int n = 0; n < pattern0.size(); n++){
		TBTKAssert(
			pattern0[n].isWildcard()
			|| pattern0[n].isLabeledWildcard(),
			"Array::contract()",
			"Invalid pattern. The patterns must only contain"
			" wildcards or labeled wildcards, but found '"
			<< pattern0[n] << "' in position '" << n << "' of"
			<< " 'pattern0'.",
			""
		);
	}
	for(unsigned int n = 0; n < pattern1.size(); n++){
		TBTKAssert(
			pattern1[n].isWildcard()
			|| pattern1[n].isLabeledWildcard(),
			"Array::contract()",
			"Invalid pattern. The patterns must only contain"
			" wildcards or labeled wildcards, but found '"
			<< pattern1[n] << "' in position '" << n << "' of"
			<< " 'pattern1'.",
			""
		);
	}

	std::vector<unsigned int> summationIndices0;
	std::vector<Subindex> wildcards0;
	for(unsigned int n = 0; n < pattern0.size(); n++){
		if(pattern0[n].isLabeledWildcard()){
			summationIndices0.push_back(n);
			wildcards0.push_back(pattern0[n]);
		}
	}
	std::vector<unsigned int> summationIndices1;
	std::vector<Subindex> wildcards1;
	for(unsigned int n = 0; n < pattern1.size(); n++){
		if(pattern1[n].isLabeledWildcard()){
			summationIndices1.push_back(n);
			wildcards1.push_back(pattern1[n]);
		}
	}

	TBTKAssert(
		wildcards0.size() == wildcards1.size(),
		"Array::contract()",
		"Incompatible patterns. The number of labeled wildcards are"
		<< " different in 'pattern0' and 'pattern1'.",
		""
	);
	for(unsigned int n = 0; n < wildcards0.size(); n++){
		for(unsigned int c = n+1; c < wildcards0.size(); c++){
			TBTKAssert(
				wildcards0[n] != wildcards0[c],
				"Array::contract()",
				"Repeated labeled wildcard in 'pattern0'",
				""
			);
		}
	}
	for(unsigned int n = 0; n < wildcards1.size(); n++){
		for(unsigned int c = n+1; c < wildcards1.size(); c++){
			TBTKAssert(
				wildcards1[n] != wildcards1[c],
				"Array::contract()",
				"Repeated labeled wildcard in 'pattern1'",
				""
			);
		}
	}

	std::vector<unsigned int> summationIndicesMap;
	for(unsigned int n = 0; n < wildcards0.size(); n++){
		for(unsigned int c = 0; c < wildcards1.size(); c++){
			if(wildcards0[n] == wildcards1[c]){
				summationIndicesMap.push_back(
					summationIndices1[c]
				);
				break;
			}
		}
		TBTKAssert(
			summationIndicesMap.size() == n + 1,
			"Array::contract()",
			"Incompatible patterns. The labeled wildcard at"
			<< " position '" << summationIndices0[n] << " in"
			<< " 'pattern0' is missing in 'pattern1'.",
			""
		);
	}

	std::vector<unsigned int> summationRanges;
	for(unsigned int n = 0; n < summationIndices0.size(); n++){
		TBTKAssert(
			ranges0[summationIndices0[n]]
				== ranges1[summationIndicesMap[n]],
			"Array::contract()",
			"Incompatible summation indices. Unable to contract"
			<< " the Subindex at position '"
			<< summationIndices0[n] << "' in 'array0' with the"
			<< " Subindex at position '" << summationIndicesMap[n]
			<< "' in 'array1' since they have different range.",
			""
		);
		summationRanges.push_back(ranges0[summationIndices0[n]]);
	}

	std::vector<unsigned int> resultRanges;
	for(unsigned int n = 0; n < ranges0.size(); n++)
		if(pattern0[n].isWildcard())
			resultRanges.push_back(ranges0[n]);
	for(unsigned int n = 0; n < ranges1.size(); n++)
		if(pattern1[n].isWildcard())
			resultRanges.push_back(ranges1[n]);

	Array result;
	if(resultRanges.size() == 0){
		result = Array({1}, 0);

		std::vector<unsigned int> summationInitialValues(
			summationRanges.size(),
			0
		);
		std::vector<unsigned int> summationIncrements(
			summationRanges.size(),
			1
		);
		MultiCounter<unsigned int> summationCounter(
			summationInitialValues,
			summationRanges,
			summationIncrements
		);
		for(
			summationCounter.reset();
			!summationCounter.done();
			++summationCounter
		){
			std::vector<unsigned int> index0(
				summationCounter.getSize()
			);
			std::vector<unsigned int> index1(
				summationCounter.getSize()
			);
			for(
				unsigned int n = 0;
				n < summationCounter.getSize();
				n++
			){
				index0[summationIndices0[n]]
					= summationCounter[n];
				index1[summationIndicesMap[n]]
					= summationCounter[n];
			}
			result[{0}] += array0[index0]*array1[index1];
		}
	}
	else{
		result = Array::create(resultRanges, 0);

		std::vector<unsigned int> resultInitialValues(
			resultRanges.size(),
			0
		);
		std::vector<unsigned int> resultIncrements(
			resultRanges.size(),
			1
		);
		MultiCounter<unsigned int> resultCounter(
			resultInitialValues,
			resultRanges,
			resultIncrements
		);

		std::vector<unsigned int> summationInitialValues(
		summationRanges.size(),
		0
		);
		std::vector<unsigned int> summationIncrements(
			summationRanges.size(),
			1
		);
		MultiCounter<unsigned int> summationCounter(
			summationInitialValues,
			summationRanges,
			summationIncrements
		);
		for(resultCounter.reset(); !resultCounter.done(); ++resultCounter){
			std::vector<unsigned int> resultIndex
				= (std::vector<unsigned int>)resultCounter;
			for(
				summationCounter.reset();
				!summationCounter.done();
				++summationCounter
			){
				std::vector<unsigned int> index0(
					resultIndex.begin(),
					resultIndex.begin() + ranges0.size()
						- summationIndices0.size()
				);
				std::vector<unsigned int> index1(
					resultIndex.begin() + ranges0.size()
						- summationIndices0.size(),
					resultIndex.end()
				);
				for(
					unsigned int n = 0;
					n < summationCounter.getSize();
					n++
				){
					index1.insert(
						index1.begin() + summationIndices1[n],
						0
					);
				}
				for(
					unsigned int n = 0;
					n < summationCounter.getSize();
					n++
				){
					index0.insert(
						index0.begin() + summationIndices0[n],
						summationCounter[n]
					);
					index1[summationIndicesMap[n]]
						= summationCounter[n];
				}
				result[resultCounter] += array0[index0]*array1[index1];
			}
		}
	}

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
inline const typename Array<DataType>::Ranges& Array<DataType>::getRanges() const{
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
inline std::string Array<DataType>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Array";
		j["data"] = data.serialize(mode);
		j["ranges"] = ranges;

		return j.dump();
	}
	default:
		TBTKExit(
			"Array::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
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

template<typename DataType>
Array<DataType>::Modifier::Modifier(
	Array &array,
	const std::vector<Subindex> &pattern
) :
	array(array),
	pattern(pattern)
{
}

template<typename DataType>
typename Array<DataType>::Modifier& Array<DataType>::Modifier::operator=(
	const DataType &rhs
){
	std::vector<std::vector<unsigned int>> indices
		= getAllCompatibleIndices();
	for(auto index : indices)
		array[index] = rhs;

	return *this;
}

template<typename DataType>
typename Array<DataType>::Modifier& Array<DataType>::Modifier::operator+=(
	const DataType &rhs
){
	std::vector<std::vector<unsigned int>> indices
		= getAllCompatibleIndices();
	for(auto index : indices)
		array[index] += rhs;

	return *this;
}

template<typename DataType>
typename Array<DataType>::Modifier& Array<DataType>::Modifier::operator-=(
	const DataType &rhs
){
	std::vector<std::vector<unsigned int>> indices
		= getAllCompatibleIndices();
	for(auto index : indices)
		array[index] -= rhs;

	return *this;
}

template<typename DataType>
typename Array<DataType>::Modifier& Array<DataType>::Modifier::operator*=(
	const DataType &rhs
){
	std::vector<std::vector<unsigned int>> indices
		= getAllCompatibleIndices();
	for(auto index : indices)
		array[index] *= rhs;

	return *this;
}

template<typename DataType>
typename Array<DataType>::Modifier& Array<DataType>::Modifier::operator/=(
	const DataType &rhs
){
	std::vector<std::vector<unsigned int>> indices
		= getAllCompatibleIndices();
	for(auto index : indices)
		array[index] /= rhs;

	return *this;
}

template<typename DataType>
std::vector<
	std::vector<unsigned int>
> Array<DataType>::Modifier::getAllCompatibleIndices(
) const{
	std::vector<Subindex> patternCopy = pattern;
	std::vector<std::vector<unsigned int>> wildcardPositions;
	for(unsigned int n = 0; n < patternCopy.size(); n++){
		if(patternCopy[n].isWildcard()){
			wildcardPositions.push_back(
				std::vector<unsigned int>()
			);
			wildcardPositions.back().push_back(n);
		}
		else if(patternCopy[n].isLabeledWildcard()){
			wildcardPositions.push_back(
				std::vector<unsigned int>()
			);
			wildcardPositions.back().push_back(n);
			for(
				unsigned int c = n + 1;
				c < patternCopy.size();
				c++
			){
				if(patternCopy[n] == patternCopy[c]){
					wildcardPositions.back().push_back(c);
					patternCopy[c] = 0;
				}
			}
		}
	}

	const std::vector<unsigned int> &ranges = array.getRanges();
	std::vector<unsigned int> limits;
	for(unsigned int n = 0; n < wildcardPositions.size(); n++)
		limits.push_back(ranges[wildcardPositions[n][0]]);

	std::vector<std::vector<unsigned int>> compatibleIndices;
	MultiCounter<unsigned int> wildcardCounter(
		std::vector<unsigned int>(wildcardPositions.size(), 0),
		limits,
		std::vector<unsigned int>(wildcardPositions.size(), 1)
	);
	for(
		wildcardCounter.reset();
		!wildcardCounter.done();
		++wildcardCounter
	){
		std::vector<unsigned int> index;
		for(unsigned int n = 0; n < pattern.size(); n++){
			if(pattern[n] >= 0){
				index.push_back((unsigned int)pattern[n]);
			}
			else if(
				pattern[n].isWildcard()
				|| pattern[n].isLabeledWildcard()
			){
				index.push_back(0);
			}
			else{
				TBTKExit(
					"Array::Modifier::getAllCompatibleIndices()",
					"Invalid pattern Subindex.",
					"This should never happen, contact the"
					<< " developer."
				);
			}
		}
		for(unsigned int n = 0; n < wildcardPositions.size(); n++){
			for(
				unsigned int c = 0;
				c < wildcardPositions[n].size();
				c++
			){
				index[wildcardPositions[n][c]]
					= wildcardCounter[n];
			}
		}

		compatibleIndices.push_back(index);
	}

	return compatibleIndices;
}

template<typename DataType>
bool Array<DataType>::Ranges::operator==(const Ranges &rhs) const{
	if(size() != rhs.size())
		return false;

	for(unsigned int n = 0; n < size(); n++)
		if((*this)[n] != rhs[n])
			return false;

	return true;
}

template<typename DataType>
bool Array<DataType>::Ranges::operator!=(const Ranges &rhs) const{
	return !(*this == rhs);
}

}; //End of namesapce TBTK

#endif
