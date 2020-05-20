/* Copyright 2020 Kristofer Björnson
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
 *  @file ArrayConverter.h
 *  @brief Convert object to and from @link Array Arrays@endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY_CONVERTER
#define COM_DAFER45_TBTK_ARRAY_CONVERTER

#include "TBTK/Array.h"
#include "TBTK/Vector3d.h"

namespace TBTK{

/** @brief Converts objects to and from @link Array Arrays@endlink. */
class ArrayConverter : public Serializable{
public:
	/** Packs a number of Vector3d objects into an Array with the Vector3d
	 *  objects as columns.
	 *
	 *  @param vectors The vector to use as columns.
	 *
	 *  @return An Array with the vectors as columns. */
	static Array<double> packColumns(const std::vector<Vector3d> &vectors);

	/** Packs a number of Vector3d objects into an Array with the Vector3d
	 *  objects as rows.
	 *
	 *  @param vectors The vector to use as rows.
	 *
	 *  @return An Array with the vectors as rows. */
	static Array<double> packRows(const std::vector<Vector3d> &vectors);

	/** Splits the columns of a 2D Array into Vector3d objects. The Array
	 *  has to have dimension 3xN for some N.
	 *
	 *  @param array The Array to split.
	 *
	 *  @result An std::vector<Vector3d> containing the columns of the
	 *  array. */
	static std::vector<Vector3d> splitColumnsToVector3d(
		const Array<double> &array
	);

	/** Splits the rows of a 2D Array into Vector3d objects. The Array has
	 *  to have dimension Nx3 for some N.
	 *
	 *  @param array The Array to split.
	 *
	 *  @result An std::vector<Vector3d> containing the rows of the array
	 */
	static std::vector<Vector3d> splitRowsToVector3d(
		const Array<double> &array
	);

	/** Converts a Vector3d to an Array.
	 *
	 *  @param v The Vector3d to convert.
	 *
	 *  @return A one-dimensional Array with three elements corresponding
	 *  to the three components of v. */
	static Array<double> toArray(const Vector3d &v);

	/** Converts a one-dimensional Array with three elements to a Vector3d.
	 *
	 *  @param array The Array to convert.
	 *
	 *  @return A Vector3d with components equal to the three corresponding
	 *  elements in the Array. */
	static Vector3d toVector3d(const Array<double> &array);
};

inline Array<double> ArrayConverter::packColumns(
	const std::vector<Vector3d> &vectors
){
	TBTKAssert(
		vectors.size() != 0,
		"ArrayConverter::packColumns()",
		"'vectors' is empty.",
		""
	);
	Array<double> result({3, (unsigned int)vectors.size()});
	for(unsigned int column = 0; column < vectors.size(); column++){
		result[{0, column}] = vectors[column].x;
		result[{1, column}] = vectors[column].y;
		result[{2, column}] = vectors[column].z;
	}

	return result;
}

inline Array<double> ArrayConverter::packRows(
	const std::vector<Vector3d> &vectors
){
	TBTKAssert(
		vectors.size() != 0,
		"ArrayConverter::packRows()",
		"'vectors' is empty.",
		""
	);
	Array<double> result({(unsigned int)vectors.size(), 3});
	for(unsigned int row = 0; row < vectors.size(); row++){
		result[{row, 0}] = vectors[row].x;
		result[{row, 1}] = vectors[row].y;
		result[{row, 2}] = vectors[row].z;
	}

	return result;
}

inline std::vector<Vector3d> ArrayConverter::splitColumnsToVector3d(
	const Array<double> &array
){
	const std::vector<unsigned int> &ranges = array.getRanges();
	TBTKAssert(
		ranges.size() == 2,
		"ArrayConverter::splitColumnsToVector3d()",
		"Unsupported array rank. The array must have rank '2', but has"
		<< " rank '" << ranges.size() << "'.",
		""
	);
	TBTKAssert(
		ranges[0] == 3,
		"ArrayConverter::splitColumnsToVector3d()",
		"Unsupported array rank. The array must have dimension '3xN',"
		<< " but has dimension '" << ranges[0] << "x" << ranges[1]
		<< "'.",
		""
	);

	std::vector<Vector3d> result;
	for(unsigned int column = 0; column < ranges[1]; column++){
		result.push_back(
			Vector3d({
				array[{0, column}],
				array[{1, column}],
				array[{2, column}]
			})
		);
	}

	return result;
}

inline std::vector<Vector3d> ArrayConverter::splitRowsToVector3d(
	const Array<double> &array
){
	const std::vector<unsigned int> &ranges = array.getRanges();
	TBTKAssert(
		ranges.size() == 2,
		"ArrayConverter::splitRowsToVector3d()",
		"Unsupported array rank. The array must have rank '2', but has"
		<< " rank '" << ranges.size() << "'.",
		""
	);
	TBTKAssert(
		ranges[1] == 3,
		"ArrayConverter::splitRowsToVector3d()",
		"Unsupported array rank. The array must have dimension 'Nx3',"
		<< " but has dimension '" << ranges[0] << "x" << ranges[1]
		<< "'.",
		""
	);

	std::vector<Vector3d> result;
	for(unsigned int row = 0; row < ranges[0]; row++){
		result.push_back(
			Vector3d({
				array[{row, 0}],
				array[{row, 1}],
				array[{row, 2}]
			})
		);
	}

	return result;
}

Array<double> ArrayConverter::toArray(const Vector3d &v){
	Array<double> result({3});
	result[{0}] = v.x;
	result[{1}] = v.y;
	result[{2}] = v.z;

	return result;
}

Vector3d ArrayConverter::toVector3d(const Array<double> &array){
	const std::vector<unsigned int> &ranges = array.getRanges();
	TBTKAssert(
		ranges.size() == 1,
		"ArrayConverter::toVector3d()",
		"Unsupported rank. 'array' must have rank '1', but has rank '"
		<< ranges.size() << "'.",
		""
	);
	TBTKAssert(
		ranges[0] == 3,
		"ArrayConverter::toVector3e()",
		"Unsupported size. 'array' must have size '3', but has size '"
		<< ranges[0] << "'.",
		""
	);

	return Vector3d({array[{0}], array[{1}], array[{2}]});
}

}; //End of namesapce TBTK

#endif
