/* Copyright 2019 Kristofer Björnson
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
 *  @file AnnotatedArray.h
 *  @brief Array with additional information about its axes.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ANNOTATED_ARRAY
#define COM_DAFER45_TBTK_ANNOTATED_ARRAY

#include "TBTK/Array.h"

namespace TBTK{

/** @brief Array with additional information about its axes.
 *
 *  The AnnotatedArray extends the Array with axes. The number of axes must be
 *  the same as the number of Array dimensions and each axis must have the same
 *  number of entries as the corresponding Array range.
 *
 *  # Example
 *  \snippet Utilities/AnnotatedArray.cpp AnnotatedArray
 *  ## Output
 *  \snippet output/Utilities/AnnotatedArray.output AnnotatedArray */
template<typename DataType, typename AxesType>
class AnnotatedArray : public Array<DataType>{
public:
	/** Constructor. */
	AnnotatedArray();

	/** Constructor.
	 *
	 *  @param array The Array data.
	 *  @param axes The axes of the Array. */
	AnnotatedArray(
		const Array<DataType> &array,
		const std::vector<std::vector<AxesType>> &axes
	);

	/** Return the axes of the AnnotatedArray.
	 *
	 *  @return The axes of the AnnotatedArray. */
	const std::vector<std::vector<AxesType>>& getAxes() const;
private:
	std::vector<std::vector<AxesType>> axes;
};

template<typename DataType, typename AxesType>
AnnotatedArray<DataType, AxesType>::AnnotatedArray(){
}

template<typename DataType, typename AxesType>
AnnotatedArray<DataType, AxesType>::AnnotatedArray(
	const Array<DataType> &array,
	const std::vector<std::vector<AxesType>> &axes
) :
	Array<DataType>(array),
	axes(axes)
{
	const std::vector<unsigned int> &ranges = array.getRanges();
	TBTKAssert(
		ranges.size() == axes.size(),
		"AnnotatedArray::AnnotatedArray()",
		"Incompatible dimensions. 'array' has '" << ranges.size()
		<< "' dimensions, while 'axes' have '" << axes.size() << "'"
		<< " dimensions.",
		""
	);
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			ranges[n] == axes[n].size(),
			"AnnotatedArray::AnnotatedArray()",
			"Incompatible sizes. Dimension '" << n << " of 'array'"
			<< " has size '" << ranges[n] << "', but axis '" << n
			<< "' has size '" << axes[n].size() << "'.",
			""
		);
	}
}

template<typename DataType, typename AxesType>
const std::vector<std::vector<AxesType>>& AnnotatedArray<
	DataType,
	AxesType
>::getAxes() const{
	return axes;
}

}; //End of namesapce TBTK

#endif
