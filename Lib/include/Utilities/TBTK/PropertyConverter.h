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
 *  @file PropertyConverter.h
 *  @brief Converts @link Property Properties@endlink to @link AnnotatedArray
 *  AnnotatedArrays@endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_CONVERTER
#define COM_DAFER45_TBTK_PROPERTY_CONVERTER

#include "TBTK/AnnotatedArray.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{

/** @brief Convert @link Property Properties@endlink to @link AnnotatedArray
 *  AnnotatedArrays@endlink.
 *
 *  If the @link Property::AbstractPropert Property@endlink is on the format
 *  Index::Descriptor::Format::None or IndexDescriptor::Format::Ranges, it can
 *  be converted to an AnnotatedArray using
 *  ```cpp
 *    AnnotatedArray<DataType, Subindex> array
 *      = PropertyConverter::convert(property);
 *  ```
 *  where *DataType* is the data type of the Propertys data elements. If the
 *  property has a block structure, the last array subindex corresponds to this
 *  block index. For example, for a Property::EnergyResolvedProperty, the last
 *  subindex corresponds to energy.
 *
 *  Similart to above, if the @link Property::AbstractProperty Property@endlink
 *  is on the format IndexDescriptor::Format::Custom, it can be converted to an
 *  AnnotatedArray using
 *  ```cpp
 *    AnnotatedArray<DataType, Subindex> array
 *      = PropertyConverter::convert(property, pattern);
 *  ```
 *  Here *pattern* is an Index that will be matched against every Index in the
 *  *property* to determine whether it should be included in the output or not.
 *  The resulting AnnotatedArray will have minimal possible ranges to conver
 *  all included @link Index Indices@endlink. The ranges for the AnnotatedArray
 *  start from zero, but the axes that can be obtained using
 *  ```cpp
 *    const std::vector<std::vector<Subindex>> &axes = array.getAxes();
 *  ```
 *  contains information about which Subindex that corresponds to which array
 *  entry. The @link Subindex Subindices@endlink for the nth dimension is
 *  contained in axes[n].
 *
 *  # Example
 *  \snippet Utilities/PropertyConverter.cpp PropertyConverter
 *  ## Output
 *  \snippet output/Utilities/PropertyConverter.txt PropertyConverter */
class PropertyConverter{
public:
	/** Converts an AbstractProperty on the format
	 *  IndexDescriptor::Format::None or IndexDescriptor::Format::Ranges to
	 *  an AnnotatedArray.
	 *
	 *  @param abstractProperty The property to convert. */
	template<typename DataType>
	static AnnotatedArray<DataType, Subindex> convert(
		const Property::AbstractProperty<DataType> &abstractProperty
	);

	/** Converts an AbstractProperty on the format
	 *  IndexDescriptor::Format::Custom to an AnnotatedArray. The data for
	 *  all points matching a given pattern Index will be extracted.
	 *
	 *  The dimension of the resulting AnnotatedArray will be equal to the
	 *  number of wild cards (plus one if the block size is larger than
	 *  one). The size of a given dimension is given by the difference
	 *  between the largest and smallest Subindex for the corresponding
	 *  Subindex position. For example, if the pattern is {_a_, 5, _a_} and
	 *  the property contains data for {3, 5, 4}, {7, 5, 5}, {4, 5, 9}, the
	 *  first dimension will range from 0 to 4 (= 7-3) and the second
	 *  dimension will range from 0 to 6 (= 9-4). The corresponding axes
	 *  will contain {3, 4, 5, 6, 7} and {4, 5, 6, 7, 8, 9}. If the
	 *  property has a block size, the last dimension ranges from 0 to
	 *  blockSize, with the corresponding axis containing
	 *  {0, ..., blockSize-1}.
	 *
	 *  If the @link Index Indices@endlink are not dense, AnnotatedArray
	 *  elements corresponding to "missing Indices" will be set to zero.
	 *
	 *  @param abstractProperty The property to convert.
	 *  @param pattern Pattern that determines which points to extract the
	 *  data for. */
	template<typename DataType>
	static AnnotatedArray<DataType, Subindex> convert(
		const Property::AbstractProperty<DataType> &abstractProperty,
		const Index &pattern
	);
};

template<typename DataType>
AnnotatedArray<DataType, Subindex> PropertyConverter::convert(
	const Property::AbstractProperty<DataType> &abstractProperty
){
	IndexDescriptor::Format format
		= abstractProperty.getIndexDescriptor().getFormat();

	switch(format){
	case IndexDescriptor::Format::None:
	{
		Array<DataType> array({abstractProperty.getBlockSize()});
		std::vector<std::vector<Subindex>> axis(1);
		for(unsigned int n = 0; n < abstractProperty.getBlockSize(); n++){
			array[{n}] = abstractProperty(n);
			axis[0].push_back(n);
		}
		return AnnotatedArray<DataType, Subindex>(array, axis);
	}
	case IndexDescriptor::Format::Ranges:
	{
		std::vector<int> ranges
			= abstractProperty.getIndexDescriptor().getRanges();
		if(abstractProperty.getBlockSize() != 1)
			ranges.push_back(abstractProperty.getBlockSize());
		std::vector<unsigned int> rangesUnsignedInt;
		for(unsigned int n = 0; n < ranges.size(); n++)
			rangesUnsignedInt.push_back(ranges[n]);

		Array<DataType> array = Array<DataType>::create(rangesUnsignedInt);
		const std::vector<DataType> &data = abstractProperty.getData();
		for(unsigned int n = 0; n < abstractProperty.getSize(); n++)
			array[n] = data[n];

		std::vector<std::vector<Subindex>> axes(ranges.size());
		for(unsigned int n = 0; n < ranges.size(); n++)
			for(unsigned int c = 0; c < (unsigned int)ranges[n]; c++)
				axes[n].push_back(c);

		return AnnotatedArray<DataType, Subindex>(array, axes);
	}
	default:
		TBTKExit(
			"PropertyConverter::convert()",
			"Unsupported format. Only IndexDescriptor::Format::None and"
			<< " IndexDescriptor::Format::Ranges supported.",
			""
		);
	}
}

template<typename DataType>
AnnotatedArray<DataType, Subindex> PropertyConverter::convert(
	const Property::AbstractProperty<DataType> &abstractProperty,
	const Index &pattern
){
	IndexDescriptor::Format format
		= abstractProperty.getIndexDescriptor().getFormat();

	switch(format){
	case IndexDescriptor::Format::Custom:
	{
		std::vector<unsigned int> wildcardPositions;
		for(unsigned int n = 0; n < pattern.getSize(); n++)
			if(pattern[n].isWildcard())
				wildcardPositions.push_back(n);

		const IndexTree &indexTree
			= abstractProperty.getIndexDescriptor().getIndexTree();
		std::vector<Index> indexList = indexTree.getIndexList(pattern);

		std::vector<Subindex> minSubindices;
		std::vector<Subindex> maxSubindices;
		for(unsigned int n = 0; n < wildcardPositions.size(); n++){
			Subindex min = indexList[0][wildcardPositions[n]];
			Subindex max = indexList[0][wildcardPositions[n]];
			for(unsigned int c = 1; c < indexList.size(); c++){
				Subindex subindex
					= indexList[c][wildcardPositions[n]];
				if(min > subindex)
					min = subindex;
				if(max < subindex)
					max = subindex;
			}
			minSubindices.push_back(min);
			maxSubindices.push_back(max);
		}
		if(abstractProperty.getBlockSize() > 1){
			minSubindices.push_back(0);
			maxSubindices.push_back(
				abstractProperty.getBlockSize() - 1
			);
		}

		std::vector<unsigned int> ranges;
		for(unsigned int n = 0; n < minSubindices.size(); n++){
			ranges.push_back(
				maxSubindices[n] - minSubindices[n] + 1
			);
		}
		Array<DataType> array = Array<DataType>::create(
			ranges,
			DataType(0)
		);
		for(unsigned int n = 0; n < indexList.size(); n++){
			const Index index = indexList[n];
			std::vector<unsigned int> arrayIndices;
			for(unsigned int c = 0; c < wildcardPositions.size(); c++){
				arrayIndices.push_back(
					index[wildcardPositions[c]]
					- minSubindices[c]
				);
			}
			if(abstractProperty.getBlockSize() > 1){
				for(
					unsigned int c = 0;
					c < abstractProperty.getBlockSize();
					c++
				){
					std::vector<
						unsigned int
					> blockExtendedArrayIndices
						= arrayIndices;
					blockExtendedArrayIndices.push_back(c);

					array[blockExtendedArrayIndices]
						= abstractProperty(index, c);
				}
			}
			else{
				array[arrayIndices] = abstractProperty(index);
			}
		}

		std::vector<std::vector<Subindex>> axes(ranges.size());
		for(
			unsigned int n = 0;
			n < ranges.size();
			n++
		){
			for(
				unsigned int c = 0;
				c < (unsigned int)ranges[n];
				c++
			){
				axes[n].push_back(minSubindices[n] + c);
			}
		}

		return AnnotatedArray<DataType, Subindex>(array, axes);
	}
	default:
		TBTKExit(
			"PropertyConverter::convert()",
			"Unsupported format. Only IndexDescriptor::Format::None and"
			<< " IndexDescriptor::Format::Ranges supported.",
			""
		);
	}
}

}; //End of namesapce TBTK

#endif
