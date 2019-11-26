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
 *  AnnotatedArrays@endlin
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_CONVERTER
#define COM_DAFER45_TBTK_PROPERTY_CONVERTER

#include "TBTK/AnnotatedArray.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{

class PropertyConverter{
public:
	/** Converts an AbstractProperty on the format
	 *  IndexDescriptor::Format::None or IndexDescriptor::Format::Ranges to
	 *  an AnnotatedArray. */
	template<typename DataType>
	static AnnotatedArray<DataType, Subindex> convert(
		const Property::AbstractProperty<DataType> &abstractProperty
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

}; //End of namesapce TBTK

#endif
