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
 *  @file Exporter.h
 *  @brief Exports data to human readable format.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EXPORTER
#define COM_DAFER45_TBTK_EXPORTER

#include "TBTK/AnnotatedArray.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/PropertyConverter.h"

#include <fstream>
#include <string>
#include <vector>

namespace TBTK{

/** @brief Exports data on human readable format.
 *
 *  The Exporter can write @link Array Arrays@endlink and @link
 *  Property::AbstractProperty Properties@endlink to a comma separated file.
 *
 *  # Output format
 *  By default, the data is written to file on column major order. This means
 *  that the right most Subindex increments the fastest, while the left most
 *  subindex changes the slowest. For @link Property::AbstractProperty
 *  Properties@endlink with a block structure, the intra block index should be
 *  understood as standing to the left of any Index. Similarly, any data type
 *  with indexable internal structure should be understood as having it's
 *  indices to the right of any Index and intra block index.
 *
 *  For example, if the property has index structure {x, y}, a block index n,
 *  and data type std::complexa<double>. The total index structure can be
 *  thought of as being {x, y, n, c}, where c=0 and c=1 corresponds to real and
 *  imaginary part, respectively. If the dimensions are given by {SIZE_X,
 *  SIZE_Y, BLOCK_SIZE, 2}, the element order is given by
 *  <center>\f$2*(BLOCK\_SIZE*(SIZE\_Y*x + y) + n) + c\f$.</center> */
class Exporter{
public:
	/** Export a @link Property::AbstractProperty Property@endlink on the
	 *  format IndexDescriptor::Format::None or
	 *  IndexDescriptor::Format::Ranges.
	 *
	 *  @param abstractProperty The @link Property::AbstractProperty
	 *  Property@endlink to export.
	 *
	 *  @param filename The name of the file to write the data to. */
	template<typename DataType>
	void save(
		const Property::AbstractProperty<DataType> &abstractProperty,
		const std::string &filename
	) const;

	/** Export a @link Property::AbstractProperty Property@endlink on the
	 *  format IndexDescriptor::Format::Custom.
	 *
	 *  @param abstractProperty The @link Property::AbstractProperty
	 *  Property@endlink to export.
	 *
	 *  @param pattern Pattern that determines which points to export the
	 *  data for.
	 *
	 *  @param filename The name of the file to write the data to. */
	template<typename DataType>
	void save(
		const Property::AbstractProperty<DataType> &abstractProperty,
		const Index &pattern,
		const std::string &filename
	) const;

	/** Export an Array to file.
	 *
	 *  @param array The Array to export.
	 *  @param filename The name of file to write the data to. */
	template<typename DataType>
	void save(
		const Array<DataType> &array,
		const std::string &filename
	) const;
};

template<typename DataType>
void Exporter::save(
	const Property::AbstractProperty<DataType> &abstractProperty,
	const std::string &filename
) const{
	AnnotatedArray<DataType, Subindex> annotatedArray
		= PropertyConverter::convert(abstractProperty);

	save(annotatedArray, filename);
}

template<typename DataType>
void Exporter::save(
	const Property::AbstractProperty<DataType> &abstractProperty,
	const Index &pattern,
	const std::string &filename
) const{
	AnnotatedArray<DataType, Subindex> annotatedArray
		= PropertyConverter::convert(abstractProperty, pattern);

	save(annotatedArray, filename);
}

template<typename DataType>
void Exporter::save(
	const Array<DataType> &array,
	const std::string &filename
) const{
	std::ofstream fout(filename);
	if(!fout){
		TBTKExit(
			"Exporter::save()",
			"Unable to open file '" << filename << "'.",
			""
		);
	}
	const std::vector<unsigned int> &ranges = array.getRanges();
	std::vector<unsigned int> begin = ranges;
	std::vector<unsigned int> end = ranges;
	std::vector<unsigned int> increment = ranges;
	for(unsigned int n = 0; n < begin.size(); n++){
		begin[n] = 0;
		increment[n] = 1;
	}
	MultiCounter<unsigned int> counter(begin, end, increment);
	switch(ranges.size()){
	case 2:
	{
		unsigned int x =  0;
		for(counter.reset(); !counter.done(); ++counter){
			if(x != counter[0]){
				x = counter[0];
				fout << "\n";
			}
			if(counter[1] != 0)
				fout << ", ";
			fout << array[counter];
		}
		break;
	}
	default:
		for(counter.reset(); !counter.done(); ++counter)
			fout << array[counter] << "\n";
	}
	fout.close();
}

}; //End of namesapce TBTK

#endif
