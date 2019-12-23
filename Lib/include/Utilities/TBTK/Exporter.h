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
#include "TBTK/SpinMatrix.h"

#include <complex>
#include <fstream>
#include <string>
#include <vector>

namespace TBTK{

/** @brief Exports data on human readable format.
 *
 *  The Exporter can write @link Array Arrays@endlink and @link
 *  Property::AbstractProperty Properties@endlink to file.
 *
 *  # Output format
 *  ## Row major order
 *  By default, the data is written to file on row major order. This means that
 *  the right most Subindex increments the fastest, while the left most
 *  subindex changes the slowest. For @link Property::AbstractProperty
 *  Properties@endlink with a block structure, the intra block index should be
 *  understood as standing to the right of any Index. Similarly, any data type
 *  with indexable internal structure should be understood as having its
 *  indices to the right of any Index and intra block index.
 *
 *  For example, assume the property has index structure {x, y}, a block index
 *  n, and data type std::complexa<double>. The total index structure can then
 *  be thought of as being {x, y, n, c}, where c=0 and c=1 corresponds to real
 *  and imaginary part, respectively. If the dimensions are given by {SIZE_X,
 *  SIZE_Y, BLOCK_SIZE, 2}, the element order is given by
 *  <center>\f$2*(BLOCK\_SIZE*(SIZE\_Y*x + y) + n) + c\f$.</center>
 *
 *  ## Column major order
 *  Languages such as Fortran and MATLAB use column major order. To simplify
 *  import to such languages, it is also possible to export the data on column
 *  major order. To do so, set the format to column major before exporting the
 *  data.
 *  ```cpp
 *    exporter.setFormat(Exporter::Format::ColumnMajor);
 *  ```
 *
 *  # Arrays
 *  @link Array Arrays@endlink are exported as follows
 *  ```cpp
 *    Exporter exporter;
 *    exporter.save(array, "Filename");
 *  ```
 *
 *  # Properties on the None and Ranges format
 *  @link Property::AbstractProperty Properties@endlink that are on the formats
 *  IndexDescriptor::Format::None and IndexDescriptor::Format::Ranges can be
 *  exported as follows.
 *  ```cpp
 *    Exporter exporter;
 *    exporter.save(property, "Filename");
 *  ```
 *
 *  # Properties on the Custom format
 *  @link Property::AbstractProperty Properties@endlink that are on the format
 *  IndexDescriptor::Format::Custom can be exported as follows.
 *  ```cpp
 *    Exporter exporter;
 *    exporter.save({_a_, 5, _a_}, property, "Filename");
 *  ```
 *  Here it is assumed that *property* has the index structure {x, y, z} and
 *  that the data for all indices satisfying the pattern {x, 5, z}. are to be
 *  exported.
 *
 *  # Export to external languages
 *  Below we demonstrate how to export data to other languages. The example is
 *  for a density with the index structure {x, y, z} and with the number of
 *  elements for the corresponding dimensions being {SIZE_X, SIZE_Y, SIZE_Z}.
 *  ## MATLAB
 *
 *  ### Export from C++
 *
 *  ```cpp
 *    Exporter exporter;
 *    exporter.setFormat(Exporter::Format::ColumnMajor);
 *    exporter.save({_a_, _a_, _a_}, density, "Filename");
 *  ```
 *
 *  ### Import to MATLAB
 *
 *  ```matlab
 *    data = dlmread('Filename')
 *    density = reshape(data, [SIZE_X, SIZE_Y, SIZE_Z])
 *  ```
 *
 *  ## Python
 *
 *  ### Export from C++
 *
 *  ```cpp
 *    Exporter exporter;
 *    exporter.save({_a_, _a_, _a_}, density, "Filename");
 *  ```
 *
 *  ### Import to Python
 *
 *  ```python
 *    import numpy as np
 *    density = np.loadtxt("Filename").reshape(SIZE_X, SIZE_Y, SIZE_Z)
 *  ``` */
class Exporter{
public:
	/** Enum class for */
	enum class Format {RowMajor, ColumnMajor};

	/** Default constructor. */
	Exporter();

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

	/** Set the output format. The default value is Format::RowMajor
	 *
	 *  @param format The output format to use.*/
	void setFormat(Format format);
private:
	/** The output format. */
	Format format;

	/** Write double to output stream. */
	template<typename DataType>
	void write(std::ofstream &stream, const DataType &value) const;
};

inline Exporter::Exporter(){
	format = Format::RowMajor;
}

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
	Array<DataType> outputArray;
	switch(format){
	case Format::RowMajor:
		outputArray = array;
		break;
	case Format::ColumnMajor:
		outputArray = array.getArrayWithReversedIndices();
		break;
	default:
		TBTKExit(
			"Exporter::save()",
			"Unknown format '" << static_cast<int>(format) << "'.",
			""
		);
	}

	std::ofstream fout(filename);
	if(!fout){
		TBTKExit(
			"Exporter::save()",
			"Unable to open file '" << filename << "'.",
			""
		);
	}
	const std::vector<unsigned int> &ranges = outputArray.getRanges();
	std::vector<unsigned int> begin = ranges;
	std::vector<unsigned int> end = ranges;
	std::vector<unsigned int> increment = ranges;
	for(unsigned int n = 0; n < begin.size(); n++){
		begin[n] = 0;
		increment[n] = 1;
	}
	MultiCounter<unsigned int> counter(begin, end, increment);
	for(counter.reset(); !counter.done(); ++counter)
		write(fout, outputArray[counter]);
	fout.close();
}

inline void Exporter::setFormat(Format format){
	this->format = format;
}

template<typename DataType>
void Exporter::write(std::ofstream &stream, const DataType &value) const{
	stream << value << "\n";
}

template<>
inline void Exporter::write<std::complex<double>>(
	std::ofstream &stream,
	const std::complex<double> &value
) const{
	stream << real(value) << "\n";
	stream << imag(value) << "\n";
}

template<>
inline void Exporter::write<SpinMatrix>(
	std::ofstream &stream,
	const SpinMatrix &value
) const{
	switch(format){
	case Format::RowMajor:
		write(stream, value.at(0, 0));
		write(stream, value.at(0, 1));
		write(stream, value.at(1, 0));
		write(stream, value.at(1, 1));
		break;
	case Format::ColumnMajor:
		write(stream, value.at(0, 0));
		write(stream, value.at(1, 0));
		write(stream, value.at(0, 1));
		write(stream, value.at(1, 1));
		break;
	default:
		TBTKExit(
			"Exporter::write()",
			"This should never happen, contact the developer.",
			""
		);
	}
}

}; //End of namesapce TBTK

#endif
