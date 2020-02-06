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
 *  \snippet output/Utilities/AnnotatedArray.txt AnnotatedArray */
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

	/** Constructs an AnnotatedArray from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the AnnotatedArray.
	 *
	 *  @param mode The mode with which the string has been serialized. */
	AnnotatedArray(
		const std::string &serialization,
		Serializable::Mode mode
	);

	/** Return the axes of the AnnotatedArray.
	 *
	 *  @return The axes of the AnnotatedArray. */
	const std::vector<std::vector<AxesType>>& getAxes() const;

	/** Implements Serilizable::serialize(). */
	std::string serialize(Serializable::Mode mode) const;
private:
	/** The array axes. */
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
AnnotatedArray<DataType, AxesType>::AnnotatedArray(
	const std::string &serialization,
	Serializable::Mode mode
) :
	Array<DataType>(
		Serializable::extractComponent(
			serialization,
			"AnnotatedArray",
			"Array",
			"array",
			mode
		),
		mode
	)
{
	TBTKAssert(
		Serializable::validate(serialization, "AnnotatedArray", mode),
		"AnnotatedArray::AnnotatedArray()",
		"Unable to parse string as AnnotatedArray '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
	{
		try{
			nlohmann::json j
				= nlohmann::json::parse(serialization);
			nlohmann::json jsonAxes = j.at("axes");
			for(
				nlohmann::json::iterator axisIterator
					= jsonAxes.begin();
				axisIterator != jsonAxes.end();
				++axisIterator
			){
				unsigned int axisId
					= atoi(axisIterator.key().c_str());
				for(unsigned int n = axes.size(); n <= axisId; n++)
					axes.push_back(std::vector<AxesType>());
				for(
					nlohmann::json::iterator iterator
						= axisIterator->begin();
					iterator != axisIterator->end();
					++iterator
				){
					axes.back().push_back(
						Serializable::deserialize<
							AxesType
						>(*iterator, mode)
					);
				}
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"AnnotatedArray::AnnotatedArray()",
				"Unable to parse string as AnnotatedArray '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"AnnotatedArray::AnnotatedArray()",
			"Unable to parse string as AnnotatedArray '"
			<< serialization << "'.",
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

template<typename DataType, typename AxesType>
std::string AnnotatedArray<DataType, AxesType>::serialize(
	Serializable::Mode mode
) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AnnotatedArray";
		j["array"] = nlohmann::json::parse(
			Array<DataType>::serialize(mode)
		);
		j["axes"] = nlohmann::json();
		for(unsigned int n = 0; n < axes.size(); n++){
			std::string index = std::to_string(n);
			j["axes"][index] = nlohmann::json();
			for(unsigned int c = 0; c < axes[n].size(); c++){
				j["axes"][index].push_back(
					Serializable::serialize(
						axes[n][c],
						mode
					)
				);
			}
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"AnnotatedArray::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

}; //End of namesapce TBTK

#endif
