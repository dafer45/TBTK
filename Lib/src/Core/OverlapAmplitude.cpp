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

/** @file OverlapAmplitude.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/OverlapAmplitude.h"
#include "TBTK/Streams.h"

#include <sstream>

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{

OverlapAmplitude::OverlapAmplitude(
	complex<double> amplitude,
	const Index &braIndex,
	const Index &ketIndex
) :
	braIndex(braIndex),
	ketIndex(ketIndex)
{
	this->amplitude = amplitude;
	this->amplitudeCallback = nullptr;
};

OverlapAmplitude::OverlapAmplitude(
	complex<double> (*amplitudeCallback)(
		const Index &bra,
		const Index &ket
	),
	const Index &braIndex,
	const Index &ketIndex
) :
	braIndex(braIndex),
	ketIndex(ketIndex)
{
	this->amplitudeCallback = amplitudeCallback;
};

OverlapAmplitude::OverlapAmplitude(
	const string &serialization,
	Serializable::Mode mode
){
	TBTKAssert(
		Serializable::validate(
			serialization,
			"OverlapAmplitude",
			mode
		),
		"OverlapAmplitude::OverlapAmplitude()",
		"Unable to parse string as OverlapAmplitude '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
	{
		try{
			amplitudeCallback = nullptr;

			nlohmann::json j = nlohmann::json::parse(serialization);
			amplitude = Serializable::deserialize<complex<double>>(
				j["amplitude"].get<string>(),
				mode
			);
			braIndex = Index(j["braIndex"].dump(), mode);
			ketIndex = Index(j["ketIndex"].dump(), mode);
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"OverlapAmplitude::OverlapAmplitude()",
				"Unable to parse string as OverlapAmplitude '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"OverlapAmplitude::OverlapAmplitude()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

string OverlapAmplitude::serialize(Serializable::Mode mode) const{
	TBTKAssert(
		amplitudeCallback == nullptr,
		"OverlapAmplitude::serialize()",
		"Unable to serialize OverlapAmplitude that uses callback"
		<< " value.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "OverlapAmplitude";
		j["amplitude"] = Serializable::serialize(amplitude, mode);
		j["braIndex"] = nlohmann::json::parse(braIndex.serialize(mode));
		j["ketIndex"] = nlohmann::json::parse(ketIndex.serialize(mode));

		return j.dump();
	}
	default:
		TBTKExit(
			"OverlapAmplitude::serialize()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
