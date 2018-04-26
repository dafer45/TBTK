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

/** @file WaveFunctions.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/WaveFunctions.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{
namespace Property{

/*WaveFunction::WaveFunction(
	int dimensions,
	const int *ranges
) :
	AbstractProperty(dimensions, ranges, 1)
{
}

WaveFunction::WaveFunction(
	int dimensions,
	const int *ranges,
	const double *data
) :
	AbstractProperty(dimensions, ranges, 1, data)
{
}*/

/*WaveFunctions::WaveFunctions(
	const IndexTree &indexTree,
	const initializer_list<unsigned int> &states
) :
	AbstractProperty(indexTree, states.size())
{
	for(unsigned int n = 0; n < states.size(); n++)
		this->states.push_back(*(states.begin() + n));

	isContinuous = true;
	for(unsigned int n = 1; n < this->states.size(); n++){
		if(this->states.at(n) != this->states.at(n-1)+1){
			isContinuous = false;
			break;
		}
	}
}*/

WaveFunctions::WaveFunctions(
	const IndexTree &indexTree,
	const vector<unsigned int> &states
) :
	AbstractProperty(indexTree, states.size())
{
	this->states = states;

	isContinuous = true;
	for(unsigned int n = 1; n < this->states.size(); n++){
		if(this->states.at(n) != this->states.at(n-1)+1){
			isContinuous = false;
			break;
		}
	}
}

/*WaveFunctions::WaveFunctions(
	const IndexTree &indexTree,
	const initializer_list<unsigned int> &states,
	const complex<double> *data
) :
	AbstractProperty(indexTree, states.size(), data)
{
	for(unsigned int n = 0; n < states.size(); n++)
		this->states.push_back(*(states.begin() + n));

	isContinuous = true;
	for(unsigned int n = 1; n < this->states.size(); n++){
		if(this->states.at(n) != this->states.at(n-1)+1){
			isContinuous = false;
			break;
		}
	}
}*/

WaveFunctions::WaveFunctions(
	const IndexTree &indexTree,
	const vector<unsigned int> &states,
	const complex<double> *data
) :
	AbstractProperty(indexTree, states.size(), data)
{
	this->states = states;

	isContinuous = true;
	for(unsigned int n = 1; n < this->states.size(); n++){
		if(this->states.at(n) != this->states.at(n-1)+1){
			isContinuous = false;
			break;
		}
	}
}

/*WaveFunctions::WaveFunctions(
	const WaveFunctions &waveFunctions
) :
	AbstractProperty(waveFunctions),
	states(waveFunctions.states)
{
	this->isContinuous = waveFunctions.isContinuous;
}

WaveFunctions::WaveFunctions(
	WaveFunctions &&waveFunctions
) :
	AbstractProperty(std::move(waveFunctions)),
	states(std::move(waveFunctions.states))
{
	this->isContinuous = waveFunctions.isContinuous;
}*/

WaveFunctions::WaveFunctions(
	const string &serialization,
	Mode mode
) :
	AbstractProperty(
		Serializable::extract(
			serialization,
			mode,
			"abstractProperty"
		),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "WaveFunctions", mode),
		"WaveFunctions::WaveFunctions()",
		"Unable to parse string as WaveFunctions '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			json j = json::parse(serialization);
			isContinuous = j.at("isContinuous").get<bool>();
			json s = j.at("states");
			for(json::iterator it = s.begin(); it < s.end(); ++it)
				states.push_back(*it);
		}
		catch(json::exception e){
			TBTKExit(
				"WaveFunctions::WaveFuntions()",
				"Unable to parse string as WaveFunctions '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"WaveFunctions::WaveFunctions()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

/*WaveFunctions::~WaveFunctions(){
}

WaveFunctions& WaveFunctions::operator=(const WaveFunctions &rhs){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);
		this->isContinuous = rhs.isContinuous;
		states = rhs.states;
	}

	return *this;
}

WaveFunctions& WaveFunctions::operator=(WaveFunctions &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));
		isContinuous = rhs.isContinuous;
		states = std::move(rhs.states);
	}

	return *this;
}*/

const complex<double>& WaveFunctions::operator()(
        const Index &index,
        unsigned int state
) const{
	if(isContinuous){
		int n = state - states.at(0);
		TBTKAssert(
			n >= 0 && (unsigned int)n < states.size(),
			"WaveFunctions::operator()",
			"WaveFunctions does not contain state '" << state << "'.",
			""
		);
		return AbstractProperty::operator()(index, n);
	}
	else{
		for(unsigned int n = 0; n < states.size(); n++){
			if(state == states.at(n))
				return AbstractProperty::operator()(index, n);
		}
		TBTKExit(
			"WaveFunctions::operator()",
			"WaveFunctions does not contain state '" << state
			<< "'.",
			""
		);
	}
}

complex<double>& WaveFunctions::operator()(
        const Index &index,
        unsigned int state
){
	if(isContinuous){
		int n = state - states.at(0);
		TBTKAssert(
			n >= 0 && (unsigned int)n < states.size(),
			"WaveFunctions::operator()",
			"WaveFunctions does not contain state '" << state
			<< "'.",
			""
		);
		return AbstractProperty::operator()(index, n);
	}
	else{
		for(unsigned int n = 0; n < states.size(); n++){
			if(state == states.at(n))
				return AbstractProperty::operator()(index, n);
		}
		TBTKExit(
			"WaveFunctions::operator()",
			"WaveFunctions does not contain state '" << state
			<< "'.",
			""
		);
	}
}

double WaveFunctions::getMinAbs() const{
	const complex<double> *data = getData();
	double min = abs(data[0]);
	for(unsigned int n = 1; n < getSize(); n++)
		if(abs(data[n]) < min)
			min = abs(data[n]);

	return min;
}

double WaveFunctions::getMaxAbs() const{
	const complex<double> *data = getData();
	double max = abs(data[0]);
	for(unsigned int n = 1; n < getSize(); n++)
		if(abs(data[n]) > max)
			max = abs(data[n]);

	return max;
}

double WaveFunctions::getMinArg() const{
	const complex<double> *data = getData();
	double min = arg(data[0]);
	for(unsigned int n = 1; n < getSize(); n++)
		if(arg(data[n]) < min)
			min = arg(data[n]);

	return min;
}

double WaveFunctions::getMaxArg() const{
	const complex<double> *data = getData();
	double max = arg(data[0]);
	for(unsigned int n = 1; n < getSize(); n++)
		if(arg(data[n]) > max)
			max = arg(data[n]);

	return max;
}

string WaveFunctions::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		json j;
		j["id"] = "WaveFunctions";
		j["isContinuous"] = isContinuous;
		for(unsigned int n = 0; n < states.size(); n++)
			j["states"].push_back(states.at(n));
		j["abstractProperty"] = json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"WaveFunctions::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
