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

/** @file WaveFunction.cpp
 *
 *  @author Kristofer Björnson
 */

#include "WaveFunction.h"

using namespace std;

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

WaveFunction::WaveFunction(
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
}

WaveFunction::WaveFunction(
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

WaveFunction::WaveFunction(
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
}

WaveFunction::WaveFunction(
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

WaveFunction::WaveFunction(
	const WaveFunction &waveFunction
) :
	AbstractProperty(waveFunction),
	states(waveFunction.states)
{
	this->isContinuous = waveFunction.isContinuous;
}

WaveFunction::WaveFunction(
	WaveFunction &&waveFunction
) :
	AbstractProperty(std::move(waveFunction)),
	states(std::move(waveFunction.states))
{
	this->isContinuous = waveFunction.isContinuous;
}

WaveFunction::~WaveFunction(){
}

WaveFunction& WaveFunction::operator=(const WaveFunction &rhs){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);
		this->isContinuous = rhs.isContinuous;
		states = rhs.states;
	}

	return *this;
}

WaveFunction& WaveFunction::operator=(WaveFunction &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));
		isContinuous = rhs.isContinuous;
		states = std::move(rhs.states);
	}

	return *this;
}

complex<double> WaveFunction::operator()(
        const Index &index,
        unsigned int state
) const{
	if(isContinuous){
		int n = state - states.at(0);
		TBTKAssert(
			n >= 0 && (unsigned int)n < states.size(),
			"WaveFunction::operator()",
			"WaveFunction does not contain state '" << state << "'.",
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
			"WaveFunction::operator()",
			"WaveFunction does not contain state '" << state << "'.",
			""
		);
	}
}


};	//End of namespace Property
};	//End of namespace TBTK
