/* Copyright 2016 Kristofer Björnson
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

/** @file AbstractState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/AbstractState.h"

using namespace std;

namespace TBTK{

//AbstractState::AbstractState(StateID stateID) : index({}), container({}){
AbstractState::AbstractState(StateID stateID) : index(), container(){
	this->stateID = stateID;

	if(numeric_limits<double>::has_infinity)
		extent = numeric_limits<double>::infinity();
	else
		extent = numeric_limits<double>::max();
}

AbstractState::~AbstractState(){
}

void AbstractState::setCoordinates(initializer_list<double> coordinates){
	this->coordinates.clear();
	for(unsigned int n = 0; n < coordinates.size(); n++)
		this->coordinates.push_back(*(coordinates.begin()+n));
}

void AbstractState::setCoordinates(const vector<double> &coordinates){
	this->coordinates.clear();
	for(unsigned int n = 0; n < coordinates.size(); n++)
		this->coordinates.push_back(*(coordinates.begin()+n));
}

void AbstractState::setSpecifiers(initializer_list<int> specifiers){
	this->specifiers.clear();
	for(unsigned int n = 0; n < specifiers.size(); n++)
		this->specifiers.push_back(*(specifiers.begin()+n));
}

void AbstractState::setSpecifiers(const vector<int> &specifiers){
	this->specifiers.clear();
	for(unsigned int n = 0; n < specifiers.size(); n++)
		this->specifiers.push_back(*(specifiers.begin()+n));
}

std::string AbstractState::serialize(Mode mode) const{
	TBTKNotYetImplemented("AbstractState::serialize()");
}

};	//End of namespace TBTK
