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

/** @file StateTreeNode.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/StateTreeNode.h"
#include "../include/TBTKMacros.h"
#include "../include/Index.h"

#include <limits>

using namespace std;

namespace TBTK{

StateTreeNode::StateTreeNode(initializer_list<double> center, double halfSize, int maxDepth) :
	numSpacePartitions(pow(2, center.size()))
{
	for(int n = 0; n < numSpacePartitions; n++)
		stateTreeNodes.push_back(NULL);

	for(unsigned int n = 0; n < center.size(); n++)
		this->center.push_back(*(center.begin() + n));

	this->halfSize = halfSize;
	this->maxDepth = maxDepth;
}

StateTreeNode::~StateTreeNode(){
}

void StateTreeNode::add(AbstractState *state){
	TBTKAssert(
		state->getCoordinates().size() == center.size(),
		"StateTreeNode::add()",
		"Incompatible dimenstions. The StateTreeNode has stores states"
		<< " with dimension '" << center.size() << ", but a state with"
		<< " dimension '" << state->getCoordinates().size() << " was"
		<< " encountered.",
		""
	);

	if(!addRecursive(state)){
		const vector<double> &stateCoordinates = state->getCoordinates();

		stringstream centerStr;
		centerStr << "{";
		for(unsigned int n = 0; n < center.size(); n++){
			if(n != 0)
				centerStr << ", ";
			centerStr << center.at(n);
		}
		centerStr << "}";

		stringstream stateStr;
		stateStr << "{";
		for(unsigned int n = 0; n < stateCoordinates.size(); n++){
			if(n != 0)
				stateStr << ", ";
			stateStr << stateCoordinates.at(n);
		}
		stateStr << "}";

		TBTKExit(
			"StateTreeNode::add()",
			"Unable to add state to state tree. The StateTreeNode"
			<< " center is '" << centerStr.str() << "' and the"
			<< " half size is '" << halfSize << "'. Tried to add"
			<< " State with coordinate '" << stateStr.str()
			<< "' and extent '" << state->getExtent() << "'.",
			"Make sure the StateTreeNode is large enough to"
			<< " contain every state with finite extent."
		);
	}
}

AbstractState* StateTreeNode::addRecursive(AbstractState *state){
	TBTKNotYetImplemented("StateTreeNode::addRecursive()");

/*	if(!state->hasFiniteExtent()){
		states.push_back(state);

		return true;
	}

	vector<double> relativeCoordinate;
	for(int 
	if(state->getExtent() + > halfSize*/

	return NULL;
}

};	//End of namespace TBTK
