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
	for(unsigned int n = 0; n < center.size(); n++)
		this->center.push_back(*(center.begin() + n));

	this->halfSize = halfSize;
	this->maxDepth = maxDepth;
}

StateTreeNode::StateTreeNode(vector<double> center, double halfSize, int maxDepth) :
	numSpacePartitions(pow(2, center.size()))
{
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

bool StateTreeNode::addRecursive(AbstractState *state){
//	TBTKNotYetImplemented("StateTreeNode::addRecursive()");

	//Add the state as high up in the tree structure as possible if it is a
	//non-local state.
	if(!state->hasFiniteExtent()){
		states.push_back(state);

		return true;
	}

	//Get coordinate of the state relative to the center of the current
	//space partition.
	vector<double> relativeCoordinates;
	const vector<double> &stateCoordinates = state->getCoordinates();
	for(unsigned int n = 0; n < center.size(); n++)
		relativeCoordinates.push_back(stateCoordinates.at(n) - center.at(n));

	//Find the largest relative coordinate.
	double largestRelativeCoordinate = 0.;
	for(unsigned int n = 0; n < relativeCoordinates.size(); n++){
		if(largestRelativeCoordinate < relativeCoordinates.at(n))
			largestRelativeCoordinate = relativeCoordinates.at(n);
	}

	//If the largest relative coordinate plus the states extent is larger
	//than the partitions half size, the state is not fully contained in
	//the partition. Therefore return false to indicate that the state
	//cannot be added to this partition.
	if(largestRelativeCoordinate + state->getExtent() > halfSize)
		return false;

	//If the maximum number of allowed chilld node generations from this
	//node is zero, add the state to this node.
	if(maxDepth == 0){
		states.push_back(state);

		return true;
	}

	//Create child nodes if they do not already exist.
	if(stateTreeNodes.size() == 0){
		for(int n = 0; n < numSpacePartitions; n++){
			vector<double> subCenter;
			for(unsigned int c = 0; c < center.size(); c++){
				subCenter.push_back(center.at(c) + ((n/(1 << c))%2 - 1/2.)*halfSize/2.);
			}

			stateTreeNodes.push_back(new StateTreeNode(subCenter, halfSize/2, maxDepth-1));
		}
	}

	//Try to add the state to one of the child nodes.
	for(unsigned int n = 0; n < stateTreeNodes.size(); n++){
		if(stateTreeNodes.at(n)->addRecursive(state))
			return true;
	}

	//State was not added to any of the child nodes, so add it to this
	//node.
	states.push_back(state);

	return true;
}

};	//End of namespace TBTK
