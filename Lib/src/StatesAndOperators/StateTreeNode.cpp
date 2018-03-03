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

#include "TBTK/Index.h"
#include "TBTK/StateTreeNode.h"
#include "TBTK/TBTKMacros.h"

#include <limits>

using namespace std;

namespace TBTK{

StateTreeNode::StateTreeNode(
	initializer_list<double> center,
	double halfSize,
	int maxDepth
) :
	numSpacePartitions(pow(2, center.size()))
{
	for(unsigned int n = 0; n < center.size(); n++)
		this->center.push_back(*(center.begin() + n));

	this->halfSize = halfSize;
	this->maxDepth = maxDepth;
}

StateTreeNode::StateTreeNode(
	vector<double> center,
	double halfSize,
	int maxDepth
) :
	numSpacePartitions(pow(2, center.size()))
{
	for(unsigned int n = 0; n < center.size(); n++)
		this->center.push_back(*(center.begin() + n));

	this->halfSize = halfSize;
	this->maxDepth = maxDepth;
}

StateTreeNode::StateTreeNode(
	const StateSet &stateSet,
	int maxDepth,
	double centerShiftMultiplier
) :
	numSpacePartitions(pow(2, stateSet.getStates().at(0)->getCoordinates().size()))
{
	const vector<AbstractState*> &states = stateSet.getStates();
	unsigned int numCoordinates = states.at(0)->getCoordinates().size();

	for(unsigned int n = 1; n < states.size(); n++){
		TBTKAssert(
			numCoordinates == states.at(n)->getCoordinates().size(),
			"StateTreeNode::StateTreeNode()",
			"Unable to handle StateSets containing states with different dimensions.",
			""
		);
	}

	//Find bounding box.
	vector<double> min;
	vector<double> max;
	double maxFiniteExtent = 0.;
	unsigned int n = 0;
	//Find first state with finite extent and initialize min and max such
	//that the state is contained inside the corresponding rectangle.
	for(; n < states.size(); n++){
		if(states.at(n)->hasFiniteExtent()){
			for(unsigned int c = 0; c < numCoordinates; c++){
				min.push_back(states.at(n)->getCoordinates().at(c) - states.at(n)->getExtent());
				max.push_back(states.at(n)->getCoordinates().at(c) + states.at(n)->getExtent());
			}
			maxFiniteExtent = states.at(n)->getExtent();
			break;
		}
	}
	//If all state have infinite extent, create a dummy box of zero extent.
	if(n == states.size()){
		for(unsigned int c = 0; c < states.at(0)->getCoordinates().size(); c++){
			min.push_back(0);
			max.push_back(0);
		}
	}
	//Loop over all remaining states and update min and max so that every
	//state is contained in the corresponding rectangle.
	for(; n < states.size(); n++){
		if(!states.at(n)->hasFiniteExtent())
			continue;

		for(unsigned int c = 0; c < numCoordinates; c++){
			if(min.at(c) > states.at(n)->getCoordinates().at(c) - states.at(n)->getExtent())
				min.at(c) = states.at(n)->getCoordinates().at(c) - states.at(n)->getExtent();
			if(max.at(c) < states.at(n)->getCoordinates().at(c) + states.at(n)->getExtent())
				max.at(c) = states.at(n)->getCoordinates().at(c) + states.at(n)->getExtent();
		}

		if(maxFiniteExtent < states.at(n)->getExtent())
			maxFiniteExtent = states.at(n)->getExtent();
	}

	//Enlarge and shift box to avoid problems with high level partition
	//boundaries cutting through states. This would be particularly
	//problematic if the states for example have three dimensional
	//coordinates, but are contained in a plane. Without shifting the box,
	//the first partition boundary would cut every state and thereby every
	//state would be added to the root node.
	for(unsigned int n = 0; n < max.size(); n++)
		max.at(n) += centerShiftMultiplier*maxFiniteExtent;
	for(unsigned int n = 0; n < min.size(); n++)
		min.at(n) -= 2.*(max.at(n) - min.at(n))*(1 - ROUNDOFF_MARGIN_MULTIPLIER);

	//Calculate center nad halfSize of the bounding box.
	halfSize = 0.;
	for(unsigned int n = 0; n < numCoordinates; n++){
		center.push_back((min.at(n) + max.at(n))/2.);
		if(halfSize < (max.at(n) - min.at(n))/2.)
			halfSize = (max.at(n) - min.at(n))/2.;
	}

	this->maxDepth = maxDepth;

	for(unsigned int n = 0; n < states.size(); n++)
		add(states.at(n));
}

StateTreeNode::~StateTreeNode(){
	for(unsigned int n = 0; n < stateTreeNodes.size(); n++)
		delete stateTreeNodes[n];
}

void StateTreeNode::add(AbstractState *state){
	TBTKAssert(
		state->getCoordinates().size() == center.size(),
		"StateTreeNode::add()",
		"Incompatible dimenstions. The StateTreeNode has stores states"
		<< " with dimension '" << center.size() << "', but a state"
		<< " with dimension '" << state->getCoordinates().size() << "'"
		<< " was encountered.",
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
		if(largestRelativeCoordinate < abs(relativeCoordinates.at(n)))
			largestRelativeCoordinate = abs(relativeCoordinates.at(n));
	}

	//If the largest relative coordinate plus the states extent is larger
	//than the partitions half size, the state is not fully contained in
	//the partition. Therefore return false to indicate that the state
	//cannot be added to this partition. The actual comparission is made to
	//a slightly region in order to ensure that rounding error don't allow
	//the state to be added here, but fail to be extracted when a similar
	//check is made.
	if(largestRelativeCoordinate + state->getExtent() > halfSize*ROUNDOFF_MARGIN_MULTIPLIER)
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
			for(unsigned int c = 0; c < center.size(); c++)
				subCenter.push_back(center.at(c) + ((n/(1 << c))%2 - 1/2.)*halfSize);

			stateTreeNodes.push_back(new StateTreeNode(subCenter, halfSize/2., maxDepth-1));
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

	return true;
}

vector<const AbstractState*>* StateTreeNode::getOverlappingStates(
	initializer_list<double> coordinates,
	double extent
) const{
	TBTKAssert(
		coordinates.size() == center.size(),
		"StateTreeNode::getOverlappingStates",
		"Incompatible dimenstions. The StateTreeNode stores states"
		<< " with dimension '" << center.size() << "', but the"
		<< " argument 'coordinates' has dimension '"
		<< coordinates.size() << "'.",
		""
	);

	vector<double> coordinatesVector;
	for(unsigned int n = 0; n < coordinates.size(); n++)
		coordinatesVector.push_back(*(coordinates.begin() + n));

	vector<const AbstractState*> *overlappingStates = new vector<const AbstractState*>();

	getOverlappingStatesRecursive(
		overlappingStates,
		coordinatesVector,
		extent
	);

	return overlappingStates;
}

vector<const AbstractState*>* StateTreeNode::getOverlappingStates(
	vector<double> coordinates,
	double extent
) const{
	TBTKAssert(
		coordinates.size() == center.size(),
		"StateTreeNode::getOverlappingStates",
		"Incompatible dimenstions. The StateTreeNode stores states"
		<< " with dimension '" << center.size() << ", but the argument"
		<< " 'coordinates' has dimension '" << coordinates.size()
		<< " was encountered.",
		""
	);

	vector<const AbstractState*> *overlappingStates = new vector<const AbstractState*>();

	getOverlappingStatesRecursive(overlappingStates, coordinates, extent);

	return overlappingStates;
}

void StateTreeNode::getOverlappingStatesRecursive(
	vector<const AbstractState*> *overlappingStates,
	vector<double> coordinates,
	double extent
) const{
	//Get distance from the center of the current space partition
	double distance = 0.;
	for(unsigned int n = 0; n < center.size(); n++)
		distance += pow(coordinates.at(n) - center.at(n), 2);
	distance = sqrt(distance);

	//If the distance from the center is larger than the half diagonal of
	//the partition plus the extent, the state is not overlapping with this
	//partition. The half diagonal is given by
	//sqrt(spaceDimension*halfSize^2)
	if(distance > sqrt(center.size()*pow(halfSize, 2)) + extent)
		return;

	//Add states on this node if they overlap with the sphere centered at
	//'coordinates' and with radius 'extent'.
	for(unsigned int n = 0; n < states.size(); n++){
		const vector<double> &stateCoordinates = states.at(n)->getCoordinates();

		double distance = 0.;
		for(unsigned int n = 0; n < coordinates.size(); n++)
			distance += pow(coordinates.at(n) - stateCoordinates.at(n), 2);
		distance = sqrt(distance);

		if(distance <= extent + states.at(n)->getExtent())
			overlappingStates->push_back(states.at(n));
	}

	//Add relevant states from child nodes.
	for(unsigned int n = 0; n < stateTreeNodes.size(); n++){
		stateTreeNodes.at(n)->getOverlappingStatesRecursive(
			overlappingStates,
			coordinates,
			extent
		);
	}
}

};	//End of namespace TBTK
