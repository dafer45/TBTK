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

using namespace std;

namespace TBTK{

StateTreeNode::StateTreeNode(){
	treeNodes[0] = NULL;
	treeNodes[1] = NULL;
	radius = 0.;
}

StateTreeNode::~StateTreeNode(){
}

void StateTreeNode::add(AbstractState *state){
	while((state = addRecursive(state)));
}

AbstractState* StateTreeNode::addRecursive(AbstractState *state){
	TBTKNotYetImplemented("StateTreeNode::addRecursive()");

	TBTKAssert(
		(treeNodes[0] == NULL && treeNodes[1] == NULL)
		|| (treeNodes[0] != NULL && treeNodes[1] != NULL),
		"StateTreeNode::addRecursive()",
		"Fatal bug. This should never have happened.",
		"Notify the developer that one and only one of treeNodes[0]"
		<< " and treeNodes[1] is NULL."
	);

	if(treeNodes[0] == NULL){
		if(this->state == NULL){
			this->state = state;
			radius = 0;
			center = state->getCoordinates();

			return NULL;
		}
		else{
			treeNodes[0] = new StateTreeNode();
			treeNodes[0]->add(this->state);
			treeNodes[1] = new StateTreeNode();
			treeNodes[1]->add(state);

			const vector<double> &coordinates1 = this->state->getCoordinates();
			const vector<double> &coordinates2 = state->getCoordinates();
			TBTKAssert(
				coordinates1.size() == coordinates2.size(),
				"StateTreeNode::addRecursive()",
				"Incompatible coordinate dimensions."
				<< " Already stored state has "
				<< coordinates1.size()
				<< " components, while state added has "
				<< coordinates2.size()
				<< " components.",
				""
			);

			center.clear();
			radius = 0.;
			for(unsigned int n = 0; n < coordinates1.size(); n++){
				center.push_back((coordinates1.at(n) + coordinates2.at(n))/2.);
				double difference = coordinates1.at(n) - coordinates2.at(n);
				radius += difference*difference;
			}
			radius = sqrt(radius);

			this->state = NULL;

			return NULL;
		}
	}

	const vector<double> *coordinates[2];
	coordinates[0] = &treeNodes[0]->getCoordinates();
	coordinates[1] = &treeNodes[1]->getCoordinates();
	TBTKAssert(
		coordinates[0]->size() == state->getCoordinates().size(),
		"StateTreeNode::addRecursive()",
		"Incompatible coordinate dimensions. Already stored states"
		<< " has " << coordinates[0]->size() << " components, while"
		<< " state added has " << coordinates[1]->size()
		<< " components.",
		""
	);

	vector<double> difference[2];
	for(unsigned int n = 0; n < state->getCoordinates().size(); n++){
		for(int c = 0; c < 2; c++)
			difference[c].push_back(state->getCoordinates().at(n) - coordinates[c]->at(n));
	}
	double differenceAbs[2];
	for(int c = 0; c < 2; c++)
		differenceAbs[c] = 0.;
	for(int c = 0; c < 2; c++)
		for(unsigned int n = 0; n < difference[c].size(); n++)
			differenceAbs[c] += difference[c].at(n)*difference[c].at(n);
	for(unsigned int c = 0; c < 2; c++)
		differenceAbs[c] = sqrt(differenceAbs[c]);

/*	if(differenceAbs[0] < differenceAbs[1]){
		if((state = treeNodes[0].addRecursive(state)){
		
		}

		return NULL;
	}*/
}

};	//End of namespace TBTK
