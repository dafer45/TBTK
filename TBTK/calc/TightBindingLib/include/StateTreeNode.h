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

/** @package TBTKcalc
 *  @file StateTreeNode.h
 *  @brief Tree structure for quick access of multiple States.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STATE_TREE_NODE
#define COM_DAFER45_TBTK_STATE_TREE_NODE

#include "AbstractState.h"

namespace TBTK{

class StateTreeNode{
public:
	/** Constructor. */
	StateTreeNode();

	/** Destructor. */
	~StateTreeNode();

	/** Add state. Does not assume ownership of the state. */
	void add(AbstractState *state);

	/** Get center coorindates. */
	const std::vector<double>& getCoordinates() const;

	/** Get radius. */
	double getRadius() const;
private:
	/** Child nodes. */
	StateTreeNode *treeNodes[2];

	AbstractState *state;

	/** Ceneter of container. */
	std::vector<double> center;

	/** Radius of container. */
	double radius;

	/** Add state. Is called by StateTreeNode::add() and is called
	 *  recursively. */
	AbstractState* addRecursive(AbstractState* state);
};

};	//End namespace TBTK

#endif
