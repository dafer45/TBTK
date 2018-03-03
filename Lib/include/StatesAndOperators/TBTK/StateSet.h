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
 *  @file StateSet.h
 *  @brief Container for States.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STATE_SET
#define COM_DAFER45_TBTK_STATE_SET

#include "TBTK/AbstractState.h"

namespace TBTK{

class StateSet{
public:
	/** Constructor.
	 *
	 *  @param isOwner Flag indicating whether the StateSet is considered
	 *  to be an owner of the states or not. If it is an owner, states
	 *  added to it will be delete when the StateSet is destructed. */
	StateSet(bool isOwner = true);

	/** Destructor. */
	~StateSet();

	/** Add state. Note: The StateSet assumes ownership of the pointers and
	 *  will delete them when it is destroyed. */
	void addState(AbstractState *state);

	/** Get states. */
	const std::vector<AbstractState*>& getStates() const;

	/** Get number of states. */
	unsigned int getNumStates() const;
private:
	/** Pointers to states. */
	std::vector<AbstractState*> states;

	/** Flag indicate whether the StateSet owns the states or not. */
	bool isOwner;
};

inline void StateSet::addState(AbstractState *state){
	states.push_back(state);
}

inline const std::vector<AbstractState*>& StateSet::getStates() const{
	return states;
}

inline unsigned int StateSet::getNumStates() const{
	return states.size();
}

};	//End of namespace TBTK

#endif
