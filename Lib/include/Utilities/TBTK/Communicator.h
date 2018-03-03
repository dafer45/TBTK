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

/** @package TBTKcalc
 *  @file Communicator.h
 *  @brief Base class that communicate their current status during execution.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_COMMUNICATOR
#define COM_DAFER45_TBTK_COMMUNICATOR

namespace TBTK{

class Communicator{
public:
	/** Constructor. */
	Communicator(bool verbose);

	/** Set verbose. */
	void setVerbose(bool verbose);

	/** Get verbose. */
	bool getVerbose() const;

	/** Set global verbose. */
	static void setGlobalVerbose(bool globalVerbose);

	/** Get global verbose. */
	static bool getGlobalVerbose();
private:
	/** Flag indicating wether the communicator is verbose. */
	bool verbose;

	/** Global flag indicating verbosity. */
	static bool globalVerbose;
};

inline void Communicator::setVerbose(bool verbose){
	this->verbose = verbose;
}

inline bool Communicator::getVerbose() const{
	return verbose;
}

inline void Communicator::setGlobalVerbose(bool globalVerbose){
	Communicator::globalVerbose = globalVerbose;
}

inline bool Communicator::getGlobalVerbose(){
	return globalVerbose;
}

};	//End namespace TBTK

#endif
