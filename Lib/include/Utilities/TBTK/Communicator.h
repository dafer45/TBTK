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
 *  @brief Base class for classes that can communicate their status during
 *  execution.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_COMMUNICATOR
#define COM_DAFER45_TBTK_COMMUNICATOR

namespace TBTK{

/** @brief Base class for classes that can communicate their status during
 *  execution.
 *
 *  The communicator allows for both the global verbosity and the verbosity of
 *  individual objects to be modified. A class that inherits from Communicator
 *  is provided with two functions for setting and getting its verbosity.
 *  Setting the verbosity to true means the object may output text to the
 *  Streams during execution.
 *
 *  The Communicator class also provides two static functions for setting the
 *  verbosity globally. If the global verbosity set to false, no information
 *  should be written to the Streams even if the individual objects have their
 *  verbosity set to true. To respect this behavior, classes that derive from
 *  the communicator should enclose any output to Streams with the statement
 *  ```cpp
 *    if(getVerbose() && Streams::getGlobalVerbose()){
 *      //Output to Streams here
 *      //...
 *    }
 *  ```
 *
 *  *Note: Error messages should not be muted.*
 *
 *  # Example
 *  \snippet Utilities/Communicator.cpp Communicator
 *  ## Output
 *  \snippet output/Utilities/Communicator.output Communicator */
class Communicator{
public:
	//TBTKFeature Utilities.Communicator.construction.1 2019-11-01
	/** Constructor.
	 *
	 *  @param verbose Flag indicating whether or not the communicator is
	 *  verbose */
	Communicator(bool verbose);

	//TBTKFeature Utilities.Communicator.setGetVerbose.1 2019-11-01
	/** Set verbose.
	 *
	 *  @param verbose Flag indicating whether or not the Communicator is
	 *  verbose. */
	void setVerbose(bool verbose);

	//TBTKFeature Utilities.Communicator.setGetVerbose.1 2019-11-01
	/** Get verbose.
	 *
	 *  @return True if the Communicator is verbose, otherwise false. */
	bool getVerbose() const;

	//TBTKFeature Utilities.Communicator.setGetGlobalVerbose.1 2019-11-01
	/** Set global verbose.
	 *
	 *  @param globalVerbose If set to false, Communicators will not
	 *  communicate even if they are verbose. */
	static void setGlobalVerbose(bool globalVerbose);

	//TBTKFeature Utilities.Communicator.setGetGlobalVerbose.1 2019-11-01
	/** Get global verbose.
	 *
	 *  @return False if communication is suppressed globally. */
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
