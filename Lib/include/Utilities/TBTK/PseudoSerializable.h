/* Copyright 2019 Kristofer Björnson
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
 *  @file PseudoSerializable.h
 *  @brief Base class for psudo-serializable objects.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PSEUDO_SERIALIZABLE
#define COM_DAFER45_TBTK_PSEUDO_SERIALIZABLE

#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{

class PseudoSerializable{
public:
	/** Serialize object. */
	std::string serialize(Serializable::Mode mode);
};

inline std::string PseudoSerializable::serialize(Serializable::Mode mode){
	TBTKExit(
		"PseudoSerializable::serialize()",
		"A class inheriting from PseudoSerializable does not implement"
		<< "the function serialize().",
		"Make sure all classes inheriting from PseudoSerializable"
		<< " implements a function with the signature 'std::string"
		<< " serializable(Serializable::Mode mode)'. If you do not"
		<< " inherit from PseudoSerializable, or all such classes"
		<< " indeed do implement the given function, this may be an"
		<< " error in TBTK. If you susceptct the later is the case,"
		<< " contact the developer."
	);
}

};	//End namespace TBTK

#endif
