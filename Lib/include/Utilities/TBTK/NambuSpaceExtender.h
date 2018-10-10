/* Copyright 2018 Kristofer Björnson
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
 *  @file NambuSpaceExtender.h
 *  @brief Extends a Model to Nambu space.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_NAMBU_SPACE_EXTENDER
#define COM_DAFER45_TBTK_NAMBU_SPACE_EXTENDER

#include "TBTK/Model.h"

namespace TBTK{

class NambuSpaceExtender{
public:
	/** Enum class used to indicate the type of the extension to be
	 *  performed. Real space extensions can be made by simply taking the
	 *  negative transpose of the original Model as hole part, but the
	 *  momentum space requires handling the inversion of the momentum
	 *  vector. */
	enum class Mode {RealSpace, MomentumSpace};

	/** Creates a new Model that is the extension of the given Model to
	 *  Nambu space.
	 *
	 *  @param model The Model to be extended.
	 *
	 *  @return The extended Model.*/
	static Model extend(const Model &model, Mode mode);
};

}; //End of namesapce TBTK

#endif
