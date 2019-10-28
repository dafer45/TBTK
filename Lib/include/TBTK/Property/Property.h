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
 *  @file Poperty.h
 *  @brief Property class.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_PROPERTY
#define COM_DAFER45_TBTK_PROPERTY_PROPERTY

#include "TBTK/json.hpp"

namespace TBTK{
namespace Property{

/** @brief Abstract Property class.
 *
 *  Base class for AbstractProperty that allows for any Property to be
 *  referenced using the same base class. See AbstractProperty for more details
 *  about @link Property Properties@endlink. */
class Property{
public:
protected:
	/** Constructs an uninitialized AbstractProperty. */
	Property();

	/** Destructor. */
	virtual ~Property();
};

};	//End namespace Property
};	//End namespace TBTK

#endif
