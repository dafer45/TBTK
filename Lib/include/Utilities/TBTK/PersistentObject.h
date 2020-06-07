/* Copyright 2020 Kristofer Björnson
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
 *  @file PersistentObject.h
 *  @brief Base class for object that can be added to a Context.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PERSISTENT_OBJECT
#define COM_DAFER45_TBTK_PERSISTENT_OBJECT

#include "TBTK/DynamicTypeInformation.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{

/** @brief Base class for objects that can be added to a Context. */
class PersistentObject : public Serializable{
	TBTK_DYNAMIC_TYPE_INFORMATION(PersistentObject)
public:
	/** Constructor. */
	PersistentObject();

	/** Copy constructor.
	 *
	 *  @param persistentObject The PersistentObject to copy. */
	PersistentObject(const PersistentObject &persisntentObject);

	/** Move constructor.
	 *
	 *  @param persistentObject The PersistentObject to move. */
	PersistentObject(const PersistentObject &&persistentObject);

	/** Destructor. */
	virtual ~PersistentObject() = 0;

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression. */
	PersistentObject& operator=(const PersistentObject &rhs);

	/** Move assignment operator.
	 *
	 *  @param rhs The right hand side of the expression. */
	PersistentObject& operator=(const PersistentObject &&rhs);
private:
};

};	//End of namespace TBTK

#endif
