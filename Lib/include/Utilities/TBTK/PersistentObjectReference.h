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
 *  @file PersistentObjectReference.h
 *  @brief Reference to PersistentObject.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PERSISTENT_OBJECT_REFERENCE
#define COM_DAFER45_TBTK_PERSISTENT_OBJECT_REFERENCE

#include "TBTK/PersistentObject.h"

namespace TBTK{

/** @brief Reference to PersistentObject. */
template<typename DataType>
class PersistentObjectReference{
public:
	/** Constructor. */
	PersistentObjectReference();

	/** Set the object to refererence.
	 *
	 *  @param data The object to reference. */
	void set(DataType &data);

	/** Get the referenced object. The function takes a template parameter
	 *  that allows the object to be cast to any compatible type. If the
	 *  cast is not valid, an error is generated.
	 *
	 *  @return The referenced object, dynamically casted to the CastType.
	 */
	template<typename CastType>
	CastType& get();
private:
	DataType *data;
};

template<typename DataType>
PersistentObjectReference<DataType>::PersistentObjectReference(){
	data = nullptr;
}

template<typename DataType>
void PersistentObjectReference<DataType>::set(DataType &data){
	this->data = &data;
}

template<typename DataType>
template<typename CastType>
CastType& PersistentObjectReference<DataType>::get(){
	TBTKAssert(
		data != nullptr,
		"PersistentObjectReference::get()",
		"The PersistentObjectReference is empty.",
		""
	);
	CastType *castData = dynamic_cast<CastType*>(data);
	TBTKAssert(
		castData != nullptr,
		"PersistentObjectReference::get()",
		"Unable to cast data tp CastType.",
		""
	);
	return *castData;
}

};	//End of namespace TBTK

#endif
