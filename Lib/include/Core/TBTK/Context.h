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
 *  @file Context.h
 *  @brief Application context.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CONTEXT
#define COM_DAFER45_TBTK_CONTEXT

#include "TBTK/PersistentObject.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{

/** @brief Application context. */
class Context{
public:
	/** Create a new object and add it to the Context.
	 *
	 *  @param name The name of the object.
	 *
	 *  @return A reference to the newly created object. */
	template<typename DataType>
	DataType& create(const std::string &name);

	/** Get object from the Context.
	 *
	 *  @param name The name of the object.
	 *
	 *  @return The object with the given name. */
	template<typename DataType>
	DataType& get(const std::string &name);

	/** Get the Context.
	 *
	 *  @return The Context Singleton. */
	static Context& getContext();
private:
	/** Key-value list of objects contained in the Context. */
	std::map<std::string, PersistentObject*> objects;

	/** Constructor. */
	Context();

	/** Prevent copy construction. */
	Context(const Context &context) = delete;

	/** Prevent copy through assignment. */
	Context& operator=(const Context &rhs) = delete;

	/** Destructor. */
	~Context();
};

template<typename DataType>
DataType& Context::create(const std::string &name){
	static_assert(
		std::is_base_of<PersistentObject, DataType>::value,
		"Unable to create object that is not derived from"
		" PersistentObject."
	);
	TBTKAssert(
		objects.count(name) == 0,
		"Context::create()",
		"Unable to create object with name '" << name << "'"
		<< " since and object with the same name already"
		<< " exists in this Context.",
		""
	);
	objects[name] = new DataType();

	return *dynamic_cast<DataType*>(objects[name]);
}

template<typename DataType>
DataType& Context::get(const std::string &name){
	static_assert(
		std::is_base_of<PersistentObject, DataType>::value,
		"Unable to get object that is not derived from"
		" PersistentObject."
	);
	try{
		PersistentObject* object = objects.at(name);
		DataType *castObject = dynamic_cast<DataType*>(object);
		TBTKAssert(
			castObject != nullptr,
			"Context::get()",
			"Invalid type. '" << name << "' cannot be cast"
			<< " to '"
			<< object->getDynamicTypeInformation(
			).getName() << "'.",
			""
		);
		return *castObject;
	}
	catch(...){
		TBTKExit(
			"Context::get()",
			"No object with the name '" << name << "'"
			<< " exists in the Context.",
			""
		);
	}
}

};	//End of namespace TBTK

#endif
