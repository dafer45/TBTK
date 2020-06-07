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
#include "TBTK/Streamable.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <mutex>
#include <string>

namespace TBTK{

/** @brief Application context. */
class Context : public Streamable{
public:
	/** Create a new object and add it to the Context.
	 *
	 *  @param name The name of the object.
	 *
	 *  @return A reference to the newly created object. */
	template<typename DataType>
	DataType& create(const std::string &name);

	/** Erase an object from the Context.
	 *
	 *  @param name The name of the object to delete. */
	void erase(const std::string &name);

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

	/** Returns a string with characteristic information about the Context.
	 *
	 *  @return A string with characteristic information about the Context.
	 */
	virtual std::string toString() const;
private:
	/** Key-value list of objects contained in the Context. */
	std::map<std::string, PersistentObject*> objects;

	/** Kay-value list of all PersistentObjects that currently exists.
	 *  Every PersistentObject registers itself using
	 *  registerPersistentObject() in its constructor. */
	std::map<std::string, const PersistentObject*> allObjects;

	/** Mutex to lock the Context during insertion and removal of
	 *  PersistentObjects from objects. */
	mutable std::mutex mutexObjects;

	/** Mutex to lock the Context during insertion and removal of
	 *  PersistentObjects from allObjects. */
	mutable std::mutex mutexAllObjects;

	/** Constructor. */
	Context();

	/** Prevent copy construction. */
	Context(const Context &context) = delete;

	/** Prevent copy through assignment. */
	Context& operator=(const Context &rhs) = delete;

	/** Destructor. */
	~Context();

	/** Register a PersistentObject. */
	void registerPersistentObject(
		const PersistentObject &persistentObject
	);

	/** Deregister a PersistentObject. */
	void deregisterPersistentObject(
		const PersistentObject &persistentObject
	);

	/** PersistentObject is declare friend in order for it to be able to
	 *  register itself. */
	friend class PersistentObject;
};

template<typename DataType>
DataType& Context::create(const std::string &name){
	static_assert(
		std::is_base_of<PersistentObject, DataType>::value,
		"Unable to create object that is not derived from"
		" PersistentObject."
	);
	TBTKAssert(
		//This check is performed to provide a weak enforcement of
		//TBTK_DYNAMIC_TYPE_INFORMATION being declared in every
		//PersistentObject. A strong test that automatically tests this
		//on every deriving class would be preferable, but no such
		//solution is currently known. See also DevelopmentNotes/Notes.
		//This test ensures that every class that is ever used together
		//with the Context has declared TBTK_DYNAMIC_TYPE_INFORMATION
		//or will alert the user of it missing.
		DataType::getTypeidName() == typeid(DataType).name(),
		"Context::create()",
		"Missing dynamic type data.",
		"Make sure TBTK_DYNAMIC_TYPE_INFORMATION(DataType) is declared"
		<< " at the top of this class declaration."
	);
	mutexObjects.lock();
	if(objects.count(name) == 0){
		DataType *object = new DataType();
		objects[name] = object;
		mutexObjects.unlock();

		return *dynamic_cast<DataType*>(object);
	}
	else{
		mutexObjects.unlock();
		TBTKExit(
			"Context::create()",
			"Unable to create object with name '" << name << "'"
			<< " since and object with the same name already"
			<< " exists in this Context.",
			""
		);
	}
}

inline void Context::erase(const std::string &name){
	try{
		mutexObjects.lock();
		delete objects.at(name);
		objects.erase(name);
		mutexObjects.unlock();
	}
	catch(...){
		mutexObjects.unlock();
		TBTKExit(
			"Context::erase()",
			"No object with the name '" << name << "'"
			<< " exists in the Context.",
			""
		);
	}
}

template<typename DataType>
DataType& Context::get(const std::string &name){
	static_assert(
		std::is_base_of<PersistentObject, DataType>::value,
		"Unable to get object that is not derived from"
		" PersistentObject."
	);
	try{
		mutexObjects.lock();
		PersistentObject* object = objects.at(name);
		mutexObjects.unlock();
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
		mutexObjects.unlock();
		TBTKExit(
			"Context::get()",
			"No object with the name '" << name << "'"
			<< " exists in the Context.",
			""
		);
	}
}

inline void Context::registerPersistentObject(
	const PersistentObject &persistentObject
){
	std::stringstream ss;
	ss << &persistentObject;
	std::string name = ss.str();
	mutexAllObjects.lock();
	TBTKAssert(
		allObjects.count(name) == 0,
		"Context::registerPersistentObject()",
		"Unable to register object with name '" << name << "' since"
		<< " an object with the same name already exists in this"
		<< " Context.",
		"This should never happen, contact the developer."
	);
	allObjects[name] = &persistentObject;
	mutexAllObjects.unlock();
}

inline void Context::deregisterPersistentObject(
	const PersistentObject &persistentObject
){
	std::stringstream ss;
	ss << &persistentObject;
	std::string name = ss.str();
	mutexAllObjects.lock();
	TBTKAssert(
		allObjects.count(name) == 1,
		"Context::registerPersistentObject()",
		"Unable to deregister object with name '" << name << "' since"
		<< " no object with this name exists in this Context.",
		"This should never happen, contact the developer."
	);
	allObjects.erase(name);
	mutexAllObjects.unlock();
}

};	//End of namespace TBTK

#endif
