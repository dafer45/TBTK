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
 *  @file Resource.h
 *  @brief Allows Serializeable objects to be saved and loaded using URI.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RESOURCE
#define COM_DAFER45_TBTK_RESOURCE

#include "Serializeable.h"

#include <string>
#include <vector>

namespace TBTK{

class Resource{
public:
	/** Constructor. */
	Resource();

	/** Destructor. */
	~Resource();

	/** Set resource. */
	void setData(const std::string &data);

	/** Get serialization. */
	const std::string& getData() const;

	/** Write resource. */
	void write(const std::string &uri);

	/** Read resource. */
	void read(const std::string &uri);
private:
	/** Serialized resource. */
	std::string data;

	/** Write callback. */
	static size_t writeCallback(
		void *data,
		size_t size,
		size_t nmemb,
		void *userdata
	);

	/** Read callback. */
	static size_t readCallback(
		char *data,
		size_t size,
		size_t nmemb,
		void *userdata
	);
};

inline void Resource::setData(const std::string &data){
	this->data = data;
}

inline const std::string& Resource::getData() const{
	return data;
}

};	//End namespace TBTK

#endif
