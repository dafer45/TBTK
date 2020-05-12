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
 *  @brief Read and write string resources from file, URL, etc.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RESOURCE
#define COM_DAFER45_TBTK_RESOURCE

#include "TBTK/Serializable.h"

#include <string>
#include <vector>

namespace TBTK{

/** @brief Read and write string resources from file, URL, etc.
 *
 *  The Resource provides the means for reading input data from files, URL's
 *  etc. It thus provides a unified mechanism for reading data that is agnostic
 *  of the underlying storage medium. Files can be accessed from the local disc
 *  or through any of the protocols supported by cURL (https://curl.haxx.se/).
 *  At the moment, it is only possible to write to a local file.
 *
 *  @link Resource Resources@endlink are particularly useful in combination
 *  with serialization for storing and reading in TBTK objects. An empty
 *  resource is created using
 *  ```cpp
 *    Resource resource;
 *  ```
 *  Once created, it is possible to read a serialized Model from file
 *  ```cpp
 *    resource.read("Model.json");
 *  ```
 *  or a URL
 *  ```cpp
 *    resource.read("http://www.second-quantization.com/v2/ExampleModel.json");
 *  ```
 *  Finally, we get the content of the resource using 'resource.getData()' and
 *  can use it to reconstruct a Model from its serialization
 *  ```cpp
 *    Model model(resource.getData(), Serializable::Mode::JSON);
 *  ```
 *
 *  # Example
 *  \snippet Resource/Resource.cpp Resource
 *  ## Output
 *  \snippet output/Resource/Resource.txt Resource */
class Resource{
public:
	/** Constructor. */
	Resource();

	/** Destructor. */
	~Resource();

	/** Set the data of the Resource.
	 *
	 *  @param data The data the resource should contain. */
	void setData(const std::string &data);

	/** Get the data from the resource.
	 *
	 *  @return The content of the resource. */
	const std::string& getData() const;

	/** Write the resource to destination. Currently the destination can
	 *  only be a local file.
	 *
	 *  @param uri The unique resource identifier (URI) of the destination.
	 *  Should be the name of the file when writing to a local file. */
	void write(const std::string &uri);

	/** Read a resource from a source.
	 *
	 *  @param uri The unique resource identifier (URI) of the source.
	 *  Should be the name of the file when reading from a local file. */
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
