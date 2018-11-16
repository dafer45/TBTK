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

/** @file IndexDescriptor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/IndexDescriptor.h"
#include "TBTK/TBTKMacros.h"

#include <string>

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{

IndexDescriptor::IndexDescriptor(){
	format = Format::None;
}

IndexDescriptor::IndexDescriptor(const std::vector<int> &ranges){
	format = Format::Ranges;
	descriptor.rangeFormat.ranges = nullptr;
	descriptor.rangeFormat.dimensions = ranges.size();
	descriptor.rangeFormat.ranges = new int[ranges.size()];
	for(unsigned int n = 0; n < ranges.size(); n++)
		descriptor.rangeFormat.ranges[n] = ranges[n];
}

IndexDescriptor::IndexDescriptor(const IndexTree &indexTree){
	TBTKAssert(
		indexTree.getLinearMapIsGenerated(),
		"IndexDescriptor::setIndexTree()",
		"Linear map not constructed for the IndexTree.",
		"First call IndexTree::generateLinearMap()."
	);

	format = Format::Custom;
	descriptor.customFormat.indexTree = new IndexTree(indexTree);
}

IndexDescriptor::IndexDescriptor(const IndexDescriptor &indexDescriptor){
	format = indexDescriptor.format;
	switch(format){
	case Format::None:
		break;
	case Format::Ranges:
		descriptor.rangeFormat.dimensions = indexDescriptor.descriptor.rangeFormat.dimensions;
		if(indexDescriptor.descriptor.rangeFormat.ranges == nullptr){
			descriptor.rangeFormat.ranges = nullptr;
		}
		else{
			descriptor.rangeFormat.ranges
				= new int[descriptor.rangeFormat.dimensions];
			for(
				unsigned int n = 0;
				n < descriptor.rangeFormat.dimensions;
				n++
			){
				descriptor.rangeFormat.ranges[n]
					= indexDescriptor.descriptor.rangeFormat.ranges[n];
			}
		}
		break;
	case Format::Custom:
		descriptor.customFormat.indexTree = new IndexTree(
			*indexDescriptor.descriptor.customFormat.indexTree
		);
		break;
	case Format::Dynamic:
		descriptor.dynamicFormat.indexedDataTree
			= new IndexedDataTree<unsigned int>(
				*indexDescriptor.descriptor.dynamicFormat.indexedDataTree
			);
		descriptor.dynamicFormat.size
			= indexDescriptor.descriptor.dynamicFormat.size;
		break;
	default:
		TBTKExit(
			"IndexDescriptor::IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

IndexDescriptor::IndexDescriptor(IndexDescriptor &&indexDescriptor){
	format = indexDescriptor.format;
	switch(format){
	case Format::None:
		break;
	case Format::Ranges:
		descriptor.rangeFormat.dimensions
			= indexDescriptor.descriptor.rangeFormat.dimensions;
		if(indexDescriptor.descriptor.rangeFormat.ranges == nullptr){
			descriptor.rangeFormat.ranges = nullptr;
		}
		else{
			descriptor.rangeFormat.ranges
				= indexDescriptor.descriptor.rangeFormat.ranges;
			indexDescriptor.descriptor.rangeFormat.ranges
				= nullptr;
		}
		break;
	case Format::Custom:
		descriptor.customFormat.indexTree
			= indexDescriptor.descriptor.customFormat.indexTree;
		indexDescriptor.descriptor.customFormat.indexTree = nullptr;
		break;
	case Format::Dynamic:
		descriptor.dynamicFormat.indexedDataTree
			= indexDescriptor.descriptor.dynamicFormat.indexedDataTree;
		indexDescriptor.descriptor.dynamicFormat.indexedDataTree
			= nullptr;
		descriptor.dynamicFormat.size
			= indexDescriptor.descriptor.dynamicFormat.size;
		break;
	default:
		TBTKExit(
			"IndexDescriptor::IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

IndexDescriptor::IndexDescriptor(const std::string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "IndexDescriptor", mode),
		"IndexDescriptor::IndexDescriptor()",
		"Unable to parse string as IndexDescriptor '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			string formatString = j.at("format").get<string>();
			if(formatString.compare("None") == 0){
				format = Format::None;
			}
			else if(formatString.compare("Ranges") == 0){
				format = Format::Ranges;

				descriptor.rangeFormat.dimensions = j.at(
					"dimensions"
				).get<int>();

				descriptor.rangeFormat.ranges = new int[
					descriptor.rangeFormat.dimensions
				];
				nlohmann::json ranges = j.at("ranges");
				unsigned int counter = 0;
				for(
					nlohmann::json::iterator it = ranges.begin();
					it < ranges.end();
					++it
				){
					descriptor.rangeFormat.ranges[counter] = *it;
					counter++;
				}
			}
			else if(formatString.compare("Custom") == 0){
				format = Format::Custom;

				descriptor.customFormat.indexTree = new IndexTree(
					j.at("indexTree").dump(),
					mode
				);
			}
			else if(formatString.compare("Dynamic") == 0){
				format = Format::Dynamic;

				descriptor.dynamicFormat.indexedDataTree
					= new IndexedDataTree<unsigned int>(
						j.at("indexedDataTree").dump(),
						mode
					);

				descriptor.dynamicFormat.size
					= j.at("size").get<unsigned int>();
			}
			else{
				TBTKExit(
					"IndexDescriptor::IndexDescriptor",
					"Unknown Format '" << formatString
					<< "'.",
					"The serialization string is either"
					<< " corrupted or the serialization"
					<< " was created with a newer version"
					<< " of TBTK that supports more"
					<< " formats."
				);
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexDescriptor::IndexDescriptor()",
				"Unable to parse string as IndexDescriptor '"
				<< serialization << "'.",
				""
			);
		}
		break;
	default:
		TBTKExit(
			"IndexDescriptor::IndexDescriptor()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

IndexDescriptor::~IndexDescriptor(){
	switch(format){
	case Format::None:
		break;
	case Format::Ranges:
		if(descriptor.rangeFormat.ranges != nullptr)
			delete [] descriptor.rangeFormat.ranges;
		break;
	case Format::Custom:
		if(descriptor.customFormat.indexTree != nullptr)
			delete descriptor.customFormat.indexTree;
		break;
	case Format::Dynamic:
		if(descriptor.dynamicFormat.indexedDataTree != nullptr)
			delete descriptor.dynamicFormat.indexedDataTree;
		break;
	default:
		TBTKExit(
			"IndexDescriptor::~IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

IndexDescriptor& IndexDescriptor::operator=(const IndexDescriptor &rhs){
	if(this != &rhs){
		switch(format){
		case Format::None:
			break;
		case Format::Ranges:
			if(descriptor.rangeFormat.ranges != nullptr){
				delete [] descriptor.rangeFormat.ranges;
				descriptor.rangeFormat.ranges = nullptr;
			}

			break;
		case Format::Custom:
			if(descriptor.customFormat.indexTree != nullptr){
				delete descriptor.customFormat.indexTree;
				descriptor.customFormat.indexTree = nullptr;
			}

			break;
		case Format::Dynamic:
			if(descriptor.dynamicFormat.indexedDataTree != nullptr){
				delete descriptor.dynamicFormat.indexedDataTree;
				descriptor.dynamicFormat.indexedDataTree = nullptr;
			}

			break;
		default:
			TBTKExit(
				"IndexDescriptor::operator=()",
				"This should never happen.",
				"Contact the developer."
			);
		}

		format = rhs.format;
		switch(format){
		case Format::None:
			break;
		case Format::Ranges:
			descriptor.rangeFormat.dimensions = rhs.descriptor.rangeFormat.dimensions;
			if(rhs.descriptor.rangeFormat.ranges == nullptr){
				descriptor.rangeFormat.ranges = nullptr;
			}
			else{
				descriptor.rangeFormat.ranges = new int[descriptor.rangeFormat.dimensions];
				for(unsigned int n = 0; n < descriptor.rangeFormat.dimensions; n++)
					descriptor.rangeFormat.ranges[n] = rhs.descriptor.rangeFormat.ranges[n];
			}
			break;
		case Format::Custom:
			descriptor.customFormat.indexTree = new IndexTree(
				*rhs.descriptor.customFormat.indexTree
			);
			break;
		case Format::Dynamic:
			descriptor.dynamicFormat.indexedDataTree
				= new IndexedDataTree<unsigned int>(
					*rhs.descriptor.dynamicFormat.indexedDataTree
				);
			descriptor.dynamicFormat.size
				= rhs.descriptor.dynamicFormat.size;
			break;
		default:
			TBTKExit(
				"IndexDescriptor::operator=()",
				"This should never happen.",
				"Contact the developer."
			);
		}
	}

	return *this;
}

IndexDescriptor& IndexDescriptor::operator=(IndexDescriptor &&rhs){
	if(this != &rhs){
		switch(format){
		case Format::None:
			break;
		case Format::Ranges:
			if(descriptor.rangeFormat.ranges != nullptr){
				delete [] descriptor.rangeFormat.ranges;
				descriptor.rangeFormat.ranges = nullptr;
			}

			break;
		case Format::Custom:
			if(descriptor.customFormat.indexTree != nullptr){
				delete descriptor.customFormat.indexTree;
				descriptor.customFormat.indexTree = nullptr;
			}

			break;
		case Format::Dynamic:
			if(descriptor.dynamicFormat.indexedDataTree != nullptr){
				delete descriptor.dynamicFormat.indexedDataTree;
				descriptor.dynamicFormat.indexedDataTree = nullptr;
			}

			break;
		default:
			TBTKExit(
				"IndexDescriptor::operator=()",
				"This should never happen.",
				"Contact the developer."
			);
		}

		format = rhs.format;
		switch(format){
		case Format::None:
			break;
		case Format::Ranges:
			descriptor.rangeFormat.dimensions = rhs.descriptor.rangeFormat.dimensions;
			if(rhs.descriptor.rangeFormat.ranges == nullptr){
				descriptor.rangeFormat.ranges = nullptr;
			}
			else{
				descriptor.rangeFormat.ranges = rhs.descriptor.rangeFormat.ranges;
				rhs.descriptor.rangeFormat.ranges = nullptr;
			}
			break;
		case Format::Custom:
			descriptor.customFormat.indexTree = rhs.descriptor.customFormat.indexTree;
			rhs.descriptor.customFormat.indexTree = nullptr;
			break;
		case Format::Dynamic:
			descriptor.dynamicFormat.indexedDataTree
				= rhs.descriptor.dynamicFormat.indexedDataTree;
			rhs.descriptor.dynamicFormat.indexedDataTree = nullptr;
			descriptor.dynamicFormat.size
				= rhs.descriptor.dynamicFormat.size;
			break;
		default:
			TBTKExit(
				"IndexDescriptor::operator=()",
				"This should never happen.",
				"Contact the developer."
			);
		}
	}

	return *this;
}

bool operator==(
	const IndexDescriptor &lhs,
	const IndexDescriptor &rhs
){
	if(lhs.format != rhs.format)
		return false;

	switch(lhs.format){
	case IndexDescriptor::Format::None:
		return true;
	case IndexDescriptor::Format::Ranges:
		if(
			lhs.descriptor.rangeFormat.dimensions
			!= rhs.descriptor.rangeFormat.dimensions
		){
			return false;
		}

		for(
			unsigned int n = 0;
			n < lhs.descriptor.rangeFormat.dimensions;
			n++
		){
			if(
				lhs.descriptor.rangeFormat.ranges[n]
				!= rhs.descriptor.rangeFormat.ranges[n]
			){
				return false;
			}
		}

		return true;
	case IndexDescriptor::Format::Custom:
	{
		return *lhs.descriptor.customFormat.indexTree
			== *rhs.descriptor.customFormat.indexTree;
	}
	case IndexDescriptor::Format::Dynamic:
	{
		TBTKExit(
			"operator==(const IndexDescriptor &lhs, const"
			<< " IndexDescriptor &rhs)",
			"Format::Dynamic not yet implemented.",
			""
		);
	}
	default:
		TBTKExit(
			"operator==(const IndexDescriptor &lhs, const"
			<< " IndexDescriptor &rhs)",
			"Unknown Format.",
			"This should never happen, contact the developer."
		);
	}
}

unsigned int IndexDescriptor::getSize() const{
	switch(format){
	case Format::None:
		return 1;
	case Format::Ranges:
	{
		int size = 1;
		for(unsigned int n = 0; n < descriptor.rangeFormat.dimensions; n++)
			size *= descriptor.rangeFormat.ranges[n];

		return size;
	}
	case Format::Custom:
		return descriptor.customFormat.indexTree->getSize();
	case Format::Dynamic:
		return descriptor.dynamicFormat.size;
	default:
		TBTKExit(
			"IndexDescriptor::operator=()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

std::string IndexDescriptor::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexDescriptor";
		switch(format){
		case Format::None:
			j["format"] = "None";
			break;
		case Format::Ranges:
			j["format"] = "Ranges";
			j["dimensions"] = descriptor.rangeFormat.dimensions;
			for(
				unsigned int n = 0;
				n < descriptor.rangeFormat.dimensions;
				n++
			){
				j["ranges"].push_back(
					descriptor.rangeFormat.ranges[n]
				);
			}
			break;
		case Format::Custom:
			j["format"] = "Custom";
			j["indexTree"] = nlohmann::json::parse(
				descriptor.customFormat.indexTree->serialize(
					mode
				)
			);
			break;
		case Format::Dynamic:
			j["format"] = "Dynamic";
			j["indexedDataTree"] = nlohmann::json::parse(
				descriptor.dynamicFormat.indexedDataTree->serialize(
					mode
				)
			);
			j["size"] = descriptor.dynamicFormat.size;
			break;
		default:
			TBTKExit(
				"IndexDescriptor::serialize()",
				"Unknown Format.",
				"This should never happen, contact the developer."
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"IndexDescriptor::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
