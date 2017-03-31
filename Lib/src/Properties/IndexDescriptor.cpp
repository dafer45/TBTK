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

#include "IndexDescriptor.h"
#include "TBTKMacros.h"

namespace TBTK{

IndexDescriptor::IndexDescriptor(Format format){
	this->format = format;
	switch(format){
	case Format::None:
		break;
	case Format::Ranges:
		descriptor.rangeFormat.ranges = nullptr;
		break;
	case Format::Custom:
		descriptor.customFormat.indexTree = new IndexTree();
		break;
	default:
		TBTKExit(
			"IndexDescriptor::IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
		);
	}
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
			descriptor.rangeFormat.ranges = new int[descriptor.rangeFormat.dimensions];
			for(unsigned int n = 0; n < descriptor.rangeFormat.dimensions; n++)
				descriptor.rangeFormat.ranges[n] = indexDescriptor.descriptor.rangeFormat.ranges[n];
		}
		break;
	case Format::Custom:
		descriptor.customFormat.indexTree = new IndexTree(*indexDescriptor.descriptor.customFormat.indexTree);
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
		descriptor.rangeFormat.dimensions = indexDescriptor.descriptor.rangeFormat.dimensions;
		if(indexDescriptor.descriptor.rangeFormat.ranges == nullptr){
			descriptor.rangeFormat.ranges = nullptr;
		}
		else{
			descriptor.rangeFormat.ranges = indexDescriptor.descriptor.rangeFormat.ranges;
			indexDescriptor.descriptor.rangeFormat.ranges = nullptr;
		}
		break;
	case Format::Custom:
		descriptor.customFormat.indexTree = indexDescriptor.descriptor.customFormat.indexTree;
		indexDescriptor.descriptor.customFormat.indexTree = nullptr;
		break;
	default:
		TBTKExit(
			"IndexDescriptor::IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
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
	default:
		TBTKExit(
			"IndexDescriptor::~IndexDescriptor()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

IndexDescriptor& IndexDescriptor::operator=(const IndexDescriptor &rhs){
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
		descriptor.customFormat.indexTree = new IndexTree(*rhs.descriptor.customFormat.indexTree);
		break;
	default:
		TBTKExit(
			"IndexDescriptor::operator=()",
			"This should never happen.",
			"Contact the developer."
		);
	}

	return *this;
}

IndexDescriptor& IndexDescriptor::operator=(IndexDescriptor &&rhs){
	if(this != &rhs){
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
	default:
		TBTKExit(
			"IndexDescriptor::operator=()",
			"This should never happen.",
			"Contact the developer."
		);
	}
}

};	//End of namespace TBTK
