/* Copyright 2019 Kristofer Björnson
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

/** @file AbstractProperty.cpp
 *  @brief Generates and validates @link AbstractProperty AbstractProperties@endlink.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/Validation/Validation.h"

#include <iostream>

using namespace std;
using namespace TBTK;

int NUM_TESTS = 2;

class PublicAbstractProperty : public Property::AbstractProperty<int>{
public:
	PublicAbstractProperty() : AbstractProperty<int>(){}

	PublicAbstractProperty(
		unsigned int blockSize,
		const CArray<int> &data
	) :
		AbstractProperty<int>(blockSize, data){}

	PublicAbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const CArray<int> &data
	) :
		AbstractProperty<int>(indexTree, blockSize, data){}

	PublicAbstractProperty(
		const string &serialization,
		Serializable::Mode mode
	) :
		AbstractProperty<int>(serialization, mode){}
};

int main(int argc, char **argv){
	Mode mode;
	int id;
	init(argc, argv, mode, id, "AbstractProperty");

	if(mode == NumTests){
		cout << NUM_TESTS;

		return 0;
	}

	PublicAbstractProperty abstractProperty;
	switch(id){
	case 0:
	{
		CArray<int> data(10);
		for(unsigned int n = 0; n < 10; n++)
			data[n] = n;

		abstractProperty = PublicAbstractProperty(10, data);
		abstractProperty.setDefaultValue(137);

		break;
	}
	case 1:
	{
		CArray<int> data(30);
		for(unsigned int n = 0; n < 30; n++)
			data[n] = n;

		IndexTree indexTree;
		indexTree.add({1, 2, 3});
		indexTree.add({1, 2, 4});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();

		abstractProperty = PublicAbstractProperty(indexTree, 10, data);
		abstractProperty.setDefaultValue(137);

		break;
	}
	default:
		TBTKExit(
			"AbstractProperty",
			"Unknown test id '" << id << "'.",
			""
		);
	}

	execute<PublicAbstractProperty>(abstractProperty, mode, id, "AbstractProperty");

	return 0;
}
