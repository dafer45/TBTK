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

/** @file IndexDescriptor.cpp
 *  @brief Generates and validates @link IndexDescriptor
 *  IndexDescriptors@endlink.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/IndexDescriptor.h"
#include "TBTK/Validation/Validation.h"

#include <iostream>

using namespace std;
using namespace TBTK;

int NUM_TESTS = 2;

int main(int argc, char **argv){
	Mode mode;
	int id;
	init(argc, argv, mode, id, "IndexDescriptor");

	if(mode == NumTests){
		cout << NUM_TESTS;

		return 0;
	}

	IndexDescriptor indexDescriptor;
	switch(id){
	case 0:
		break;
	case 1:
	{
		IndexTree indexTree;
		indexTree.add({{1, 2}, {3, 4}});
		indexTree.add({{1, 2}, {4, 5}});
		indexTree.add({{2}, {1, 2}});
		indexTree.add({3, 4, 5});
		indexTree.generateLinearMap();

		indexDescriptor = IndexDescriptor(indexTree);

		break;
	}
	default:
		TBTKExit(
			"IndexDescriptor",
			"Unknown test id '" << id << "'.",
			""
		);
	}

	execute<IndexDescriptor>(indexDescriptor, mode, id, "IndexDescriptor");

	return 0;
}
