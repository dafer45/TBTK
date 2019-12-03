/* Copyright 2018 Kristofer Björnson
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

/** @file Vector2d.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/Vector2d.h"

using namespace std;

namespace TBTK{

Vector2d::Vector2d(){
}

Vector2d::Vector2d(initializer_list<double> components){
	TBTKAssert(
		components.size() == 2,
		"Vector3d::Vector3d()",
		"Incompatible dimensions. Vector2d takes 2 components, but "
		<< components.size() << " was given.",
		""
	);

	x = *(components.begin() + 0);
	y = *(components.begin() + 1);
}

Vector2d::Vector2d(const vector<double> &components){
	TBTKAssert(
		components.size() == 2,
		"Vector3d::Vector2d()",
		"Incompatible dimensions. Vector2d takes 2 components, but "
		<< components.size() << " was given.",
		""
	);

	x = *(components.begin() + 0);
	y = *(components.begin() + 1);
}

};	//End of namespace TBTK
