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

/** @file Context.cpp
 *  @author Kristofer Björnson
 */

#include "TBTK/Context.h"

using namespace std;

namespace TBTK{

Context::Context(){
}

Context::~Context(){
	for(auto object : objects)
		delete object.second;
}

Context& Context::getContext(){
	static Context context;
	return context;
}

};
