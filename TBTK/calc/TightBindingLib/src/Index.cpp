/* Copyright 2016 Kristofer Björnson
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

/** @file Index.cpp
 *  @author Kristofer Björnson
 */

#include "../include/Index.h"
#include "../include/Streams.h"
#include "../include/TBTKMacros.h"

using namespace std;

namespace TBTK{

bool operator<(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.size() < i2.size())
		minNumIndices = i1.size();
	else
		minNumIndices = i2.size();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return true;
		else
			return false;
	}

	TBTKExit(
		"operator<(Index &i1, Index &i2)",
		"Comparison between indices of types mutually incompatible with the TreeNode structure.",
		""
	);
}

bool operator>(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.size() < i2.size())
		minNumIndices = i1.size();
	else
		minNumIndices = i2.size();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return false;
		else
			return true;
	}

	TBTKExit(
		"operator>(Index &i1, Index &i2)",
		"Comparison between indices of types mutually incompatible with the TreeNode structure.",
		""
	);
}

Index Index::getUnitRange(){
	Index unitRange = *this;

	for(unsigned int n = 0; n < size(); n++)
		unitRange.at(n) = 1;

	return unitRange;
}

};
