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

/** @file IndexTree.cpp
 *  @author Kristofer Björnson
 */

#include "IndexTree.h"
#include "TBTKMacros.h"

using namespace std;

namespace TBTK{

IndexTree::IndexTree(){
	indexIncluded = false;
	wildcardIndex = false;
	linearIndex = -1;
	size = -1;
}

IndexTree::~IndexTree(){
}

void IndexTree::add(const Index &index){
	add(index, 0);
}

void IndexTree::add(const Index &index, unsigned int subindex){
	if(subindex < index.size()){
		//If the current subindex is not the last, the Index is
		//propagated to the next node level.

		//Get current subindex
		int currentIndex = index.at(subindex);

		if(currentIndex < 0){
			if(!wildcardIndex){
				if(children.size() > 0){
					Streams::err << "Error in IndexTree::add(). Tried to add index with wild card in subindex position for which non wild card indices already have been added:\n";
					index.print();
					exit(1);
				}
				wildcardIndex = true;
			}
			currentIndex = 0;
		}
		else if(wildcardIndex){
			Streams::err << "Error in IndexTree::add(). Unable to add index because an index with a wild card in subindex position " << subindex << " already have been added:\n";
			index.print();
			exit(1);
		}

		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if(currentIndex >= (int)children.size())
			for(int n = children.size(); n <= currentIndex; n++)
				children.push_back(IndexTree());
		//Error detection:
		//If the current node has the indexIncluded flag set, another
		//Index with fewer subindices than the current Index have
		//previously been added to this node. This is an error because
		//different number of subindices is only allowed if the Indices
		//differ in one of their common indices.
		if(indexIncluded){
			Streams::err << "Error in IndexTree::add, index incompatible with previously added index at subindex " << subindex << ":\n";
			index.print();
			exit(1);
		}
		//Propagate to the next node level.
		children.at(currentIndex).add(index, subindex+1);
	}
	else{
		//If the current subindex is the last, the Index is marked as included.

		//Error detection:
		//If childen is non-zeros, another HoppingAmplitude with more
		//indices have already been added to this node. This is an
		//error because different number of subindices is only allowed
		//if the HoppingAmplitudes differ in one of their common
		//indices.
		if(children.size() != 0){
			Streams::err << "Error in IndexTree::add, index incompatible with previously added index at subindex " << subindex << ":\n";
			index.print();
			exit(1);
		}

		indexIncluded = true;
	}
}

void IndexTree::generateLinearMap(){
	size = generateLinearMap(0);
}

int IndexTree::generateLinearMap(int i){
	if(children.size() == 0){
		if(indexIncluded){
			linearIndex = i;
			return i + 1;
		}
		else{
			return i;
		}
	}

	for(unsigned int n = 0; n < children.size(); n++)
		i = children.at(n).generateLinearMap(i);

	return i;
}

int IndexTree::getLinearIndex(const Index &index, bool ignoreWildcards) const{
	return getLinearIndex(index, 0, ignoreWildcards);
}

int IndexTree::getLinearIndex(const Index &index, unsigned int subindex, bool ignoreWildcards) const{
	if(ignoreWildcards && wildcardIndex)
		return children.at(0).getLinearIndex(index, subindex, ignoreWildcards);

	if(subindex < index.size()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		int currentIndex = index.at(subindex);
		if(currentIndex < 0){
			if(wildcardIndex){
				currentIndex = 0;
			}
			else{
				Streams::err << "Error in IndexTree::getLinearIndex. Subindex " << subindex << " should not be a wild card index:\n";
				index.print();
				exit(1);
			}
		}
		else if(wildcardIndex){
			Streams::err << "Error in IndexTree::getLinearIndex. Subindex " << subindex << " has to be a wild card index:\n";
			index.print();
			exit(1);
		}
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			Streams::err << "Error, index out of bound: ";
			index.print();
			exit(1);
		}
		//Continue to the next node level.
		return children.at(currentIndex).getLinearIndex(index, subindex+1, ignoreWildcards);
	}
	else{
		//If the current subindex is the last, return linear index.
		if(indexIncluded){
			return linearIndex;
		}
		else{
			Streams::err << "Error, index not included in the IndexTree: ";
			index.print();
			exit(1);
		}
	}
}

Index IndexTree::getPhysicalIndex(int linearIndex) const{
	TBTKAssert(
		linearIndex >= 0 && linearIndex < this->size,
		"IndexTree::getPhysicalIndex()",
		"Linear index out of bound.",
		""
	);

	vector<int> indices;
	getPhysicalIndex(linearIndex, &indices);

	return Index(indices);
}

void IndexTree::getPhysicalIndex(int linearIndex, vector<int> *indices) const{
	if(this->linearIndex != -1)
		return;

	for(unsigned int n = 0; n < children.size(); n++){
		int min = children.at(n).getMinIndex();
		int max = children.at(n).getMaxIndex();

		if(min == -1)
			continue;

		if(min <= linearIndex && linearIndex <= max){
			if(wildcardIndex)
				indices->push_back(IDX_ALL);
			else
				indices->push_back(n);
			children.at(n).getPhysicalIndex(linearIndex, indices);
			break;
		}
	}
}

int IndexTree::getMinIndex() const{
	if(linearIndex != -1)
		return linearIndex;

	int min = -1;
	for(unsigned int n = 0; n < children.size(); n++){
		min = children.at(n).getMinIndex();
		if(min != -1)
			break;
	}

	return min;
}

int IndexTree::getMaxIndex() const{
	if(linearIndex != -1)
		return linearIndex;

	int max = -1;
	for(int n = children.size()-1; n >= 0; n--){
		max = children.at(n).getMaxIndex();
		if(max != -1)
			break;
	}

	return max;
}

};
