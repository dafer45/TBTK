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
 *  @file IndexedDataTree.h
 *  @brief Data structure for storing data associated with an index.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEXED_DATA_TREE
#define COM_DAFER45_TBTK_INDEXED_DATA_TREE

#include "Index.h"
#include "TBTKMacros.h"

namespace TBTK{

template<typename Data>
class IndexedDataTree{
public:
	/** Constructor. */
	IndexedDataTree();

	/** Destructor. */
	virtual ~IndexedDataTree();

	/** Add indexed data. */
	void add(const Data &data, const Index &index);

	/** Get data. */
	bool get(Data &data, const Index &index) const;

	/** Clear. */
	void clear();

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;
private:
	/** Child nodes.*/
	std::vector<IndexedDataTree> children;

	/** Flag indicating whether the given node corresponds to an index that
	 *  is included in the set. */
	bool indexIncluded;

	/** Data. */
	Data data;

	/** Add indexed data. Is called by the public function
	 *  IndexedDataTree:add() and is called recursively. */
	void add(const Data &data, const Index& index, unsigned int subindex);

	/** Get indexed data. Is called by the public function
	 *  IndexedDataTree::get() and is called recuresively. */
	bool get(Data &data, const Index& index, unsigned int subindex) const;
};

template<typename Data>
IndexedDataTree<Data>::IndexedDataTree(){
	indexIncluded = false;
}

template<typename Data>
IndexedDataTree<Data>::~IndexedDataTree(){
}

template<typename Data>
void IndexedDataTree<Data>::add(
	const Data &data,
	const Index &index
){
	add(data, index, 0);
}

template<typename Data>
void IndexedDataTree<Data>::add(
	const Data &data,
	const Index &index,
	unsigned int subindex
){
	if(subindex < index.getSize()){
		//If the current subindex is not the last, the Index is
		//propagated to the next node level.

		//Get current subindex
		int currentIndex = index.at(subindex);

		TBTKAssert(
			currentIndex >= 0,
			"IndexedDataTree::add()",
			"Invalid Index. Negative indices not allowed, but the"
			<< " index " << index.toString() << " have a negative"
			<< " index in position " << subindex << ".",
			""
		);

		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if(currentIndex >= children.size())
			for(int n = children.size(); n <= currentIndex; n++)
				children.push_back(IndexedDataTree());
		//Error detection:
		//If the current node has the indexIncluded flag set, another
		//Index with fewer subindices than the current Index have
		//previously been added to this node. This is an error because
		//different number of subindices is only allowed if the Indices
		//differ in one of their common indices.
		TBTKAssert(
			!indexIncluded,
			"IndexedDataTree::add()",
			"Incompatible indices. The Index " << index.toString()
			<< " cannot be added because an Index of length "
			<< subindex + 1 << " which exactly agrees with the "
			<< subindex + 1 << " first indices of the current"
			<< " Index has already been added.",
			""
		);

		children.at(currentIndex).add(data, index, subindex+1);
	}
	else{
		//If the current subindex is the last, the index is marked as
		//included.

		//Error detection:
		//If children is non-zero, another Data with more subindices
		//have already been added to this node. This is an error
		//because different number of subindices is only allowed if the
		// indices differ in one of their common indices.
		TBTKAssert(
			children.size() == 0,
			"IndexedDataTree::add()",
			"Incompatible indices. The Index " << index.toString()
			<< " cannot be added because a longer Index which"
			<< " exactly agrees with the current Index in the"
			<< " common indices has already been added.",
			""
		);

		indexIncluded = true;
		this->data = data;
	}
}

template<typename Data>
bool IndexedDataTree<Data>::get(Data &data, const Index &index) const{
	return get(data, index, 0);
}

template<typename Data>
bool IndexedDataTree<Data>::get(
	Data &data,
	const Index &index,
	unsigned int subindex
) const{
	if(subindex < index.getSize()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex.
		int currentIndex = index.at(subindex);

		TBTKAssert(
			currentIndex >= 0,
			"IndexedDataTree::add()",
			"Invalid Index. Negative indices not allowed, but the"
			<< " index " << index.toString() << " have a negative"
			<< " index in position " << subindex << ".",
			""
		);

		//Return false because the Index is not included.
		if(currentIndex >= children.size())
			return false;

		return children.at(currentIndex).get(data, index, subindex+1);
	}
	else{
		//If the current subindex is the last, try to extract the data.
		//Return true if successful but false if the data does not
		//exist.
		if(indexIncluded){
			data = this->data;

			return true;
		}
		else{
			return false;
		}
	}
}

template<typename Data>
void IndexedDataTree<Data>::clear(){
	indexIncluded = false;
	children.clear();
}

template<typename Data>
unsigned int IndexedDataTree<Data>::getSizeInBytes() const{
	unsigned int size = sizeof(IndexedDataTree<Data>);
	for(unsigned int n = 0; n < children.size(); n++)
		size += children.at(n).getSizeInBytes();

	return size;
}

}; //End of namesapce TBTK

#endif
