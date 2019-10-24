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

#include "TBTK/ElementNotFoundException.h"
#include "TBTK/Index.h"
#include "TBTK/PseudoSerializable.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <map>
#include <sstream>

//This is used to work around incompatibilities between nlohmann::json and
//CUDA. This effectively forbids instantiation of IndexedDataTree in CUDA code.
#ifndef TBTK_DISABLE_NLOHMANN_JSON
#	include "TBTK/json.hpp"
#endif

namespace TBTK{

template<typename Data>
class IndexedDataTree : public Serializable{
public:
	/** Constructor. */
	IndexedDataTree();

	/** Constructor. Constructs the IndexedDataTree from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the IndexedDataTree.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	IndexedDataTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~IndexedDataTree();

	/** Add indexed data. Multiple calls to the function with the same
	 *  Index will overwrite previous data.
	 *
	 *  @param data Data element to add.
	 *  @param index Index at which to add the element. */
	void add(const Data &data, const Index &index);

	/** Get data.
	 *
	 *  @param data Reference to an object that the data will be written
	 *  to.
	 *
	 *  @param index Index for which to extract the data for.
	 *
	 *  @return True if a data element was found, otherwise false. */
	bool get(Data &data, const Index &index) const;

	/** Get data.
	 *
	 *  @param index Index for which to extract the data for.
	 *
	 *  @return Reference to the element.
	 *
	 *  @throws ElementNotFoundException If the no element with the
	 *  requested Index exists. */
	Data& get(const Index &index);

	/** Get data.
	 *
	 *  @param index Index for which to extract the data for.
	 *
	 *  @return Reference to the element.
	 *
	 *  @throws ElementNotFoundException If the no element with the
	 *  requested Index exists. */
	const Data& get(const Index &index) const;

	/** Clear the data. */
	void clear();

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the IndexedDataTree. */
	unsigned int getSizeInBytes() const;

	/** Serilaize.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string representation of the IndexedDataTree. */
	virtual std::string serialize(Mode mode) const;

	class Iterator;
	class ConstIterator;
private:
	/** Base classed used by Iterator and ConstIterator for iterating
	 *  through the elements stored in the tree structure. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const Data&,
			Data&
		>::type DataReferenceType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		DataReferenceType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs) const;

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs) const;

		/** Get the current Index. */
		const Index& getCurrentIndex() const;
	private:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const IndexedDataTree*,
			IndexedDataTree*
		>::type IndexedDataTreePointerType;

		/** IndexedDataTree to iterate over. */
		IndexedDataTreePointerType indexedDataTree;

		/** Current Index. */
		Index currentIndex;

		/** Private constructor. Limits the ability to construct an
		 *  Iterator to the IndexedDataTree. */
		_Iterator(IndexedDataTreePointerType indexedDataTree, bool end = false);

		/** Make the IndexedDataTree able to construct an Iterator. */
		friend class Iterator;
		friend class ConstIterator;
	};
public:
	/** Iterator for iterating through the elements stored in the tree
	 *  structure. */
	class Iterator : public _Iterator<false>{
	private:
		Iterator(
			IndexedDataTree *indexedDataTree,
			bool end = false
		) : _Iterator<false>(indexedDataTree, end){};

		/** Make the IndexedDataTree able to construct an Iterator. */
		friend class IndexedDataTree;
	};

	/** ConstIterator for iterating through the elements stored in the tree
	 *  structure. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const IndexedDataTree *indexedDataTree,
			bool end = false
		) : _Iterator<true>(indexedDataTree, end){};

		/** Make the IndexedDataTree able to construct an Iterator. */
		friend class IndexedDataTree;
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  IndexedDataTree. */
	Iterator begin();

	/** Create ConstIterator.
	 *
	 *  @return ConstIterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator begin() const;

	/** Create ConstIterator.
	 *
	 *  @return ConstIterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator cbegin() const;

	/** Get Iterator pointing to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	Iterator end();

	/** Get ConstIterator pointing to the end.
	 *
	 *  @return A ConstIterator pointing at the end of the IndexedDataTree. */
	ConstIterator end() const;

	/** Get ConstIterator that points to the end.
	 *
	 *  @return A ConstIterator pointing at the end of the IndexedDataTree. */
	ConstIterator cend() const;
private:
	/** Child nodes. */
	std::map<Subindex, IndexedDataTree> children;

	/** Flag indicating whether the given node corresponds to an index that
	 *  is included in the set. */
	bool indexIncluded;

	/** Flag indicating whether the given node is an Index-separator. I.e.,
	 *  whether the next node level corresponds to the first subindex of a
	 *  new Index. */
	bool indexSeparator;

	/** Data. */
	Data data;

	/** Add indexed data. Is called by the public function
	 *  IndexedDataTree:add() and is called recursively. */
	void add(const Data &data, const Index& index, unsigned int subindex);

	/** Get indexed data. Is called by the public function
	 *  IndexedDataTree::get() and is called recuresively. */
	bool get(Data &data, const Index& index, unsigned int subindex) const;

	/** Get indexed data. Is called by the public function
	 *  IndexedDataTree::get() and is called recuresively. */
	const Data& get(const Index& index, unsigned int subindex) const;

	/** Returns the first Index for which an element exists. */
	Index getFirstIndex() const;

	/** Function called by IndexedDataTree::getFirstIndex() to perform the
	 *  actual work of finding the first Index. */
	bool getFirstIndex(Index &index) const;

	/** Get the Index that follows the given Index. Returns the empty Index
	 *  if no next Index exists. */
	Index getNextIndex(const Index &index) const;

	/** Function called by IndexedDataTree::getNextIndex() to perform the
	 *  actual work of finding the next Index. Is called recursively and
	 *  returns true when the Index has been found. */
	bool getNextIndex(
		const Index &currentIndex,
		Index &nextIndex
	) const;
};

//This is used to work around incompatibilities between nlohmann::json and
//CUDA. This effectively forbids instantiation of IndexedDataTree in CUDA code.
#ifndef TBTK_DISABLE_NLOHMANN_JSON

template<typename Data>
IndexedDataTree<Data>::IndexedDataTree(){
	indexIncluded = false;
	indexSeparator = false;
}

template<typename Data>
inline IndexedDataTree<Data>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<bool>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<bool> '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(
				serialization
			);
			indexIncluded = j.at("indexIncluded").get<bool>();
			indexSeparator = j.at("indexSeparator").get<bool>();
			data = Serializable::deserialize<Data>(
				j.at("data").get<std::string>(),
				mode
			);
			nlohmann::json jsonChildren = j.at("children");
			for(
				nlohmann::json::const_iterator iterator
					= jsonChildren.cbegin();
				iterator != jsonChildren.cend();
				++iterator
			){
				children.insert({
					Subindex(
						iterator.key(),
						Serializable::Mode::JSON
					),
					IndexedDataTree(
						iterator.value().dump(),
						Serializable::Mode::JSON
					)
				});
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<bool>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<bool> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<Data>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename Data>
IndexedDataTree<Data>::~IndexedDataTree(){
}

template<typename Data>
void IndexedDataTree<Data>::add(const Data &data, const Index &index){
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
		Subindex currentIndex = index.at(subindex);

		if(currentIndex.isIndexSeparator()){
			if(children.size() == 0){
				indexSeparator = true;
			}
			else{
				TBTKAssert(
					indexSeparator,
					"IndexedDataTree:add()",
					"Invalid index '" << index.toString()
					<< "'. Another Index has already been"
					<< " added to the tree that has a"
					<< " conflicting index at the index"
					<< " separator at subindex '"
					<< subindex << "'.",
					"Note that a separation point between"
					<< " two indices counts as a subindex."
				);
			}

			indexSeparator = false;
			add(data, index, subindex+1);
			indexSeparator = true;
			return;
		}
		else{
			TBTKAssert(
				!indexSeparator,
				"IndexedDataTree:add()",
				"Invalid index '" << index.toString() << "'."
				<< " Another Index has already been added to"
				<< " the tree that has a conflicting index"
				<< " separator at subindex '"
				<< subindex << "'.",
				"Note that a separation point between two"
				<< " indices counts as a subindex."
			);
		}

		TBTKAssert(
			currentIndex >= 0,
			"IndexedDataTree::add()",
			"Invalid Index. Negative indices not allowed, but the"
			<< "index " << index.toString() << " have a negative"
			<< " index" << " in position " << subindex << ".",
			"Compound indices such as {{1, 2, 3}, {4, 5, 6}} are"
			<< " separated by IDX_SEPARATOR with the value '"
			<< IDX_SEPARATOR << "' and are" << " represented as {1"
			<< ", 2, 3, " << IDX_SEPARATOR << ", 4, 5, 6}. This is"
			<< " the only allowed instance of negative numbers."
		);

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

		children[currentIndex].add(data, index, subindex+1);
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

		//Return false because this is a leaf node without the
		//indexIncluded flag set. This means it must have been added to
		//pad the parents child vector and not as a consequence of an
		//actual Index having been associated with the node.
		if(children.size() == 0 && !indexIncluded)
			return false;

		//Get current subindex.
		Subindex currentIndex = index.at(subindex);

		if(currentIndex.isIndexSeparator()){
			if(indexSeparator){
				return get(data, index, subindex+1);
			}
			else{
				TBTKExit(
					"IndexedDataTree::get()",
					"Invalid Index. Found IDX_SEPARATOR at"
					<< " subindex '" << subindex << "',"
					<< " but the node is not an index"
					<< " separator.",
					""
				);
			}
		}

		TBTKAssert(
			currentIndex >= 0,
			"IndexedDataTree::add()",
			"Invalid Index. Negative indices not allowed, but the"
			<< " index " << index.toString() << " have a negative"
			<< " index in position " << subindex << ".",
			""
		);

		try{
			return children.at(currentIndex).get(data, index, subindex+1);
		}
		catch(std::out_of_range &e){
			return false;
		}
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
Data& IndexedDataTree<Data>::get(const Index &index){
	//Casting is safe because we do not guarantee that the IndexedDataTree
	//is not modified. Casting away const from the returned reference is
	//therefore not a violation of any promisse made by this function.
	//See also "Avoiding Duplication in const and Non-const Member
	//Function" in S. Meyers, Effective C++.
	return const_cast<Data&>(
		static_cast<const IndexedDataTree<Data>*>(this)->get(index, 0)
	);
}

template<typename Data>
const Data& IndexedDataTree<Data>::get(const Index &index) const{
	return get(index, 0);
}

template<typename Data>
const Data& IndexedDataTree<Data>::get(
	const Index &index,
	unsigned int subindex
) const{
	if(subindex < index.getSize()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Throw ElementNotFoundException if the Index is not included.
		//This statement is executed if this is a leaf node without the
		//indexIncluded flag set. This means it must have been added to
		//pad the parents child vector and not as a consequence of an
		//actual Index having been associated with the node.
		if(children.size() == 0 && !indexIncluded){
			throw ElementNotFoundException(
				"IndexedDataTree()",
				TBTKWhere,
				"Tried to get element with Index '"
				+ index.toString() + "', but no such element"
				+ " exists.",
				""
			);
		}

		//Get current subindex.
		Subindex currentIndex = index.at(subindex);

		if(currentIndex.isIndexSeparator()){
			if(indexSeparator){
				return get(index, subindex+1);
			}
			else{
				TBTKExit(
					"IndexedDataTree::get()",
					"Invalid Index. Found IDX_SEPARATOR at"
					<< " subindex '" << subindex << "',"
					<< " but the node is not an index"
					<< " separator.",
					""
				);
			}
		}

		TBTKAssert(
			currentIndex >= 0,
			"IndexedDataTree::get()",
			"Invalid Index. Negative indices not allowed, but the"
			<< "index " << index.toString() << " have a negative"
			<< " index" << " in position " << subindex << ".",
			"Compound indices such as {{1, 2, 3}, {4, 5, 6}} are"
			<< " separated by IDX_SEPARATOR with the value '"
			<< IDX_SEPARATOR << "' and are" << " represented as {1"
			<< ", 2, 3, " << IDX_SEPARATOR << ", 4, 5, 6}. This is"
			<< " the only allowed instance of negative numbers."
		);

		try{
			return children.at(currentIndex).get(index, subindex+1);
		}
		catch(std::out_of_range &e){
			throw ElementNotFoundException(
				"IndexedDataTree()",
				TBTKWhere,
				"Tried to get element with Index '"
				+ index.toString() + "', but no such element"
				+ " exists.",
				""
			);
		}
	}
	else{
		//If the current subindex is the last, try to extract the data.
		//Return data if successful but throw ElementNotFoundException if the
		//data does not exist.
		if(indexIncluded){
			return data;
		}
		else{
			throw ElementNotFoundException(
				"IndexedDataTree()",
				TBTKWhere,
				"Tried to get element with Index '"
				+ index.toString() + "', but no such element"
				+ " exists.",
				""
			);
		}
	}
}

template<typename Data>
Index IndexedDataTree<Data>::getFirstIndex() const{
	Index index;
	getFirstIndex(index);

	return index;
}

template<typename Data>
bool IndexedDataTree<Data>::getFirstIndex(Index &index) const{
	if(indexIncluded)
		return true;

	if(indexSeparator)
		index.pushBack(IDX_SEPARATOR);

	for(
		typename std::map<
			Subindex,
			IndexedDataTree
		>::const_iterator iterator = children.cbegin();
		iterator != children.cend();
		++iterator
	){
		Subindex subindex = iterator->first;
		index.pushBack(subindex);
		if(iterator->second.getFirstIndex(index))
			return true;

		index.popBack();
	}

	if(indexSeparator)
		index.popBack();

	return false;
}

template<typename Data>
Index IndexedDataTree<Data>::getNextIndex(const Index &index) const{
	if(index.getSize() == 0)
		return Index();

	Index nextIndex;
	getNextIndex(index, nextIndex);

	return nextIndex;
}

template<typename Data>
bool IndexedDataTree<Data>::getNextIndex(
	const Index &currentIndex,
	Index &nextIndex
) const{
	if(indexIncluded){
		if(currentIndex.equals(nextIndex))
			return false;

		return true;
	}

	if(indexSeparator)
		nextIndex.pushBack(IDX_SEPARATOR);

	bool hasSameIndexStructure = true;
	if(currentIndex.getSize() > nextIndex.getSize()){
		for(unsigned int n = 0; n < nextIndex.getSize(); n++){
			if(currentIndex[n] != nextIndex[n]){
				hasSameIndexStructure = false;
				break;
			}
		}
	}
	else{
		hasSameIndexStructure = false;
	}

	typename std::map<Subindex, IndexedDataTree>::const_iterator iterator;
	if(hasSameIndexStructure)
		iterator = children.find(currentIndex[nextIndex.getSize()]);
	else
		iterator = children.cbegin();
	while(iterator != children.cend()){
		nextIndex.pushBack(iterator->first);
		if(iterator->second.getNextIndex(currentIndex, nextIndex))
			return true;
		nextIndex.popBack();

		++iterator;
	}

	if(indexSeparator)
		nextIndex.popBack();

	return false;
}

template<typename Data>
void IndexedDataTree<Data>::clear(){
	indexIncluded = false;
	children.clear();
}

template<typename Data>
unsigned int IndexedDataTree<Data>::getSizeInBytes() const{
	unsigned int size = sizeof(IndexedDataTree<Data>);
	for(
		typename std::map<
			Subindex,
			IndexedDataTree
		>::const_iterator iterator = children.cbegin();
		iterator != children.cend();
		++iterator
	){
		size += iterator->second.getSizeInBytes();
	}

	return size;
}

template<typename Data>
inline std::string IndexedDataTree<Data>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = Serializable::serialize(data, mode);
		j["children"] = nlohmann::json();
		for(
			typename std::map<
				Subindex,
				IndexedDataTree
			>::const_iterator iterator = children.cbegin();
			iterator != children.cend();
			++iterator
		){
			j["children"][
				iterator->first.serialize(
					Serializable::Mode::JSON
				)
			] = nlohmann::json::parse(
				iterator->second.serialize(
					Serializable::Mode::JSON
				)
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"IndexedDataTree<Data>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename Data>
typename IndexedDataTree<Data>::Iterator IndexedDataTree<Data>::begin(){
	return Iterator(this);
}

template<typename Data>
typename IndexedDataTree<Data>::ConstIterator IndexedDataTree<
	Data
>::begin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data>::ConstIterator IndexedDataTree<
	Data
>::cbegin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data>::Iterator IndexedDataTree<Data>::end(){
	return Iterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data>::ConstIterator IndexedDataTree<Data>::end() const{
	return ConstIterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data>::ConstIterator IndexedDataTree<Data>::cend() const{
	return ConstIterator(this, true);
}

template<typename Data> template<bool isConstIterator>
void IndexedDataTree<Data>::_Iterator<isConstIterator>::operator++(){
	currentIndex = indexedDataTree->getNextIndex(currentIndex);
}

template<typename Data> template<bool isConstIterator>
typename IndexedDataTree<Data>::template _Iterator<
	isConstIterator
>::DataReferenceType IndexedDataTree<Data>::_Iterator<
	isConstIterator
>::operator*(){
	return indexedDataTree->get(currentIndex);
}

template<typename Data> template<bool isConstIterator>
bool IndexedDataTree<Data>::_Iterator<isConstIterator>::operator==(
	const IndexedDataTree<Data>::_Iterator<isConstIterator> &rhs
) const{
	if(
		indexedDataTree == rhs.indexedDataTree
		&& currentIndex.equals(rhs.currentIndex)
	){
		return true;
	}
	else{
		return false;
	}
}

template<typename Data> template<bool isConstIterator>
bool IndexedDataTree<Data>::_Iterator<isConstIterator>::operator!=(
	const IndexedDataTree<Data>::_Iterator<isConstIterator> &rhs
) const{
	if(
		indexedDataTree != rhs.indexedDataTree
		|| !currentIndex.equals(rhs.currentIndex)
	){
		return true;
	}
	else{
		return false;
	}
}

template<typename Data> template<bool isConstIterator>
const Index& IndexedDataTree<Data>::_Iterator<
	isConstIterator
>::getCurrentIndex(
) const{
	return currentIndex;
}

template<typename Data> template<bool isConstIterator>
IndexedDataTree<Data>::_Iterator<isConstIterator>::_Iterator(
	IndexedDataTreePointerType indexedDataTree,
	bool end
){
	this->indexedDataTree = indexedDataTree;
	if(end)
		currentIndex = Index();
	else
		currentIndex = indexedDataTree->getFirstIndex();
}

//This is used to work around incompatibilities between nlohmann::json and
//CUDA. This effectively forbids instantiation of IndexedDataTree in CUDA code.
#endif

}; //End of namesapce TBTK

#endif
