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
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <sstream>

//This is used to work around incompatibilities between nlohmann::json and
//CUDA. This effectively forbids instantiation of IndexedDataTree in CUDA code.
#ifndef TBTK_DISABLE_NLOHMANN_JSON
#	include "TBTK/json.hpp"
#endif

namespace TBTK{

template<typename Data, bool isSerializeable = std::is_base_of<Serializable, Data>::value>
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
	std::vector<IndexedDataTree> children;

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

template<typename Data>
class IndexedDataTree<Data, true> : public Serializable{
public:
	/** Constructor. */
	IndexedDataTree();

	/** Constructor. Constructs the IndexedDataTree from a serialization
	 *  string. */
	IndexedDataTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~IndexedDataTree();

	/** Add indexed data. */
	void add(const Data &data, const Index &index);

	/** Get data. */
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

	/** Clear. */
	void clear();

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;

	/** Serilaize. */
	virtual std::string serialize(Mode mode) const;

	/** Base class used by Iterator and ConstIterator for iterating through
	 *  the elements stored in the tree structure. */
	class Iterator;
	class ConstIterator;
private:
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

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator begin() const;

	/** Create Iterator for constant elements.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator cbegin() const;

	/** Get Iterator pointing to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	Iterator end();

	/** Get Iterator pointing to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	ConstIterator end() const;

	/** Get Iterator for constatne elements that points to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	ConstIterator cend() const;
private:
	/** Child nodes. */
	std::vector<IndexedDataTree> children;

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

template<typename Data>
class IndexedDataTree<Data, false> : public Serializable{
public:
	/** Constructor. */
	IndexedDataTree();

	/** Constructor. Constructs the IndexedDataTree from a serialization
	 *  string. */
	IndexedDataTree(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~IndexedDataTree();

	/** Add indexed data. */
	void add(const Data &data, const Index &index);

	/** Get data. */
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

	/** Clear. */
	void clear();

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;

	/** Serilaize. */
	virtual std::string serialize(Mode mode) const;

	/** Base class used by Iterator and ConstIterator for iterating through
	 *  the elements stored in the tree structure. */
	class Iterator;
	class ConstIterator;
private:
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

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator begin() const;

	/** Create Iterator for constant elements.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  IndexedDataTree. */
	ConstIterator cbegin() const;

	/** Get Iterator pointing to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	Iterator end();

	/** Get Iterator pointing to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	ConstIterator end() const;

	/** Get Iterator for constatne elements that points to the end.
	 *
	 *  @return An Iterator pointing at the end of the IndexedDataTree. */
	ConstIterator cend() const;
private:
	/** Child nodes. */
	std::vector<IndexedDataTree> children;

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

	/** Returns the first Index for which an element exists. Return the
	 *  empty Index if no element exists. */
	Index getFirstIndex() const;

	/** Function called by IndexedDataTree::getFirstIndex() to perform the
	 *  actual work of finding the first Index. Is called recursively and
	 *  returns true when the Index has been found. */
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

template<typename Data, bool isSerializable>
IndexedDataTree<Data, isSerializable>::IndexedDataTree(){
	indexIncluded = false;
	indexSeparator = false;
}

template<typename Data>
IndexedDataTree<Data, true>::IndexedDataTree(){
	indexIncluded = false;
	indexSeparator = false;
}

template<typename Data>
IndexedDataTree<Data, false>::IndexedDataTree(){
	indexIncluded = false;
	indexSeparator = false;
}

template<>
inline IndexedDataTree<bool, false>::IndexedDataTree(
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
			data = j.at("data").get<bool>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<bool>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
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
			"IndexedDataTree<bool>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline IndexedDataTree<char, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<char>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<char> '"
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
			data = j.at("data").get<char>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<char>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<char>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<char> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<char>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline IndexedDataTree<int, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<int>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<int> '"
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
			data = j.at("data").get<int>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<int>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<int>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<int> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<int>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline IndexedDataTree<float, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<float>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<float> '"
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
			data = j.at("data").get<float>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<float>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<float>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<float> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<float>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline IndexedDataTree<double, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<double>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<double> '"
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
			data = j.at("data").get<double>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<double>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<double>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<double> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<double>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline IndexedDataTree<std::complex<double>, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<std::complex<double>>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<std::complex<double>> '"
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
			std::string dataString = j.at("data").get<std::string>();
			std::stringstream ss(dataString);
			ss >> data;
//			data = j.at("data").get<std::complex<double>>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<std::complex<double>>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<std::complex<double>>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<std::complex<double>> '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexedDataTree<std::complex<double>>::IndexedDataTree()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename Data>
IndexedDataTree<Data, true>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<Data>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<Data> '"
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
			std::string dataString = j.at("data").get<std::string>();
			data = Data(dataString, mode);
//			data = j.at("data").get<Data>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<Data>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexedDataTree<Data>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<Data> '"
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
IndexedDataTree<Data, false>::IndexedDataTree(
	const std::string &serialization,
	Mode mode
){
	TBTKNotYetImplemented("IndexedDataTree<Data, false>");
/*	TBTKAssert(
		validate(serialization, "IndexedDataTree", mode),
		"IndexedDataTree<Data>::IndexedDataTree()",
		"Unable to parse string as IndexedDataTree<Data> '"
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
			std::string dataString = j.at("data").get<std::string>();
			data = Data(dataString, mode);
//			data = j.at("data").get<Data>();
			try{
				nlohmann::json children = j.at("children");
				for(
					nlohmann::json::iterator it = children.begin();
					it != children.end();
					++it
				){
					this->children.push_back(
						IndexedDataTree<Data>(
							it->dump(),
							mode
						)
					);
				}
			}
			catch(nlohmann::json::exception e){
				//It is valid to not have children.
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"IndexedDataTree<Data>::IndexedDataTree()",
				"Unable to parse string as"
				<< " IndexedDataTree<Data> '"
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
	}*/
}

template<typename Data, bool isSerializable>
IndexedDataTree<Data, isSerializable>::~IndexedDataTree(){
}

template<typename Data>
IndexedDataTree<Data, true>::~IndexedDataTree(){
}

template<typename Data>
IndexedDataTree<Data, false>::~IndexedDataTree(){
}

template<typename Data, bool isSerializable>
void IndexedDataTree<Data, isSerializable>::add(
	const Data &data,
	const Index &index
){
	add(data, index, 0);
}

template<typename Data>
void IndexedDataTree<Data, true>::add(
	const Data &data,
	const Index &index
){
	add(data, index, 0);
}

template<typename Data>
void IndexedDataTree<Data, false>::add(
	const Data &data,
	const Index &index
){
	add(data, index, 0);
}

template<typename Data, bool isSerializable>
void IndexedDataTree<Data, isSerializable>::add(
	const Data &data,
	const Index &index,
	unsigned int subindex
){
	if(subindex < index.getSize()){
		//If the current subindex is not the last, the Index is
		//propagated to the next node level.

		//Get current subindex
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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
void IndexedDataTree<Data, true>::add(
	const Data &data,
	const Index &index,
	unsigned int subindex
){
	if(subindex < index.getSize()){
		//If the current subindex is not the last, the Index is
		//propagated to the next node level.

		//Get current subindex
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if((unsigned int)currentIndex >= children.size())
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
void IndexedDataTree<Data, false>::add(
	const Data &data,
	const Index &index,
	unsigned int subindex
){
	if(subindex < index.getSize()){
		//If the current subindex is not the last, the Index is
		//propagated to the next node level.

		//Get current subindex
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if(currentIndex >= (int)children.size())
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

template<typename Data, bool isSerializable>
bool IndexedDataTree<Data, isSerializable>::get(Data &data, const Index &index) const{
	return get(data, index, 0);
}

template<typename Data>
bool IndexedDataTree<Data, true>::get(Data &data, const Index &index) const{
	return get(data, index, 0);
}

template<typename Data>
bool IndexedDataTree<Data, false>::get(Data &data, const Index &index) const{
	return get(data, index, 0);
}

template<typename Data, bool isSerializable>
bool IndexedDataTree<Data, isSerializable>::get(
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
bool IndexedDataTree<Data, true>::get(
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
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//Return false because the Index is not included.
		if((unsigned int)currentIndex >= children.size())
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
bool IndexedDataTree<Data, false>::get(
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
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//Return false because the Index is not included.
		if(currentIndex >= (int)children.size())
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
Data& IndexedDataTree<Data, true>::get(const Index &index){
	//Casting is safe because we do not guarantee that the IndexedDataTree
	//is not modified. Casting away const from the returned reference is
	//therefore not a violation of any promisse made by this function.
	//See also "Avoiding Duplication in const and Non-const Member
	//Function" in S. Meyers, Effective C++.
	return const_cast<Data&>(
		static_cast<const IndexedDataTree<Data, true>*>(
			this
		)->get(index, 0)
	);
}

template<typename Data>
const Data& IndexedDataTree<Data, true>::get(const Index &index) const{
	return get(index, 0);
}

template<typename Data>
Data& IndexedDataTree<Data, false>::get(const Index &index){
	//Casting is safe because we do not guarantee that the IndexedDataTree
	//is not modified. Casting away const from the returned reference is
	//therefore not a violation of any promisse made by this function.
	//See also "Avoiding Duplication in const and Non-const Member
	//Function" in S. Meyers, Effective C++.
	return const_cast<Data&>(
		static_cast<const IndexedDataTree<Data, false>*>(
			this
		)->get(index, 0)
	);
}

template<typename Data>
const Data& IndexedDataTree<Data, false>::get(const Index &index) const{
	return get(index, 0);
}

template<typename Data>
const Data& IndexedDataTree<Data, true>::get(
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
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//Throw ElementNotFoundException if the Index is not included.
		if((unsigned int)currentIndex >= children.size()){
			throw ElementNotFoundException(
				"IndexedDataTree()",
				TBTKWhere,
				"Tried to get element with Index '"
				+ index.toString() + "', but no such element"
				+ " exists.",
				""
			);
		}

		return children.at(currentIndex).get(index, subindex+1);
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
const Data& IndexedDataTree<Data, false>::get(
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
		int currentIndex = index.at(subindex);

		if(currentIndex == IDX_SEPARATOR){
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

		//Throw ElementNotFoundException if the Index is not included.
		if((unsigned int)currentIndex >= children.size()){
			throw ElementNotFoundException(
				"IndexedDataTree()",
				TBTKWhere,
				"Tried to get element with Index '"
				+ index.toString() + "', but no such element"
				+ " exists.",
				""
			);
		}

		return children.at(currentIndex).get(index, subindex+1);
	}
	else{
		//If the current subindex is the last, try to extract the data.
		//Return data if successful but throw ElementNotFoundException
		//if the data does not exist.
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
Index IndexedDataTree<Data, true>::getFirstIndex() const{
	Index index;
	getFirstIndex(index);

	return index;
}

template<typename Data>
Index IndexedDataTree<Data, false>::getFirstIndex() const{
	Index index;
	getFirstIndex(index);

	return index;
}

template<typename Data>
bool IndexedDataTree<Data, true>::getFirstIndex(Index &index) const{
	if(indexIncluded)
		return true;

	if(indexSeparator)
		index.push_back(IDX_SEPARATOR);

	for(unsigned int n = 0; n < children.size(); n++){
		index.push_back(n);
		if(children[n].getFirstIndex(index))
			return true;

		index.popBack();
	}

	if(indexSeparator)
		index.popBack();

	return false;
}

template<typename Data>
bool IndexedDataTree<Data, false>::getFirstIndex(Index &index) const{
	if(indexIncluded)
		return true;

	if(indexSeparator)
		index.push_back(IDX_SEPARATOR);

	for(unsigned int n = 0; n < children.size(); n++){
		index.push_back(n);
		if(children[n].getFirstIndex(index))
			return true;

		index.popBack();
	}

	if(indexSeparator)
		index.popBack();

	return false;
}

template<typename Data>
Index IndexedDataTree<Data, true>::getNextIndex(const Index &index) const{
	if(index.getSize() == 0)
		return Index();

	Index nextIndex;
	getNextIndex(index, nextIndex);

	return nextIndex;
}

template<typename Data>
Index IndexedDataTree<Data, false>::getNextIndex(const Index &index) const{
	if(index.getSize() == 0)
		return Index();

	Index nextIndex;
	getNextIndex(index, nextIndex);

	return nextIndex;
}

template<typename Data>
bool IndexedDataTree<Data, true>::getNextIndex(
	const Index &currentIndex,
	Index &nextIndex
) const{
	if(indexIncluded){
		if(currentIndex.equals(nextIndex))
			return false;

		return true;
	}

	if(indexSeparator)
		nextIndex.push_back(IDX_SEPARATOR);

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

	unsigned int startIndex = 0;
	if(hasSameIndexStructure)
		startIndex = currentIndex[nextIndex.getSize()];
	for(unsigned int n = startIndex; n < children.size(); n++){
		nextIndex.push_back(n);
		if(children[n].getNextIndex(currentIndex, nextIndex))
			return true;

		nextIndex.popBack();
	}

	if(indexSeparator)
		nextIndex.popBack();

	return false;
}

template<typename Data>
bool IndexedDataTree<Data, false>::getNextIndex(
	const Index &currentIndex,
	Index &nextIndex
) const{
	if(indexIncluded){
		if(currentIndex.equals(nextIndex))
			return false;

		return true;
	}

	if(indexSeparator)
		nextIndex.push_back(IDX_SEPARATOR);

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

	unsigned int startIndex = 0;
	if(hasSameIndexStructure)
		startIndex = currentIndex[nextIndex.getSize()];
	for(unsigned int n = startIndex; n < children.size(); n++){
		nextIndex.push_back(n);
		if(children[n].getNextIndex(currentIndex, nextIndex))
			return true;

		nextIndex.popBack();
	}

	if(indexSeparator)
		nextIndex.popBack();

	return false;
}

template<typename Data, bool isSerializable>
void IndexedDataTree<Data, isSerializable>::clear(){
	indexIncluded = false;
	children.clear();
}

template<typename Data>
void IndexedDataTree<Data, true>::clear(){
	indexIncluded = false;
	children.clear();
}

template<typename Data>
void IndexedDataTree<Data, false>::clear(){
	indexIncluded = false;
	children.clear();
}

template<typename Data, bool isSerializable>
unsigned int IndexedDataTree<Data, isSerializable>::getSizeInBytes() const{
	unsigned int size = sizeof(IndexedDataTree<Data>);
	for(unsigned int n = 0; n < children.size(); n++)
		size += children.at(n).getSizeInBytes();

	return size;
}

template<typename Data>
unsigned int IndexedDataTree<Data, true>::getSizeInBytes() const{
	unsigned int size = sizeof(IndexedDataTree<Data>);
	for(unsigned int n = 0; n < children.size(); n++)
		size += children.at(n).getSizeInBytes();

	return size;
}

template<typename Data>
unsigned int IndexedDataTree<Data, false>::getSizeInBytes() const{
	unsigned int size = sizeof(IndexedDataTree<Data>);
	for(unsigned int n = 0; n < children.size(); n++)
		size += children.at(n).getSizeInBytes();

	return size;
}

template<>
inline std::string IndexedDataTree<bool, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data;
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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

template<>
inline std::string IndexedDataTree<char, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data;
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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

template<>
inline std::string IndexedDataTree<int, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data;
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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

template<>
inline std::string IndexedDataTree<float, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data;
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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

template<>
inline std::string IndexedDataTree<double, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data;
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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

template<>
inline std::string IndexedDataTree<std::complex<double>, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		std::stringstream ss;
		ss << "(" << real(data) << "," << imag(data) << ")";
		j["data"] = ss.str();
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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
std::string IndexedDataTree<Data, true>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data.serialize(mode);
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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
std::string IndexedDataTree<Data, false>::serialize(Mode mode) const{
	TBTKNotYetImplemented("IndexedDataTree<Data, false>");
/*	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexedDataTree";
		j["indexIncluded"] = indexIncluded;
		j["indexSeparator"] = indexSeparator;
		j["data"] = data.serialize(mode);
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(
					children.at(n).serialize(mode)
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
	}*/
}

template<typename Data>
typename IndexedDataTree<Data, true>::Iterator IndexedDataTree<Data, true>::begin(){
	return Iterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, true>::ConstIterator IndexedDataTree<Data, true>::begin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, false>::Iterator IndexedDataTree<Data, false>::begin(){
	return Iterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, false>::ConstIterator IndexedDataTree<Data, false>::begin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, true>::ConstIterator IndexedDataTree<Data, true>::cbegin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, false>::ConstIterator IndexedDataTree<Data, false>::cbegin() const{
	return ConstIterator(this);
}

template<typename Data>
typename IndexedDataTree<Data, true>::Iterator IndexedDataTree<Data, true>::end(){
	return Iterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data, true>::ConstIterator IndexedDataTree<Data, true>::end() const{
	return ConstIterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data, false>::Iterator IndexedDataTree<Data, false>::end(){
	return Iterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data, false>::ConstIterator IndexedDataTree<Data, false>::end() const{
	return ConstIterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data, true>::ConstIterator IndexedDataTree<Data, true>::cend() const{
	return ConstIterator(this, true);
}

template<typename Data>
typename IndexedDataTree<Data, false>::ConstIterator IndexedDataTree<Data, false>::cend() const{
	return ConstIterator(this, true);
}

template<typename Data> template<bool isConstIterator>
void IndexedDataTree<Data, true>::_Iterator<isConstIterator>::operator++(){
	currentIndex = indexedDataTree->getNextIndex(currentIndex);
}

template<typename Data> template<bool isConstIterator>
void IndexedDataTree<Data, false>::_Iterator<isConstIterator>::operator++(){
	currentIndex = indexedDataTree->getNextIndex(currentIndex);
}

template<typename Data> template<bool isConstIterator>
typename IndexedDataTree<Data, true>::template _Iterator<
	isConstIterator
>::DataReferenceType IndexedDataTree<Data, true>::_Iterator<
	isConstIterator
>::operator*(){
	return indexedDataTree->get(currentIndex);
}

template<typename Data> template<bool isConstIterator>
typename IndexedDataTree<Data, false>::template _Iterator<
	isConstIterator
>::DataReferenceType IndexedDataTree<Data, false>::_Iterator<
	isConstIterator
>::operator*(){
	return indexedDataTree->get(currentIndex);
}

template<typename Data> template<bool isConstIterator>
bool IndexedDataTree<Data, true>::_Iterator<isConstIterator>::operator==(
	const IndexedDataTree<Data, true>::_Iterator<isConstIterator> &rhs
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
bool IndexedDataTree<Data, false>::_Iterator<isConstIterator>::operator==(
	const IndexedDataTree<Data, false>::_Iterator<isConstIterator> &rhs
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
bool IndexedDataTree<Data, true>::_Iterator<isConstIterator>::operator!=(
	const IndexedDataTree<Data, true>::_Iterator<isConstIterator> &rhs
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
bool IndexedDataTree<Data, false>::_Iterator<isConstIterator>::operator!=(
	const IndexedDataTree<Data, false>::_Iterator<isConstIterator> &rhs
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
const Index& IndexedDataTree<Data, true>::_Iterator<isConstIterator>::getCurrentIndex(
) const{
	return currentIndex;
}

template<typename Data> template<bool isConstIterator>
const Index& IndexedDataTree<Data, false>::_Iterator<isConstIterator>::getCurrentIndex(
) const{
	return currentIndex;
}

template<typename Data> template<bool isConstIterator>
IndexedDataTree<Data, true>::_Iterator<isConstIterator>::_Iterator(
	IndexedDataTreePointerType indexedDataTree,
	bool end
){
	this->indexedDataTree = indexedDataTree;
	if(end)
		currentIndex = Index();
	else
		currentIndex = indexedDataTree->getFirstIndex();
}

template<typename Data> template<bool isConstIterator>
IndexedDataTree<Data, false>::_Iterator<isConstIterator>::_Iterator(
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
