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

#include "TBTK/IndexException.h"
#include "TBTK/IndexTree.h"
#include "TBTK/TBTKMacros.h"

#include <sstream>

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{

IndexTree::IndexTree(){
	indexIncluded = false;
	wildcardIndex = false;
	wildcardType = 0;
	indexSeparator = false;
	linearIndex = -1;
	size = -1;
}

IndexTree::IndexTree(const string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "IndexTree", mode),
		"IndexTree:IndexTree()",
		"Unable to parse string as IndexTree '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);
		vector<string> elements = split(content, mode);

		unsigned int counter = 0;
		while(
			hasID(elements.at(counter), mode)
			&& getID(
				elements.at(counter), mode
			).compare("IndexTree") == 0
		){
			children.push_back(
				IndexTree(elements.at(counter), mode)
			);
			counter++;
		}
		if(counter == 0)
			counter++;
		stringstream ss;
		ss.str(elements.at(counter++));
		ss >> indexIncluded;
		ss.clear();
		ss.str(elements.at(counter++));
		ss >> wildcardIndex;
		ss.clear();
		ss.str(elements.at(counter++));
		ss >> wildcardType;
		ss.clear();
		ss.str(elements.at(counter++));
		ss >> indexSeparator;
		ss.clear();
		ss.str(elements.at(counter++));
		ss >> linearIndex;
		ss.clear();
		ss.str(elements.at(counter++));
		ss >> size;

		break;
	}
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			try{
				nlohmann::json c = j.at("children");
				for(nlohmann::json::iterator it = c.begin(); it != c.end(); ++it){
					children.push_back(
						IndexTree(it->dump(), mode)
					);
				}
			}
			catch(nlohmann::json::exception &e){
				//It is valid to not have children.
			}
			indexIncluded = j.at("indexIncluded").get<bool>();
			wildcardIndex = j.at("wildcardIndex").get<bool>();
			wildcardType = j.at("wildcardType").get<int>();
			indexSeparator = j.at("indexSeparator").get<bool>();
			linearIndex = j.at("linearIndex").get<int>();
			size = j.at("size").get<int>();
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"IndexTree::IndexTree()",
				"Unable to parse string as IndexTree '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"IndexTree::IndexTree()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

IndexTree::~IndexTree(){
}

bool operator==(const IndexTree &lhs, const IndexTree &rhs){
	if(lhs.children.size() != rhs.children.size())
		return false;

	for(unsigned int n = 0; n < lhs.children.size(); n++){
		if(lhs.children[n] != rhs.children[n])
			return false;
	}

	if(lhs.indexIncluded != rhs.indexIncluded)
		return false;

	if(lhs.wildcardIndex != rhs.wildcardIndex)
		return false;

	if(lhs.wildcardType != rhs.wildcardType)
		return false;

	if(lhs.indexSeparator != rhs.indexSeparator)
		return false;

	if(lhs.linearIndex != rhs.linearIndex)
		return false;

	if(lhs.size != rhs.size)
		return false;

	return true;
}

void IndexTree::add(const Index &index){
	add(index, 0);
}

void IndexTree::add(const Index &index, unsigned int subindex){
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
					"IndexTree::add()",
					"Invalid index '" << index.toString()
					<< "'. Another Index has already been"
					<< " added to the tree that has a"
					<< " conflicting index at the index"
					<< " separator at subindex "
					<< subindex << "'.",
					"Note that a separation point between"
					<< " two indices counts as a subindex."
				);
			}

			indexSeparator = false;
			add(index, subindex+1);
			indexSeparator = true;
			return;
		}
		else{
			TBTKAssert(
				!indexSeparator,
				"IndexTree::add()",
				"Invalid index '" << index.toString() << "'."
				<< " Another Index has already been added to"
				<< " the tree that has a conflicting index"
				<< " separator at subindex '"
				<< subindex << "'.",
				"Note that a separation point between two"
				<< " indices counts as a subindex."
			);
		}

		if(currentIndex < 0){
			if(!wildcardIndex){
				if(children.size() > 0){
					Streams::err
						<< "Error in IndexTree::add(). "
						<< "Tried to add index with "
						<< "wild card in subindex "
						<< "position for which non "
						<< "wild card indices already "
						<< "have been added:\n";
					Streams::err << index.toString() << "\n";
					exit(1);
				}
				wildcardIndex = true;
				wildcardType = currentIndex;
			}
			if(currentIndex != wildcardType){
				Streams::err << "Error in IndexTree::add(). "
					<< "Tried to add index with wild card "
					<< "in subindex position for which "
					<< "wild card with different wild card"
					<< " type already have been added:\n";
				Streams::err << index.toString() << "\n";
				exit(1);
			}
			currentIndex = 0;
		}
		else if(wildcardIndex){
			Streams::err << "Error in IndexTree::add(). Unable to "
				<< "add index because an index with a wild "
				<< "card in subindex position " << subindex
				<< " already have been added:\n";
			Streams::err << index.toString() << "\n";
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
			Streams::err << "Error in IndexTree::add(), index "
				<< "incompatible with previously added index "
				<< "at subindex " << subindex << ":\n";
			Streams::err << index.toString() << "\n";
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
			Streams::err << "Error in IndexTree::add(), index "
				<< "incompatible with previously added index "
				<< "at subindex " << subindex << ":\n";
			Streams::err << index.toString() << "\n";
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

int IndexTree::getLinearIndex(
	const Index &index,
	SearchMode searchMode,
	bool returnNegativeForMissingIndex
) const{
	return getLinearIndex(
		index,
		0,
		searchMode,
		returnNegativeForMissingIndex
	);
}

int IndexTree::getLinearIndex(
	const Index &index,
	unsigned int subindex,
	SearchMode searchMode,
	bool returnNegativeForMissingIndex
) const{
	switch(searchMode){
	case SearchMode::StrictMatch:
		break;
	case SearchMode::MatchWildcards:
		break;
	default:
		TBTKExit(
			"IndexTree::getLinearIndex()",
			"Unknown SearchMode.",
			"This should never happen, contact the developer."
		);
	}

	if(subindex < index.getSize()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		Subindex currentIndex = index.at(subindex);

		if(currentIndex.isIndexSeparator()){
			if(indexSeparator){
				return getLinearIndex(
					index,
					subindex+1,
					searchMode,
					returnNegativeForMissingIndex
				);
			}
			else{
				TBTKExit(
					"IndexTree::getLinearIndex()",
					"Invalid Index. Found IDX_SEPARATOR at"
					<< " subindex '" << subindex << "',"
					<< " but the node is not an index "
					<< " separator.",
					""
				);
			}
		}

		if(currentIndex < 0){
			if(wildcardIndex){
				TBTKAssert(
					currentIndex == wildcardType,
					"IndexTree::getLinearIndex()",
					"Invalid wildcard type. The Index '"
					<< index.toString() << "' has a"
					<< " wildcard in subindex '"
					<< subindex << "' which is different"
					<< " from the wildcard type encoded in"
					<< " the IndexTree.",
					""
				);
				currentIndex = 0;
			}
			else{
				Streams::err << "Error in "
					<< "IndexTree::getLinearIndex(). "
					<< "Subindex " << subindex << " should"
					<< " not be a wild card index:\n";
				Streams::err << index.toString() << "\n";
				exit(1);
			}
		}
		else if(wildcardIndex){
			if(searchMode == SearchMode::MatchWildcards){
				currentIndex = 0;
			}
			else{
				Streams::err << "Error in IndexTree::getLinearIndex()."
					<< " Subindex " << subindex << " has to be a "
					<< "wild card index:\n";
				Streams::err << index.toString();
				exit(1);
			}
		}
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			if(returnNegativeForMissingIndex){
				return -1;
			}
			else{
				throw IndexException(
					"IndexTree::getLinearIndex()",
					TBTKWhere,
					"Index not included in the IndexTree '"
					+ index.toString() + "'.",
					""
				);
			}
		}
		//Continue to the next node level.
		return children.at(currentIndex).getLinearIndex(
			index,
			subindex+1,
			searchMode,
			returnNegativeForMissingIndex
		);
	}
	else{
		//If the current subindex is the last, return linear index.
		if(indexIncluded){
			return linearIndex;
		}
		else if(returnNegativeForMissingIndex){
			return -1;
		}
		else{
			throw IndexException(
				"IndexTree::getLinearIndex()",
				TBTKWhere,
				"Index not included in the IndexTree '"
				+ index.toString() + "'.",
				""
			);
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

	vector<Subindex> indices;
	getPhysicalIndex(linearIndex, indices);

	return Index(indices);
}

bool IndexTree::contains(const Index &index){
	try{
		getLinearIndex(index);
		return true;
	}
	catch(const IndexException &e){
		return false;
	}
}

vector<unsigned int> IndexTree::getSubindicesMatching(
	int subindexValue,
	const Index &index,
	SearchMode searchMode
) const{
	int linearIndex = getLinearIndex(index, searchMode);
	Index physicalIndex = getPhysicalIndex(linearIndex);
	vector<unsigned int> matches;
	for(unsigned int n = 0; n < physicalIndex.getSize(); n++)
		if(physicalIndex.at(n) == subindexValue)
			matches.push_back(n);

	return matches;
}

vector<Index> IndexTree::getIndexList(const Index &pattern) const{
	vector<Index> indexList;

	for(
		IndexTree::ConstIterator iterator = cbegin();
		iterator != cend();
		++iterator
	){
		Index index = *iterator;
		if(index.equals(pattern, true)){
			//Index::equals() has here determined that the Indices
			//are equal up to IDX_ALL wildcards. However, we do not
			//want to add 'index' if the match is because 'index'
			//has an IDX_ALL wildcard in a subindex where 'pattern'
			//has some other value. Only IDX_ALL wildcards in the
			//'pattern' Index should be treated as wildcards.
			bool equalityIsValid = true;
			for(unsigned int n = 0; n < index.getSize(); n++){
				if(index[n].isWildcard()){
					if(!pattern[n].isWildcard()){
						equalityIsValid = false;
						break;
					}
				}
			}

			if(equalityIsValid)
				indexList.push_back(index);
		}
	}

	return indexList;
}

bool IndexTree::equals(const IndexTree &indexTree) const{
	ConstIterator iterator0 = cbegin();
	ConstIterator iterator1 = indexTree.cbegin();
	while(iterator0 != cend() && iterator1 != indexTree.cend()){
		if(!(*iterator0).equals((*iterator1)))
			return false;

		++iterator0;
		++iterator1;
	}

	return (iterator0 == cend() && iterator1 == indexTree.cend());
}

void IndexTree::getPhysicalIndex(int linearIndex, vector<Subindex> &indices) const{
	if(this->linearIndex != -1)
		return;

	if(indexSeparator)
		indices.push_back(IDX_SEPARATOR);

	for(unsigned int n = 0; n < children.size(); n++){
		int min = children.at(n).getMinIndex();
		int max = children.at(n).getMaxIndex();

		if(min == -1)
			continue;

		if(min <= linearIndex && linearIndex <= max){
			if(wildcardIndex)
				indices.push_back(wildcardType);
			else
				indices.push_back(n);

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

string IndexTree::toString() const{
	stringstream stream;
	stream << "IndexTree";
	for(ConstIterator iterator = cbegin(); iterator != cend(); ++iterator)
		stream << "\n\t" << *iterator;

	return stream.str();
}

string IndexTree::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "IndexTree(";
		for(unsigned int n = 0; n < children.size(); n++){
			if(n != 0)
				ss << ",";
			ss << children.at(n).serialize(mode);
		}
		ss << "," << Serializable::serialize(indexIncluded, mode);
		ss << "," << Serializable::serialize(wildcardIndex, mode);
		ss << "," << Serializable::serialize(wildcardType, mode);
		ss << "," << Serializable::serialize(indexSeparator, mode);
		ss << "," << Serializable::serialize(linearIndex, mode);
		ss << "," << Serializable::serialize(size, mode);
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "IndexTree";
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				nlohmann::json::parse(children.at(n).serialize(mode))
			);
		}
		j["indexIncluded"] = indexIncluded;
		j["wildcardIndex"] = wildcardIndex;
		j["wildcardType"] = wildcardType;
		j["indexSeparator"] = indexSeparator;
		j["linearIndex"] = linearIndex;
		j["size"] = size;

		return j.dump();
	}
	default:
		TBTKExit(
			"IndexTree:IndexTree()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
