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

/** @file HoppingAmplitudeTree.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/HoppingAmplitudeTree.h"
#include "TBTK/Streams.h"

#include <algorithm>

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

const HoppingAmplitudeTree HoppingAmplitudeTree::emptyTree;

HoppingAmplitudeTree::HoppingAmplitudeTree(){
	basisIndex = -1;
	basisSize = -1;
	isPotentialBlockSeparator = true;
}

HoppingAmplitudeTree::HoppingAmplitudeTree(const vector<unsigned int> &capacity){
	basisIndex = -1;
	basisSize = -1;
	isPotentialBlockSeparator = true;

	if(capacity.size() != 0){
		vector<unsigned int> childCapacity;
		for(unsigned int n = 1; n < capacity.size(); n++)
			childCapacity.push_back(capacity[n]);

		children.reserve(capacity[0]);
		for(unsigned int n = 0; n < capacity[0]; n++){
			children.push_back(
				HoppingAmplitudeTree(childCapacity)
			);
		}
	}
}

HoppingAmplitudeTree::HoppingAmplitudeTree(
	const string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "HoppingAmplitudeTree", mode),
		"HoppingAmplitudeTree::HoppingAmplitudeTree()",
		"Unable to parse string as HoppingAmplitudeTree '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		stringstream ss;
		ss.str(elements.at(0));
		ss >> basisIndex;
		ss.clear();
		ss.str(elements.at(1));
		ss >> basisSize;
		ss.clear();
		ss.str(elements.at(2));
		ss >> isPotentialBlockSeparator;

		unsigned int counter = 3;
		while(
			counter < elements.size() &&
			getID(
				elements.at(counter),
				mode
			).compare("HoppingAmplitude") == 0
		){
			hoppingAmplitudes.push_back(
				HoppingAmplitude(
					elements.at(counter),
					mode
				)
			);
			counter++;
		}
		for(unsigned int n = counter; n < elements.size(); n++){
			TBTKAssert(
				getID(
					elements.at(n),
					mode
				).compare("HoppingAmplitudeTree") == 0,
				"HoppingAmplitudeTree::HoppingAmplitudeTree()",
				"Unable to parse string as"
				<< " HoppingAmplitudeTree. Expected"
				<< " 'HoppingAmplitudeTree' but found '"
				<< getID(elements.at(n), mode) << "'.",
				""
			)
			children.push_back(HoppingAmplitudeTree(elements.at(n), mode));
		}

		break;
	}
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			basisIndex = j.at("basisIndex").get<int>();
			basisSize = j.at("basisSize").get<int>();
			isPotentialBlockSeparator = j.at(
				"isPotentialBlockSeparator"
			).get<bool>();
			try{
/*				json has = j.at(
					"hoppingAmplitudes"
				).get<json::array>();*/
				json has = j.at("hoppingAmplitudes");
				for(json::iterator it = has.begin(); it != has.end(); ++it){
					hoppingAmplitudes.push_back(
						HoppingAmplitude(it->dump(), mode)
					);
				}
			}
			catch(json::exception e){
				//It is valid to not have HoppingAmplitudes.
			}

			try{
/*				json c = j.at(
					"children"
				).get<json>();*/
				json c = j.at("children");
				for(json::iterator it = c.begin(); it != c.end(); ++it){
					children.push_back(
						HoppingAmplitudeTree(it->dump(), mode)
					);
				}
			}
			catch(json::exception e){
				//It is valid to not have children.
			}
		}
		catch(json::exception e){
			TBTKExit(
				"HoppingAmplitudeTree::HoppingAmplitudeTree()",
				"Unable to parse string as"
				<< " HoppingAmplitudeTree '" << serialization
				<< "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"HoppingAmplitudeTree::HoppingAmplitudeTree()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

HoppingAmplitudeTree::~HoppingAmplitudeTree(){
}

vector<Index> HoppingAmplitudeTree::getIndexList(const Index &pattern) const{
	vector<Index> indexList;

/*	Iterator it = begin();
	const HoppingAmplitude *ha;
	while((ha = it.getHA())){
//		if(ha->fromIndex.equals(pattern, true)){
		if(ha->getFromIndex().equals(pattern, true)){
			if(
				indexList.size() == 0
				|| !indexList.back().equals(ha->getFromIndex(), false)
//				|| !indexList.back().equals(ha->fromIndex, false)
			){
				indexList.push_back(ha->getFromIndex());
//				indexList.push_back(ha->fromIndex);
			}
		}

		it.searchNextHA();
	}*/
	for(
		Iterator iterator = begin();
		iterator != end();
		++iterator
	){
//		if(ha->fromIndex.equals(pattern, true)){
		if((*iterator).getFromIndex().equals(pattern, true)){
			if(
				indexList.size() == 0
				|| !indexList.back().equals((*iterator).getFromIndex(), false)
//				|| !indexList.back().equals(ha->fromIndex, false)
			){
				indexList.push_back((*iterator).getFromIndex());
//				indexList.push_back(ha->fromIndex);
			}
		}
	}

	return indexList;
}

void HoppingAmplitudeTree::print(){
	print(0);
}

void HoppingAmplitudeTree::print(unsigned int subindex){
	for(unsigned int n = 0; n < subindex; n++)
		Streams::out << "\t";
	Streams::out << basisIndex << ":" << hoppingAmplitudes.size() << "\n";
	for(unsigned int n = 0; n < children.size(); n++)
		children.at(n).print(subindex + 1);
}

void HoppingAmplitudeTree::add(HoppingAmplitude ha){
	_add(ha, 0);
}

void HoppingAmplitudeTree::_add(HoppingAmplitude &ha, unsigned int subindex){
//	if(subindex < ha.fromIndex.size()){
	if(subindex < ha.getFromIndex().getSize()){
		//If the current subindex is not the last, the HoppingAmplitude
		//is propagated to the next node level.

		//Get current subindex
//		int currentIndex = ha.fromIndex.at(subindex);
		int currentIndex = ha.getFromIndex().at(subindex);
		//Error detection:
		//Negative indices not allowed.
		TBTKAssert(
			currentIndex >= 0,
			"HoppingAmplitude:_add()",
			"Invalid Index. Only indices with non-negative"
			<< " subindices can be added. But the from-Index "
			<< ha.getFromIndex().toString() << " has a negative"
			<< " subindex in position '" << subindex << "'.",
			""
		);
		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if(currentIndex >= (int)children.size()){
			for(int n = children.size(); n <= currentIndex; n++){
				children.push_back(HoppingAmplitudeTree());
			}
		}
		//Error detection:
		//If a HoppingAmplitude is found on this level, another
		//HoppingAmplitude with fewer subindices than the current
		//HoppingAmplitude have previously been added to this node.
		//This is an error because different number of subindices is
		//only allowed if the HoppingAmplitudes differ in one of their
		//common indices.
		TBTKAssert(
			hoppingAmplitudes.size() == 0,
			"HoppingAmplitudeTree::_add()",
			"Incompatible HoppingAmplitudes. Tried to add a"
			<< " HoppingAmplitude with from-Index "
			<< ha.getFromIndex().toString() << ", but"
			<< " HoppingAmplitude with from-Index "
			<< hoppingAmplitudes[0].getFromIndex().toString()
			<< " has already been added.",
			""
		);
/*		if(hoppingAmplitudes.size() != 0){
			Streams::err << "Error, incompatible amplitudes:\n";
			ha.print();
			hoppingAmplitudes.at(0).print();
			exit(1);
		}*/
		//Ensure isPotentialBlockSeparator is set to false in case the
		//'toIndex' and the 'fromIndex' differs in the subindex
		//corresponding to this HoppingAmplitudeTree level.
//		if(ha.toIndex.size() <= subindex || currentIndex != ha.toIndex.at(subindex))
		if(ha.getToIndex().getSize() <= subindex || currentIndex != ha.getToIndex().at(subindex))
			isPotentialBlockSeparator = false;
		//Propagate to the next node level.
		children.at(currentIndex)._add(ha, subindex+1);
	}
	else{
		//If the current subindex is the last, the HoppingAmplitude
		//is added to this node.

		//Error detection:
		//If childen is non-zeros, another HoppingAmplitude with more
		//indices have already been added to this node. This is an
		//error because different number of subindices is only allowed
		//if the HoppingAmplitudes differ in one of their common
		//indices.
		TBTKAssert(
			children.size() == 0,
			"HoppingAmplitudeTree:_add(),",
			"Incompatible HoppingAmplitudes. Tried to add a"
			<< " HoppingAmplitude with from-Index "
			<< ha.getFromIndex().toString() << ", but"
			<< " HoppingAmplitude with from-Index "
			<< getFirstHA().getFromIndex().toString() << " has"
			<< " already been added.",
			""
		);
/*		if(children.size() != 0){
			Streams::err << "Error, incompatible amplitudes:\n";
			ha.print();
			getFirstHA().print();
			exit(1);
		}*/
		//Add HoppingAmplitude to node.
		hoppingAmplitudes.push_back(ha);
	}
}

const HoppingAmplitudeTree* HoppingAmplitudeTree::getSubTree(
	const Index &subspace
) const{
	for(unsigned int n = 0; n < subspace.getSize(); n++){
		if(subspace.at(n) < 0){
			TBTKExit(
				"HoppingAmplitudeTree::getSubTree()",
				"Invalid subspace index '" << subspace.toString() << "'.",
				"Subspace indices cannot have negative subindices."
			);
		}
	}

	return getSubTree(subspace, 0);
}

const HoppingAmplitudeTree* HoppingAmplitudeTree::getSubTree(
	const Index &subspace,
	unsigned int subindex
) const{
	if(subindex == subspace.getSize()){
		//Correct node reached

		return this;
	}

	if((unsigned int)subspace.at(subindex) < children.size()){
		return children.at(subspace.at(subindex)).getSubTree(
			subspace,
			subindex+1
		);
	}
	else{
/*		TBTKExit(
			"HoppingAmplitudeTree::getSubTree()",
			"Subspace index '" << subspace.toString() << "' does"
			<< " not exist.",
			""
		);*/
		return &emptyTree;
	}
}

bool HoppingAmplitudeTree::isProperSubspace(const Index &subspace) const{
	for(unsigned int n = 0; n < subspace.getSize(); n++){
		if(subspace.at(n) < 0){
			TBTKExit(
				"HoppingAmplitudeTree::getSubTree()",
				"Invalid subspace index '"
				<< subspace.toString() << "'.",
				"Subspace indices cannot have negative"
				<< " subindices."
			);
		}
	}

	return _isProperSubspace(subspace, 0);
}

bool HoppingAmplitudeTree::_isProperSubspace(
	const Index &subspace,
	unsigned int subindex
) const{
	if(subindex+1 == subspace.getSize())
		return isPotentialBlockSeparator;

	if(isPotentialBlockSeparator){
		if((unsigned int)subspace.at(subindex) < children.size()){
			return children[subspace[subindex]]._isProperSubspace(
				subspace,
				subindex+1
			);
		}
		else{
/*			TBTKExit(
				"HoppingAmplitudeTree::isProperSubspace()",
				"Subspace index '" << subspace.toString()
				<< "' does not exist.",
				""
			);*/
			//The subspace is empty and getSubTree will return
			//HoppingAmaplitudeTree::emptyTree. The empty subspace
			//is considered a proper subspace.
			return true;
		}
	}
	else{
		return false;
	}
}

IndexTree HoppingAmplitudeTree::getSubspaceIndices() const{
	IndexTree blockIndices;
	if(isPotentialBlockSeparator)
		getBlockIndices(blockIndices, Index());
//		getBlockIndices(blockIndices, Index({}));
	blockIndices.generateLinearMap();

	return blockIndices;
}

void HoppingAmplitudeTree::getBlockIndices(
	IndexTree &blockIndices,
	Index index
) const{
	if(children.size() > 0 && isPotentialBlockSeparator){
		for(unsigned int n = 0; n < children.size(); n++){
			index.push_back(n);
			children.at(n).getBlockIndices(blockIndices, index);
			index.popBack();
		}
	}
	else if(children.size() > 0 || basisIndex != -1){
		blockIndices.add(index);
	}
}

const std::vector<
	HoppingAmplitude
>& HoppingAmplitudeTree::getHoppingAmplitudes(
	Index index
) const{
	return _getHoppingAmplitudes(index, 0);
}

const std::vector<
	HoppingAmplitude
>& HoppingAmplitudeTree::_getHoppingAmplitudes(
	Index index,
	unsigned int subindex
) const{
	if(subindex < index.getSize()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		int currentIndex = index.at(subindex);
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			Streams::err << "Error, index out of bound: ";
			index.print();
			exit(1);
		}
		//Continue to the next node level.
		return children.at(currentIndex)._getHoppingAmplitudes(
			index,
			subindex+1
		);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return hoppingAmplitudes;
	}
}

int HoppingAmplitudeTree::getBasisIndex(const Index &index) const{
	return _getBasisIndex(index, 0);
}

int HoppingAmplitudeTree::_getBasisIndex(const Index &index, unsigned int subindex) const{
	if(subindex < index.getSize()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		int currentIndex = index.at(subindex);
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			Streams::err << "Error, index out of bound: ";
			index.print();
			exit(1);
		}
		//Continue to the next node level.
		return children.at(currentIndex)._getBasisIndex(index, subindex+1);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return basisIndex;
	}
}

Index HoppingAmplitudeTree::getPhysicalIndex(int basisIndex) const{
	TBTKAssert(
		basisIndex >= 0 && basisIndex < this->basisSize,
		"HoppingAmplitudeTree::getPhysicalIndex()",
		"Hilbert space index out of bound.",
		""
	);

	vector<int> indices;
	_getPhysicalIndex(basisIndex, &indices);

	return Index(indices);
}

void HoppingAmplitudeTree::_getPhysicalIndex(
	int basisIndex,
	vector<int> *indices
) const{
	if(this->basisIndex != -1)
		return;

	for(unsigned int n = 0; n < children.size(); n++){
		int min = children.at(n).getMinIndex();
		int max = children.at(n).getMaxIndex();

		if(min == -1)
			continue;

		if(min <= basisIndex && basisIndex <= max){
			indices->push_back(n);
			children.at(n)._getPhysicalIndex(basisIndex, indices);
			break;
		}
	}
}

int HoppingAmplitudeTree::getMinIndex() const{
	if(basisIndex != -1)
		return basisIndex;

	int min = -1;
	for(unsigned int n = 0; n < children.size(); n++){
		min = children.at(n).getMinIndex();
		if(min != -1)
			break;
	}

	return min;
}

int HoppingAmplitudeTree::getMaxIndex() const{
	if(basisIndex != -1)
		return basisIndex;

	int max = -1;
	for(int n = children.size()-1; n >= 0; n--){
		max = children.at(n).getMaxIndex();
		if(max != -1)
			break;
	}

	return max;
}

void HoppingAmplitudeTree::generateBasisIndices(){
	basisSize = generateBasisIndices(0);
}

int HoppingAmplitudeTree::generateBasisIndices(int i){
	if(children.size() == 0){
		if(hoppingAmplitudes.size() != 0){
			basisIndex = i;
			return i + 1;
		}
		else{
			return i;
		}
	}

	for(unsigned int n = 0; n < children.size(); n++){
		i = children.at(n).generateBasisIndices(i);
	}

	return i;
}

class SortHelperClass{
public:
	static HoppingAmplitudeTree *rootNode;
	inline bool operator() (const HoppingAmplitude& ha1, const HoppingAmplitude& ha2){
/*		int basisIndex1 = rootNode->getBasisIndex(ha1.toIndex);
		int basisIndex2 = rootNode->getBasisIndex(ha2.toIndex);*/
		int basisIndex1 = rootNode->getBasisIndex(ha1.getToIndex());
		int basisIndex2 = rootNode->getBasisIndex(ha2.getToIndex());
		if(basisIndex1 < basisIndex2)
			return true;
		else
			return false;
	}
};

HoppingAmplitudeTree *SortHelperClass::rootNode = NULL;

void HoppingAmplitudeTree::sort(HoppingAmplitudeTree *rootNode){
	if(hoppingAmplitudes.size() != 0){
		SortHelperClass::rootNode = rootNode;
		std::sort(hoppingAmplitudes.begin(), hoppingAmplitudes.end(), SortHelperClass());
	}
	else if(children.size() != 0){
		for(unsigned int n = 0; n < children.size(); n++)
			children.at(n).sort(rootNode);
	}
}

string HoppingAmplitudeTree::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "HoppingAmplitudeTree(";
		ss << Serializable::serialize(basisIndex, mode);
		ss << "," << Serializable::serialize(basisSize, mode);
		ss << "," << Serializable::serialize(
			isPotentialBlockSeparator,
			mode
		);
		for(unsigned int n = 0; n < hoppingAmplitudes.size(); n++){
			ss << ",";
			ss << hoppingAmplitudes.at(n).serialize(mode);
		}
		for(unsigned int n = 0; n < children.size(); n++){
			ss << ",";
			ss << children.at(n).serialize(mode);
		}

		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		json j;
		j["id"] = "HoppingAmplitudeTree";
		j["basisIndex"] = basisIndex;
		j["basisSize"] = basisSize;
		j["isPotentialBlockSeparator"] = isPotentialBlockSeparator;
		for(unsigned int n = 0; n < hoppingAmplitudes.size(); n++){
			j["hoppingAmplitudes"].push_back(
				json::parse(
					hoppingAmplitudes.at(n).serialize(
						Serializable::Mode::JSON
					)
				)
			);
		}
		for(unsigned int n = 0; n < children.size(); n++){
			j["children"].push_back(
				json::parse(
					children.at(n).serialize(
						Serializable::Mode::JSON
					)
				)
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"HoppingAmplitudeTree::serialize()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

/*HoppingAmplitudeTree::Iterator::Iterator(
	const HoppingAmplitudeTree::Iterator &iterator
){
	tree = iterator.tree;
	currentIndex = iterator.currentIndex;
	currentHoppingAmplitude = iterator.currentHoppingAmplitude;
}

HoppingAmplitudeTree::Iterator::Iterator(
	HoppingAmplitudeTree::Iterator &&iterator
){
	tree = iterator.tree;
	currentIndex = std::move(iterator.currentIndex);
	currentHoppingAmplitude = iterator.currentHoppingAmplitude;
}*/

HoppingAmplitudeTree::Iterator::Iterator(
	const HoppingAmplitudeTree *tree,
	bool end
){
	if(end){
		this->tree = tree;
		if(tree->children.size() == 0){
			currentHoppingAmplitude = -1;
		}
		else{
			currentIndex.push_back(tree->children.size());
			currentHoppingAmplitude = -1;
		}
	}
	else{
		this->tree = tree;
		if(tree->children.size() == 0){
			//Handle the special case when the data is stored on the head
			//node. Can for example be the case when iterating over a
			//single leaf node.
			currentHoppingAmplitude = -1;
			searchNext(tree, -1);
		}
		else{
			currentIndex.push_back(0);
			currentHoppingAmplitude = -1;
			searchNext(tree, 0);
		}
	}
}

/*HoppingAmplitudeTree::Iterator& HoppingAmplitudeTree::Iterator::operator=(
	const HoppingAmplitudeTree::Iterator &rhs
){
	if(this != &rhs){
		tree = rhs.tree;
		currentIndex = rhs.currentIndex;
		currentHoppingAmplitude = rhs.currentHoppingAmplitude;
	}

	return *this;
}

HoppingAmplitudeTree::Iterator& HoppingAmplitudeTree::Iterator::operator=(
	HoppingAmplitudeTree::Iterator &&rhs
){
	if(this != &rhs){
		tree = rhs.tree;
		currentIndex = std::move(rhs.currentIndex);
		currentHoppingAmplitude = rhs.currentHoppingAmplitude;
	}

	return *this;
}

void HoppingAmplitudeTree::Iterator::reset(){
	currentIndex.clear();
	if(tree->children.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		currentHoppingAmplitude = -1;
		searchNext(tree, -1);
	}
	else{
		currentIndex.push_back(0);
		currentHoppingAmplitude = -1;
		searchNext(tree, 0);
	}
}

void HoppingAmplitudeTree::Iterator::searchNextHA(){
	if(tree->children.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		searchNext(tree, -1);
	}
	else{
		searchNext(tree, 0);
	}
}*/

void HoppingAmplitudeTree::Iterator::operator++(){
	if(tree->children.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		searchNext(tree, -1);
	}
	else{
		searchNext(tree, 0);
	}
}

bool HoppingAmplitudeTree::Iterator::searchNext(
	const HoppingAmplitudeTree *hoppingAmplitudeTree,
	unsigned int subindex
){
	if(subindex+1 == currentIndex.size()){
		//If the node level corresponding to the current index is
		//reached, try to execute leaf node actions.

		if(currentHoppingAmplitude != -1){
			//The iterator is in the process of iterating over
			//HoppingAmplitudes on this leaf node. Try to iterate further.

			currentHoppingAmplitude++;
			if(currentHoppingAmplitude == (int)hoppingAmplitudeTree->hoppingAmplitudes.size()){
				//Last HoppingAmplitude already reached. Reset
				//currentHoppingAmplitude and return false to
				//indicate that no more HoppingAMplitudes exist
				//on this node.
				currentHoppingAmplitude = -1;
				return false;
			}
			else{
				//Return true to indicate that the next
				//HoppingAmplitude succesfully has been found.
				return true;
			}
		}

		//We are here guaranteed that the iterator is not currently in
		//a state where it is iterating over HoppingAmplitudes on this
		//node.

		if(hoppingAmplitudeTree->children.size() == 0){
			//The node has no children and is therefore either a
			//leaf node with HoppingAmplitudes stored on it, or an
			//empty dummy node.

			if(hoppingAmplitudeTree->hoppingAmplitudes.size() != 0){
				//There are HoppingAMplitudes on this node,
				//initialize the iterator to start iterating
				//over these. Return true to indicate that a
				//HoppingAmplitude was found.
				currentHoppingAmplitude = 0;
				return true;
			}
			else{
				//The node is an empty dymmy node. Return false
				//to indicate that no more HoppingAmplitudes
				//exist on this node.
				return false;
			}
		}
	}

	//We are here guaranteed that this is not a leaf or dummy node. We know
	//this because either the tests inside the previous if-statements
	//failed, or we are iterating through children that already have been
	//visited on an earlier call to searchNext if the outer if-statement
	//itself failed.

	//Perform depth first search for the next HoppingAmplitude. Starts from
	//the child node reffered to by currentIndex.
	unsigned int n = currentIndex.at(subindex);
	while(n < hoppingAmplitudeTree->children.size()){
		if(subindex+1 == currentIndex.size()){
			//The deepest point visited so far on this branch has
			//been reached. Initialize the depth first search for
			//child n to start from child n's zeroth child.
			currentIndex.push_back(0);
		}
		if(searchNext(&hoppingAmplitudeTree->children.at(n), subindex+1)){
			//Depth first search on child n succeded at finding a
			//HoppingAmplitude. Return true to indicate success.
			return true;
		}
		//Child n does not have any more HoppingAmplitudes. Pop
		//the subindex corresponding to child n's node level and
		//increment the subindex corresponding to this node level to
		//prepare for depth first search of child n+1.
		currentIndex.pop_back();
		n = ++currentIndex.back();
	}

	//Return false to indicate that no more HoppingAmplitudes could be
	//found on this node.
	return false;
}

/*const HoppingAmplitude* HoppingAmplitudeTree::Iterator::getHA() const{
	if(currentIndex.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		if(currentHoppingAmplitude == -1)
			return NULL;
		else
			return &tree->hoppingAmplitudes.at(currentHoppingAmplitude);
	}

	if(currentIndex.at(0) == (int)tree->children.size()){
		return NULL;
	}
	const HoppingAmplitudeTree *tn = this->tree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		tn = &tn->children.at(currentIndex.at(n));
	}

	return &tn->hoppingAmplitudes.at(currentHoppingAmplitude);
}*/

const HoppingAmplitude& HoppingAmplitudeTree::Iterator::operator*(){
	if(currentIndex.size() == 0){
		//Handle the special case when the data is stored on the head
		//node. Can for example be the case when iterating over a
		//single leaf node.
		if(currentHoppingAmplitude == -1){
			TBTKExit(
				"HoppingAmplitudeTree::Iterator::operator*()",
				"Out of range access. Tried to access an"
				<< " element using an iterator that points"
				<< " beyond the last element.",
				""
			);
		}
		else{
			return tree->hoppingAmplitudes.at(currentHoppingAmplitude);
		}
	}

	if(currentIndex.at(0) == (int)tree->children.size()){
		TBTKExit(
			"HoppingAmplitudeTree::Iterator::operator*()",
			"Out of range access. Tried to access an"
			<< " element using an iterator that points"
			<< " beyond the last element.",
			""
		);
	}
	const HoppingAmplitudeTree *tn = this->tree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		tn = &tn->children.at(currentIndex.at(n));
	}

	return tn->hoppingAmplitudes.at(currentHoppingAmplitude);
}

bool HoppingAmplitudeTree::Iterator::operator==(const Iterator &rhs) const{
	if(
		this->tree == rhs.tree
		&& currentIndex.size() == rhs.currentIndex.size()
	){
		for(unsigned int n = 0; n < currentIndex.size(); n++){
			if(currentIndex[n] != rhs.currentIndex[n])
				return false;
		}

		if(currentHoppingAmplitude == rhs.currentHoppingAmplitude)
			return true;
		else
			return false;
	}
	else{
		return false;
	}
}

bool HoppingAmplitudeTree::Iterator::operator!=(const Iterator &rhs) const{
	return !operator==(rhs);
}

int HoppingAmplitudeTree::Iterator::getMinBasisIndex() const{
	return tree->getMinIndex();
}

int HoppingAmplitudeTree::Iterator::getMaxBasisIndex() const{
	return tree->getMaxIndex();
}

int HoppingAmplitudeTree::Iterator::getNumBasisIndices() const{
	if(getMaxBasisIndex() == -1)
		return 0;
	else
		return 1 + getMaxBasisIndex() - getMinBasisIndex();
}

HoppingAmplitudeTree::Iterator HoppingAmplitudeTree::begin() const{
	return Iterator(this);
}

HoppingAmplitudeTree::Iterator HoppingAmplitudeTree::end() const{
	return Iterator(this, true);
}

HoppingAmplitude HoppingAmplitudeTree::getFirstHA() const{
	if(children.size() == 0)
		return hoppingAmplitudes.at(0);

	for(unsigned int n = 0; n < children.size(); n++){
		if(children.at(n).children.size() != 0 || children.at(n).hoppingAmplitudes.size() != 0)
			return children.at(n).getFirstHA();
	}

	//Sould never happen. Line added to avoid compiler warnings.
	return HoppingAmplitude(0, {0, 0, 0}, {0, 0, 0});
}

};	//End of namespace TBTK
