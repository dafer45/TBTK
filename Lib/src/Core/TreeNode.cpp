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

/** @file TreeNode.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTKMacros.h"
#include "TreeNode.h"
#include "Streams.h"

#include <algorithm>

using namespace std;

namespace TBTK{

TreeNode::TreeNode(){
	basisIndex = -1;
	basisSize = -1;
	isPotentialBlockSeparator = true;
}

TreeNode::TreeNode(const string &serialization, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		TBTKAssert(
			validate(serialization, "TreeNode", mode),
			"TreeNode::TreeNode()",
			"Unable to parse string as TreeNode '" << serialization
			<< "'.",
			""
		);
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
				).compare("TreeNode") == 0,
				"TreeNode::TreeNode()",
				"Unable to parse string as TreeNode. Expected"
				<< " 'TreeNode' but found '"
				<< getID(elements.at(n), mode) << "'.",
				""
			)
			children.push_back(TreeNode(elements.at(n), mode));
		}

		break;
	}
	default:
		TBTKExit(
			"TreeNode::TreeNode()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

TreeNode::~TreeNode(){
}

vector<Index> TreeNode::getIndexList(const Index &pattern) const{
	vector<Index> indexList;

	Iterator it = begin();
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
	}

	return indexList;
}

void TreeNode::print(){
	print(0);
}

void TreeNode::print(unsigned int subindex){
	for(unsigned int n = 0; n < subindex; n++)
		Streams::out << "\t";
	Streams::out << basisIndex << ":" << hoppingAmplitudes.size() << "\n";
	for(unsigned int n = 0; n < children.size(); n++)
		children.at(n).print(subindex + 1);
}

void TreeNode::add(HoppingAmplitude ha){
	add(ha, 0);
}

void TreeNode::add(HoppingAmplitude &ha, unsigned int subindex){
//	if(subindex < ha.fromIndex.size()){
	if(subindex < ha.getFromIndex().size()){
		//If the current subindex is not the last, the HoppingAmplitude
		//is propagated to the next node level.

		//Get current subindex
//		int currentIndex = ha.fromIndex.at(subindex);
		int currentIndex = ha.getFromIndex().at(subindex);
		//If the subindex is bigger than the current number of child
		//nodes, create empty nodes.
		if(currentIndex >= (int)children.size()){
			for(int n = children.size(); n <= currentIndex; n++){
				children.push_back(TreeNode());
			}
		}
		//Error detection:
		//If a HoppingAmplitude is found on this level, another
		//HoppingAmplitude with fewer subindices than the current
		//HoppingAmplitude have previously been added to this node.
		//This is an error because different number of subindices is
		//only allowed if the HoppingAmplitudes differ in one of their
		//common indices.
		if(hoppingAmplitudes.size() != 0){
			Streams::err << "Error, incompatible amplitudes:\n";
			ha.print();
			hoppingAmplitudes.at(0).print();
			exit(1);
		}
		//Ensure isPotentialBlockSeparator is set to false in case the
		//'toIndex' and the 'fromIndex' differs in the subindex
		//corresponding to this TreeNode level.
//		if(ha.toIndex.size() <= subindex || currentIndex != ha.toIndex.at(subindex))
		if(ha.getToIndex().size() <= subindex || currentIndex != ha.getToIndex().at(subindex))
			isPotentialBlockSeparator = false;
		//Propagate to the next node level.
		children.at(currentIndex).add(ha, subindex+1);
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
		if(children.size() != 0){
			Streams::err << "Error, incompatible amplitudes:\n";
			ha.print();
			getFirstHA().print();
			exit(1);
		}
		//Add HoppingAmplitude to node.
		hoppingAmplitudes.push_back(ha);
	}
}

const TreeNode* TreeNode::getSubTree(const Index &subspace) const{
	for(unsigned int n = 0; n < subspace.size(); n++){
		if(subspace.at(n) < 0){
			TBTKExit(
				"TreeNode::getSubTree()",
				"Invalid subspace index '" << subspace.toString() << "'.",
				"Subspace indices cannot have negative subindices."
			);
		}
	}

	return getSubTree(subspace, 0);
}

const TreeNode* TreeNode::getSubTree(const Index &subspace, unsigned int subindex) const{
	if(subindex == subspace.size()){
		//Correct node reached

		return this;
	}

	if((unsigned int)subspace.at(subindex) < children.size()){
		return children.at(subspace.at(subindex)).getSubTree(subspace, subindex+1);
	}
	else{
		TBTKExit(
			"TreeNode::getSubTree()",
			"Subspace index '" << subspace.toString() << "' does not exist.",
			""
		);
	}
}

bool TreeNode::isProperSubspace(const Index &subspace){
	for(unsigned int n = 0; n < subspace.size(); n++){
		if(subspace.at(n) < 0){
			TBTKExit(
				"TreeNode::getSubTree()",
				"Invalid subspace index '" << subspace.toString() << "'.",
				"Subspace indices cannot have negative subindices."
			);
		}
	}

	return isProperSubspace(subspace, 0);
}

bool TreeNode::isProperSubspace(const Index &subspace, unsigned int subindex){
	if(subindex == subspace.size())
		return isPotentialBlockSeparator;

	if(isPotentialBlockSeparator){
		if((unsigned int)subspace.at(subindex) < children.size()){
			return isProperSubspace(subspace, subindex+1);
		}
		else{
			TBTKExit(
				"TreeNode::isProperSubspace()",
				"Subspace index '" <<subspace.toString() << "' does not exist.",
				""
			);
		}
	}
	else{
		return false;
	}
}

IndexTree TreeNode::getSubspaceIndices() const{
	IndexTree blockIndices;
	if(isPotentialBlockSeparator)
		getBlockIndices(blockIndices, Index());
//		getBlockIndices(blockIndices, Index({}));
	blockIndices.generateLinearMap();

	return blockIndices;
}

void TreeNode::getBlockIndices(IndexTree &blockIndices, Index index) const{
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

const std::vector<HoppingAmplitude>* TreeNode::getHAs(Index index) const{
	return getHAs(index, 0);
}

const std::vector<HoppingAmplitude>* TreeNode::getHAs(Index index, unsigned int subindex) const{
	if(subindex < index.size()){
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
		return children.at(currentIndex).getHAs(index, subindex+1);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return &hoppingAmplitudes;
	}
}

int TreeNode::getBasisIndex(const Index &index) const{
	return getBasisIndex(index, 0);
}

int TreeNode::getBasisIndex(const Index &index, unsigned int subindex) const{
	if(subindex < index.size()){
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
		return children.at(currentIndex).getBasisIndex(index, subindex+1);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return basisIndex;
	}
}

Index TreeNode::getPhysicalIndex(int basisIndex) const{
	TBTKAssert(
		basisIndex >= 0 && basisIndex < this->basisSize,
		"TreeNode::getPhysicalIndex()",
		"Hilbert space index out of bound.",
		""
	);

	vector<int> indices;
	getPhysicalIndex(basisIndex, &indices);

	return Index(indices);
}

void TreeNode::getPhysicalIndex(int basisIndex, vector<int> *indices) const{
	if(this->basisIndex != -1)
		return;

	for(unsigned int n = 0; n < children.size(); n++){
		int min = children.at(n).getMinIndex();
		int max = children.at(n).getMaxIndex();

		if(min == -1)
			continue;

		if(min <= basisIndex && basisIndex <= max){
			indices->push_back(n);
			children.at(n).getPhysicalIndex(basisIndex, indices);
			break;
		}
	}
}

int TreeNode::getMinIndex() const{
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

int TreeNode::getMaxIndex() const{
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

void TreeNode::generateBasisIndices(){
	basisSize = generateBasisIndices(0);
}

int TreeNode::generateBasisIndices(int i){
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
	static TreeNode *rootNode;
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

TreeNode *SortHelperClass::rootNode = NULL;

void TreeNode::sort(TreeNode *rootNode){
	if(hoppingAmplitudes.size() != 0){
		SortHelperClass::rootNode = rootNode;
		std::sort(hoppingAmplitudes.begin(), hoppingAmplitudes.end(), SortHelperClass());
	}
	else if(children.size() != 0){
		for(unsigned int n = 0; n < children.size(); n++)
			children.at(n).sort(rootNode);
	}
}

string TreeNode::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "TreeNode(";
		ss << Serializeable::serialize(basisIndex, mode);
		ss << "," << Serializeable::serialize(basisSize, mode);
		ss << "," << Serializeable::serialize(
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
		stringstream ss;
		ss << "{";
		ss << "id:'TreeNode'";
		ss << "," << "basisIndex:" << Serializeable::serialize(
			basisIndex,
			mode
		);
		ss << "," << "basisSize:" << Serializeable::serialize(
			basisSize,
			mode
		);
		ss << "," << "isPotentialBlockSeparator"
			<< Serializeable::serialize(
				isPotentialBlockSeparator,
				mode
			);
		ss << "," << "hoppingAmplitudes:[";
		for(unsigned int n = 0; n < hoppingAmplitudes.size(); n++){
			if(n != 0)
				ss << ",";
			ss << hoppingAmplitudes.at(n).serialize(mode);
		}
		ss << "]";
		ss << "," << "children:[";
		for(unsigned int n = 0; n < children.size(); n++){
			if(n != 0)
				ss << ",";
			ss << children.at(n).serialize(mode);
		}
		ss << "]";
		ss << "}";

		return ss.str();
	}
	default:
		TBTKExit(
			"TreeNode::serialize()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

TreeNode::Iterator::Iterator(const TreeNode *tree){
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

void TreeNode::Iterator::reset(){
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

void TreeNode::Iterator::searchNextHA(){
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

bool TreeNode::Iterator::searchNext(const TreeNode *treeNode, unsigned int subindex){
	if(subindex+1 == currentIndex.size()){
		//If the node level corresponding to the current index is
		//reached, try to execute leaf node actions.

		if(currentHoppingAmplitude != -1){
			//The iterator is in the process of iterating over
			//HoppingAmplitudes on this leaf node. Try to iterate further.

			currentHoppingAmplitude++;
			if(currentHoppingAmplitude == (int)treeNode->hoppingAmplitudes.size()){
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

		if(treeNode->children.size() == 0){
			//The node has no children and is therefore either a
			//leaf node with HoppingAmplitudes stored on it, or an
			//empty dummy node.

			if(treeNode->hoppingAmplitudes.size() != 0){
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
	while(n < treeNode->children.size()){
		if(subindex+1 == currentIndex.size()){
			//The deepest point visited so far on this branch has
			//been reached. Initialize the depth first search for
			//child n to start from child n's zeroth child.
			currentIndex.push_back(0);
		}
		if(searchNext(&treeNode->children.at(n), subindex+1)){
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

const HoppingAmplitude* TreeNode::Iterator::getHA() const{
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
	const TreeNode *tn = this->tree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		tn = &tn->children.at(currentIndex.at(n));
	}

	return &tn->hoppingAmplitudes.at(currentHoppingAmplitude);
}

int TreeNode::Iterator::getMinBasisIndex() const{
	return tree->getMinIndex();
}

int TreeNode::Iterator::getMaxBasisIndex() const{
	return tree->getMaxIndex();
}

int TreeNode::Iterator::getNumBasisIndices() const{
	if(getMaxBasisIndex() == -1)
		return 0;
	else
		return 1 + getMaxBasisIndex() - getMinBasisIndex();
}

TreeNode::Iterator TreeNode::begin() const{
	return Iterator(this);
}

HoppingAmplitude TreeNode::getFirstHA() const{
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
