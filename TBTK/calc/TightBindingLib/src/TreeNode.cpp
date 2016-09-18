/** @file TreeNode.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/TreeNode.h"
#include <iostream>
#include <algorithm>

using namespace std;

namespace TBTK{

TreeNode::TreeNode(){
	basisIndex = -1;
	basisSize = -1;
}

void TreeNode::print(){
	print(0);
}

void TreeNode::print(unsigned int subindex){
	for(unsigned int n = 0; n < subindex; n++)
		std::cout << "\t";
	std::cout << basisIndex << ":" << hoppingAmplitudes.size() << "\n";
	for(unsigned int n = 0; n < children.size(); n++)
		children.at(n).print(subindex + 1);
}

void TreeNode::add(HoppingAmplitude ha){
	add(ha, 0);
}

void TreeNode::add(HoppingAmplitude &ha, unsigned int subindex){
	if(subindex < ha.fromIndex.size()){
		//If the current subindex is not the last, the HoppingAmplitude
		//is propagated to the next node level.

		//Get current subindex
		int currentIndex = ha.fromIndex.at(subindex);
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
			std::cout << "Error, incompatible amplitudes1:\n";
			ha.print();
			hoppingAmplitudes.at(0).print();
			exit(0);
		}
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
			std::cout << "Error, incompatible amplitudes2:\n";
			ha.print();
			getFirstHA().print();
			exit(0);
		}
		//Add HoppingAmplitude to node.
		hoppingAmplitudes.push_back(ha);
	}
}

std::vector<HoppingAmplitude>* TreeNode::getHAs(Index index){
	return getHAs(index, 0);
}

std::vector<HoppingAmplitude>* TreeNode::getHAs(Index index, unsigned int subindex){
	if(subindex < index.size()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		int currentIndex = index.at(subindex);
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			cout << "Error, index out of bound: ";
			index.print();
		}
		//Continue to the next node level.
		return children.at(currentIndex).getHAs(index, subindex+1);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return &hoppingAmplitudes;
	}
}

int TreeNode::getBasisIndex(const Index &index){
	return getBasisIndex(index, 0);
}

int TreeNode::getBasisIndex(const Index &index, unsigned int subindex){
	if(subindex < index.size()){
		//If the current subindex is not the last, continue to the next
		//node level.

		//Get current subindex
		int currentIndex = index.at(subindex);
		//Error detection:
		//If the subindex is bigger than the current number of child
		//nodes, an error has occured.
		if(currentIndex >= (int)children.size()){
			cout << "Error, index out of bound: ";
			index.print();
		}
		//Continue to the next node level.
		return children.at(currentIndex).getBasisIndex(index, subindex+1);
	}
	else{
		//If the current subindex is the last, return HoppingAmplitudes.
		return basisIndex;
	}
}

Index TreeNode::getPhysicalIndex(int basisIndex){
	if(basisIndex < 0 || basisIndex > this->basisSize){
		cout << "Error in TreeNode::getPhysicalIndex(): Hilbert space index out of bound.";
		exit(1);
	}
	vector<int> indices;

	getPhysicalIndex(basisIndex, &indices);

	return Index(indices);
}

void TreeNode::getPhysicalIndex(int basisIndex, vector<int> *indices){
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

int TreeNode::getMinIndex(){
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

int TreeNode::getMaxIndex(){
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
		int basisIndex1 = rootNode->getBasisIndex(ha1.toIndex);
		int basisIndex2 = rootNode->getBasisIndex(ha2.toIndex);
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

TreeNode::Iterator::Iterator(TreeNode *tree){
	this->tree = tree;
	currentIndex.push_back(0);
	currentHoppingAmplitude = -1;
	searchNext(tree, 0);
}

void TreeNode::Iterator::reset(){
	currentIndex.clear();
	currentIndex.push_back(0);
	currentHoppingAmplitude = -1;
	searchNext(tree, 0);
}

void TreeNode::Iterator::searchNextHA(){
	searchNext(tree, 0);
}

bool TreeNode::Iterator::searchNext(TreeNode *treeNode, unsigned int subindex){
	if(subindex+1 == currentIndex.size()){
		if(currentHoppingAmplitude != -1){
			currentHoppingAmplitude++;
			if(currentHoppingAmplitude == (int)treeNode->hoppingAmplitudes.size()){
				currentHoppingAmplitude = -1;
				return false;
			}
			else{
				return true;
			}
		}
		if(treeNode->children.size() == 0){
			if(treeNode->hoppingAmplitudes.size() != 0){
				currentHoppingAmplitude = 0;
				return true;
			}
			else{
				return false;
			}
		}
	}
	unsigned int n = currentIndex.at(subindex);
	while(n < treeNode->children.size()){
		if(subindex+1 == currentIndex.size())
			currentIndex.push_back(0);
		if(searchNext(&treeNode->children.at(n), subindex+1)){
			return true;
		}
		currentIndex.pop_back();
		n = ++currentIndex.back();
	}

	return false;
}

HoppingAmplitude* TreeNode::Iterator::getHA(){
	if(currentIndex.at(0) == (int)tree->children.size()){
		return NULL;
	}
	TreeNode *tn = this->tree;
	for(unsigned int n = 0; n < currentIndex.size()-1; n++){
		tn = &tn->children.at(currentIndex.at(n));
	}

	return &tn->hoppingAmplitudes.at(currentHoppingAmplitude);
}

TreeNode::Iterator TreeNode::begin(){
	return Iterator(this);
}

HoppingAmplitude TreeNode::getFirstHA(){
	if(children.size() == 0)
		return hoppingAmplitudes.at(0);

	for(unsigned int n = 0; n < children.size(); n++){
		if(children.at(n).children.size() != 0 || children.at(n).hoppingAmplitudes.size() != 0)
			return children.at(n).getFirstHA();
	}

	//Sould never happen. Line added to avoid compiler warnings.
	return HoppingAmplitude({0, 0, 0}, {0, 0, 0}, 0);
}

};	//End of namespace TBTK
