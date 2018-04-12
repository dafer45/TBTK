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

/** @file HALinkedList.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HALinkedList.h"

namespace TBTK{

HALinkedList::HALinkedList(const HoppingAmplitudeSet &as){
	int numHoppingAmplitudes = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator = as.cbegin();
		iterator != as.cend();
		++iterator
	){
		numHoppingAmplitudes++;
	}

	linkArraySize = numHoppingAmplitudes;
	linkArray = new HALink[numHoppingAmplitudes];
	linkList = new HALink*[as.getBasisSize()];
	HALink **lastLinks = new HALink*[as.getBasisSize()];
	for(int n = 0; n < as.getBasisSize(); n++){
		linkList[n] = NULL;
		lastLinks[n] = NULL;
	}
	int counter = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator = as.cbegin();
		iterator != as.cend();
		++iterator
	){
		linkArray[counter].from = as.getBasisIndex((*iterator).getFromIndex());
		linkArray[counter].to = as.getBasisIndex((*iterator).getToIndex());
		linkArray[counter].amplitude = (*iterator).getAmplitude();
		linkArray[counter].next1 = NULL;
		linkArray[counter].next2 = NULL;

		int from = linkArray[counter].from;
		if(lastLinks[from] == NULL)
			linkList[from] = &linkArray[counter];
		else
			lastLinks[from]->next1 = &linkArray[counter];

		lastLinks[from] = &linkArray[counter];

		counter++;
	}

	mainListFirst = NULL;
	mainListLast = NULL;

	inMainList = new bool[as.getBasisSize()];
	for(int n = 0; n < as.getBasisSize(); n++)
		inMainList[n] = false;

	delete [] lastLinks;
}

HALinkedList::~HALinkedList(){
	delete [] linkArray;
	delete [] linkList;
}

void HALinkedList::addLinkedList(int from){
	if(mainListFirst == NULL){
		mainListFirst = linkList[from];

		HALink *link = mainListFirst;
		while(link != NULL){
			link->next2 = link->next1;
			mainListLast = link;
			link = link->next1;
		}
//		mainListLast = link;

		inMainList[from] = true;
	}

	if(!inMainList[from]){
		mainListLast->next2 = linkList[from];

		HALink *link = mainListLast->next2;
		while(link != NULL){
			link->next2 = link->next1;
			mainListLast = link;
			link = link->next1;
		}

		inMainList[from] = true;
	}
}

};	//End of namespace TBTK
