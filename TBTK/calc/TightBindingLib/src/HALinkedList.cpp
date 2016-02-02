#include "../include/HALinkedList.h"

HALinkedList::HALinkedList(AmplitudeSet &as){
	AmplitudeSet::iterator it = as.getIterator();
	HoppingAmplitude *ha;
	int numHoppingAmplitudes = 0;
	while((ha = it.getHA())){
		numHoppingAmplitudes++;
		it.searchNextHA();
	}

	linkArraySize = numHoppingAmplitudes;
	linkArray = new HALink[numHoppingAmplitudes];
	linkList = new HALink*[as.getBasisSize()];
	HALink **lastLinks = new HALink*[as.getBasisSize()];
	for(int n = 0; n < as.getBasisSize(); n++){
		linkList[n] = NULL;
		lastLinks[n] = NULL;
	}
	it.reset();
	int counter = 0;
	while((ha = it.getHA())){
		linkArray[counter].from = as.getBasisIndex(ha->fromIndex);
		linkArray[counter].to = as.getBasisIndex(ha->toIndex);
		linkArray[counter].amplitude = ha->getAmplitude();
		linkArray[counter].next1 = NULL;
		linkArray[counter].next2 = NULL;

		int from = linkArray[counter].from;
		if(lastLinks[from] == NULL)
			linkList[from] = &linkArray[counter];
		else
			lastLinks[from]->next1 = &linkArray[counter];

		lastLinks[from] = &linkArray[counter];

		counter++;

		it.searchNextHA();
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
