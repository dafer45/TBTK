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

/** @package TBTKcalc
 *  @file HALinkedList.h
 *  @brief Support data structure for ChebyshevSolver
 *
 *  Experimental
 *
 *  Only used in the experimental function
 *  ChebyshevSolver::calculateCoefficients(..., componentCutoff, ...)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HA_LINKED_LIST
#define COM_DAFER45_TBTK_HA_LINKED_LIST

#include "HoppingAmplitudeSet.h"
#include <complex>

namespace TBTK{

class HALink{
public:
	int from;
	int to;
	std::complex<double> amplitude;

	HALink *next1;
	HALink *next2;
};

class HALinkedList{
public:
	HALinkedList(HoppingAmplitudeSet &as);
	~HALinkedList();

	void addLinkedList(int from);
	HALink* getFirstMainLink();
	HALink* getLinkArray();
	int getLinkArraySize();
	void rescaleAmplitudes(double scaleFactor);
private:
	int linkArraySize;

	HALink *linkArray;
	HALink **linkList;
	HALink *mainListFirst;
	HALink *mainListLast;

	bool *inMainList;
};

inline HALink* HALinkedList::getFirstMainLink(){
	return mainListFirst;
}

inline HALink* HALinkedList::getLinkArray(){
	return linkArray;
}

inline int HALinkedList::getLinkArraySize(){
	return linkArraySize;
}

inline void HALinkedList::rescaleAmplitudes(double scaleFactor){
	for(int n = 0; n < linkArraySize; n++){
		linkArray[n].amplitude /= scaleFactor;
	}
}

};	//End of namespace TBTK

#endif
