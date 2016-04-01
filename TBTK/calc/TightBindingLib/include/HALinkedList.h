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

#include "AmplitudeSet.h"
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
	HALinkedList(AmplitudeSet &as);
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

};

#endif
