/** @file HoppingAmplitude.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/HoppingAmplitude.h"
#include <iostream>

using namespace std;

namespace TBTK{

HoppingAmplitude::HoppingAmplitude(
	Index fromIndex,
	Index toIndex,
	complex<double> amplitude
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	this->amplitude = amplitude;
	this->amplitudeCallback = NULL;
	this->toUnitCell = NULL;
};

HoppingAmplitude::HoppingAmplitude(
	Index fromIndex,
	Index toIndex,
	complex<double> (*amplitudeCallback)(Index, Index)
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	this->amplitudeCallback = amplitudeCallback;
	this->toUnitCell = NULL;
};

HoppingAmplitude::HoppingAmplitude(
	complex<double> amplitude,
	Index toIndex,
	Index fromIndex
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	this->amplitude = amplitude;
	this->amplitudeCallback = NULL;
	this->toUnitCell = NULL;
};

HoppingAmplitude::HoppingAmplitude(
	complex<double> (*amplitudeCallback)(Index, Index),
	Index toIndex,
	Index fromIndex
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	this->amplitudeCallback = amplitudeCallback;
	this->toUnitCell = NULL;
};

HoppingAmplitude::HoppingAmplitude(
	complex<double> amplitude,
	Index toIndex,
	Index fromIndex,
	Index toUnitCell
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	amplitudeCallback = NULL;
	this->toUnitCell = new Index(toUnitCell);
}

HoppingAmplitude::HoppingAmplitude(
	complex<double> (*amplitudeCallback)(Index, Index),
	Index toIndex,
	Index fromIndex,
	Index toUnitCell
) :
	fromIndex(fromIndex),
	toIndex(toIndex)
{
	this->amplitudeCallback = amplitudeCallback;
	this->toUnitCell = new Index(toUnitCell);
}

HoppingAmplitude::HoppingAmplitude(
	const HoppingAmplitude &ha
) :
	fromIndex(ha.fromIndex),
	toIndex(ha.toIndex)
{
	amplitude = ha.amplitude;
	this->amplitudeCallback = ha.amplitudeCallback;

	if(ha.toUnitCell != NULL){
		this->toUnitCell = new Index(*ha.toUnitCell);
	}
	else{
		this->toUnitCell = NULL;
	}
}

HoppingAmplitude HoppingAmplitude::getHermitianConjugate(){
	if(amplitudeCallback)
		return HoppingAmplitude(toIndex, fromIndex, amplitudeCallback);
	else
		return HoppingAmplitude(toIndex, fromIndex, conj(amplitude));
}

void HoppingAmplitude::print(){
	cout << "From index:\t";
	for(unsigned int n = 0; n < fromIndex.size(); n++){
		cout << fromIndex.at(n) << " ";
	}
	cout << "\n";
	cout << "To index:\t";
	for(unsigned int n = 0; n < toIndex.size(); n++){
		cout << toIndex.at(n) << " ";
	}
	cout << "\n";
	cout << "Amplitude:\t" << getAmplitude() << "\n";
}

};	//End of namespace TBTK
