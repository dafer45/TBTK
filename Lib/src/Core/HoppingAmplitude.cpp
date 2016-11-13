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

/** @file HoppingAmplitude.cpp
 *
 *  @author Kristofer Björnson
 */

#include "HoppingAmplitude.h"
#include "Streams.h"

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
}

HoppingAmplitude::HoppingAmplitude(
	const HoppingAmplitude &ha
) :
	fromIndex(ha.fromIndex),
	toIndex(ha.toIndex)
{
	amplitude = ha.amplitude;
	this->amplitudeCallback = ha.amplitudeCallback;
}

HoppingAmplitude HoppingAmplitude::getHermitianConjugate() const{
	if(amplitudeCallback)
		return HoppingAmplitude(toIndex, fromIndex, amplitudeCallback);
	else
		return HoppingAmplitude(toIndex, fromIndex, conj(amplitude));
}

void HoppingAmplitude::print(){
	Streams::out << "From index:\t";
	for(unsigned int n = 0; n < fromIndex.size(); n++){
		Streams::out << fromIndex.at(n) << " ";
	}
	Streams::out << "\n";
	Streams::out << "To index:\t";
	for(unsigned int n = 0; n < toIndex.size(); n++){
		Streams::out << toIndex.at(n) << " ";
	}
	Streams::out << "\n";
	Streams::out << "Amplitude:\t" << getAmplitude() << "\n";
}

};	//End of namespace TBTK
