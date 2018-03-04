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

/** @file DPropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

namespace{
	complex<double> i(0,1);
}

ArnoldiIterator::ArnoldiIterator(Solver::ArnoldiIterator &aSolver){
	this->aSolver = &aSolver;
}

ArnoldiIterator::~ArnoldiIterator(){
}

Property::EigenValues ArnoldiIterator::getEigenValues(){
	int size = aSolver->getNumEigenValues();
	const complex<double> *ev = aSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	double *data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = real(ev[n]);

	return eigenValues;
}

Property::WaveFunctions ArnoldiIterator::calculateWaveFunctions(
	initializer_list<Index> patterns,
	initializer_list<int> states
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if(*states.begin() == IDX_ALL){
			for(int n = 0; n < aSolver->getModel().getBasisSize(); n++)
				statesVector.push_back(n);
		}
		else{
			TBTKAssert(
				*states.begin() >= 0,
				"PropertyExtractor::ArnoldiIterator::calculateWaveFunctions()",
				"Found unexpected index symbol.",
				"Use only positive numbers or '{IDX_ALL}'"
			);
			statesVector.push_back(*states.begin());
		}
	}
	else{
		for(unsigned int n = 0; n < states.size(); n++){
			TBTKAssert(
				*(states.begin() + n) >= 0,
				"PropertyExtractor::ArnoldiIterator::calculateWaveFunctions()",
				"Found unexpected index symbol.",
				"Use only positive numbers or '{IDX_ALL}'"
			);
			statesVector.push_back(*(states.begin() + n));
		}
	}

	Property::WaveFunctions waveFunctions(memoryLayout, statesVector);

	hint = new Property::WaveFunctions*[1];
	((Property::WaveFunctions**)hint)[0] = &waveFunctions;

	calculate(
		calculateWaveFunctionsCallback,
		allIndices,
		memoryLayout,
		waveFunctions
	);

	delete [] (Property::WaveFunctions**)hint;

	return waveFunctions;
}

/*Property::GreensFunction* APropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	unsigned int numPoles = aSolver->getModel().getBasisSize();

	complex<double> *positions = new complex<double>[numPoles];
	complex<double> *amplitudes = new complex<double>[numPoles];
	for(int n = 0; n < aSolver->getNumEigenValues(); n++){
		positions[n] = aSolver->getEigenValue(n);

		complex<double> uTo = aSolver->getAmplitude(n, to);
		complex<double> uFrom = aSolver->getAmplitude(n, from);

		amplitudes[n] = uTo*conj(uFrom);
	}

	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
		Property::GreensFunction::Format::Poles,
		numPoles,
		positions,
		amplitudes
	);

	delete [] positions;
	delete [] amplitudes;

	return greensFunction;
}*/

Property::DOS ArnoldiIterator::calculateDOS(){
	const complex<double> *ev = aSolver->getEigenValues();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	double *data = dos.getDataRW();
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < aSolver->getNumEigenValues(); n++){
		int e = (int)(((real(ev[n]) - lowerBound)/(upperBound - lowerBound))*energyResolution);
		if(e >= 0 && e < energyResolution){
			data[e] += 1./dE;
		}
	}

	return dos;
}

Property::LDOS ArnoldiIterator::calculateLDOS(
	Index pattern,
	Index ranges
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"PropertyExtractor::ArnoldiIterator::calculateLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to"
		<< " ensure eigen vectors are calculated."
	);

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[1];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::LDOS ldos(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateLDOSCallback,
		(void*)ldos.getData(),
		pattern,
		ranges,
		0,
		energyResolution
	);

	return ldos;
}

Property::LDOS ArnoldiIterator::calculateLDOS(
	initializer_list<Index> patterns
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"PropertyExtractor::ArnoldiIterator::calculateLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to ensure eigen vectors are calculated."
	);

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[1];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	IndexTree allIndices = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos
	);

	return ldos;
}

Property::SpinPolarizedLDOS ArnoldiIterator::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"ArnoldiIterator::calculateSpinPolarizedLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to"
		<< " ensure eigen vectors are calculated."
	);

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	((int**)hint)[1][1] = -1;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int**)hint)[1][1] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int**)hint)[1][1] == -1){
		delete [] ((double**)hint)[0];
		delete [] ((int**)hint)[1];
		delete [] (void**)hint;
		TBTKExit(
			"PropertyExtractor::ArnoldiIterator::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		(void*)spinPolarizedLDOS.getDataRW(),
		pattern,
		ranges,
		0,
		energyResolution
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS ArnoldiIterator::calculateSpinPolarizedLDOS(
	initializer_list<Index> patterns
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"PropertyExtractor::ArnoldiIterator::calculateSpinPolarizedLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to"
		<< " ensure eigen vectors are calculated."
	);

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	IndexTree allIndices = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		&(((int**)hint)[1][1])
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}

void ArnoldiIterator::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	void *waveFunctions,
	const Index &index,
	int offset
){
	ArnoldiIterator *pe = (ArnoldiIterator*)cb_this;

	const vector<unsigned int> states = ((Property::WaveFunctions**)pe->hint)[0]->getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		((complex<double>*)waveFunctions)[offset + n] += pe->getAmplitude(states.at(n), index);
}

void ArnoldiIterator::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	ArnoldiIterator *pe = (ArnoldiIterator*)cb_this;

	const complex<double> *eigenValues = pe->aSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];

	double step_size = (u_lim - l_lim)/(double)resolution;

	double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->aSolver->getNumEigenValues(); n++){
		if(real(eigenValues[n]) > l_lim && real(eigenValues[n]) < u_lim){
			complex<double> u = pe->aSolver->getAmplitude(n, index);

			int e = (int)((real(eigenValues[n]) - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)ldos)[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void ArnoldiIterator::calculateSpinPolarizedLDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	ArnoldiIterator *pe = (ArnoldiIterator*)cb_this;

	const complex<double> *eigenValues = pe->aSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];
	int spin_index = ((int**)pe->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->aSolver->getNumEigenValues(); n++){
		if(real(eigenValues[n]) > l_lim && real(eigenValues[n]) < u_lim){
			complex<double> u_u = pe->aSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->aSolver->getAmplitude(n, index_d);

			int e = (int)((real(eigenValues[n]) - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
