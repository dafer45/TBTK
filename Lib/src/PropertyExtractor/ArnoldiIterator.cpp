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

Property::EigenValues ArnoldiIterator::getEigenValues(){
	int size = aSolver->getNumEigenValues();
	const CArray<complex<double>> &ev = aSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	std::vector<double> &data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = real(ev[n]);

	return eigenValues;
}

Property::WaveFunctions ArnoldiIterator::calculateWaveFunctions(
	vector<Index> patterns,
	vector<Subindex> states
){
	TBTKAssert(
                aSolver->getCalculateEigenVectors(),
                "PropertyExtractor::ArnoldiIterator::calculateWaveFunctions()",
                "Eigenvectors not available.",
                "Configure the solver to calculate eigenvectors using"
                << " Solver::ArnoldiIterator::setCalculateEigenVectors()."
        );
	IndexTree allIndices = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if((*states.begin()).isWildcard()){
			for(int n = 0; n < aSolver->getNumEigenValues(); n++)
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

	Information information;
	calculate(
		calculateWaveFunctionsCallback,
		allIndices,
		memoryLayout,
		waveFunctions,
		information
	);

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
	const CArray<complex<double>> &ev = aSolver->getEigenValues();

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	std::vector<double> &data = dos.getDataRW();
	double dE = dos.getDeltaE();
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

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	double energyResolution = getEnergyResolution();

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::LDOS ldos(
		loopRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	Information information;
	calculate(
		calculateLDOSCallback,
		ldos,
		pattern,
		ranges,
		0,
		energyResolution,
		information
	);

	return ldos;
}

Property::LDOS ArnoldiIterator::calculateLDOS(
	vector<Index> patterns
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"PropertyExtractor::ArnoldiIterator::calculateLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to ensure eigen vectors are calculated."
	);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTree allIndices = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	Information information;
	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos,
		information
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

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Information information;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n).isSpinIndex()){
			information.setSpinIndex(n);
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(information.getSpinIndex() == -1){
		TBTKExit(
			"PropertyExtractor::ArnoldiIterator::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		loopRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		spinPolarizedLDOS,
		pattern,
		ranges,
		0,
		energyResolution,
		information
	);

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS ArnoldiIterator::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	TBTKAssert(
		aSolver->getCalculateEigenVectors(),
		"PropertyExtractor::ArnoldiIterator::calculateSpinPolarizedLDOS()",
		"Eigen vectors not calculated.",
		"Use Solver::ArnoldiIterator::setCalculateEigenVectors() to"
		<< " ensure eigen vectors are calculated."
	);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTree allIndices = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		aSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	Information information;
	calculate(
		calculateSpinPolarizedLDOSCallback,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		information
	);

	return spinPolarizedLDOS;
}

void ArnoldiIterator::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ArnoldiIterator *propertyExtractor = (ArnoldiIterator*)cb_this;
	Property::WaveFunctions &waveFunctions
		= (Property::WaveFunctions&)property;
	vector<complex<double>> &data = waveFunctions.getDataRW();

	const vector<unsigned int> states = waveFunctions.getStates();
	for(unsigned int n = 0; n < states.size(); n++){
		data[offset + n] += propertyExtractor->getAmplitude(
			states.at(n),
			index
		);
	}
}

void ArnoldiIterator::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ArnoldiIterator *propertyExtractor = (ArnoldiIterator*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();
	Solver::ArnoldiIterator &solver = *propertyExtractor->aSolver;

	const CArray<complex<double>> &eigenValues = solver.getEigenValues();

	double lowerBound = propertyExtractor->getLowerBound();
	double upperBound = propertyExtractor->getUpperBound();
	int energyResolution = propertyExtractor->getEnergyResolution();

	double dE = ldos.getDeltaE();
	for(int n = 0; n < solver.getNumEigenValues(); n++){
		if(
			real(eigenValues[n]) > lowerBound
			&& real(eigenValues[n]) < upperBound
		){
			complex<double> u = solver.getAmplitude(n, index);

			int e = (int)((real(eigenValues[n]) - lowerBound)/dE);
			if(e >= energyResolution)
				e = energyResolution - 1;
			data[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void ArnoldiIterator::calculateSpinPolarizedLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ArnoldiIterator *propertyExtractor = (ArnoldiIterator*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();
	Solver::ArnoldiIterator &solver = *propertyExtractor->aSolver;

	const CArray<complex<double>> &eigenValues = solver.getEigenValues();

	int spinIndex = information.getSpinIndex();

	double lowerBound = propertyExtractor->getLowerBound();
	double upperBound = propertyExtractor->getUpperBound();
	int energyResolution = propertyExtractor->getEnergyResolution();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	double dE = spinPolarizedLDOS.getDeltaE();
	for(int n = 0; n < solver.getNumEigenValues(); n++){
		if(
			real(eigenValues[n]) > lowerBound
			&& real(eigenValues[n]) < upperBound
		){
			complex<double> u_u = solver.getAmplitude(n, index_u);
			complex<double> u_d = solver.getAmplitude(n, index_d);

			int e = (int)((real(eigenValues[n]) - lowerBound)/dE);
			if(e >= energyResolution)
				e = energyResolution - 1;
			data[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			data[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			data[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			data[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
