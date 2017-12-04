/* Copyright 2017 Kristofer Björnson
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

/** @file BPropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "BPropertyExtractor.h"
#include "Functions.h"
#include "Streams.h"

using namespace std;

namespace TBTK{

namespace{
	complex<double> i(0,1);
}

BPropertyExtractor::BPropertyExtractor(BlockDiagonalizationSolver &bSolver){
	this->bSolver = &bSolver;
}

BPropertyExtractor::~BPropertyExtractor(){
}

/*void BPropertyExtractor::saveEigenValues(string path, string filename){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		fout << dSolver->getEigenValues()[n] << "\n";
	}
	fout.close();
}

void DPropertyExtractor::getTabulatedHoppingAmplitudeSet(
	complex<double> **amplitudes,
	int **indices,
	int *numHoppingAmplitudes,
	int *maxIndexSize
){
	dSolver->getModel()->getHoppingAmplitudeSet()->tabulate(
		amplitudes,
		indices,
		numHoppingAmplitudes,
		maxIndexSize
	);
}*/

Property::EigenValues BPropertyExtractor::getEigenValues(){
	int size = bSolver->getModel().getBasisSize();
//	const double *ev = bSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	double *data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = bSolver->getEigenValue(n);
//		data[n] = ev[n];

	return eigenValues;
}

Property::WaveFunction BPropertyExtractor::calculateWaveFunction(
	initializer_list<Index> patterns,
	initializer_list<int> states
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if(*states.begin() == IDX_ALL){
			for(int n = 0; n < bSolver->getModel().getBasisSize(); n++)
				statesVector.push_back(n);
		}
		else{
			TBTKAssert(
				*states.begin() >= 0,
				"BPropertyExtractor::calculateWaveFunction()",
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
				"BPropertyExtractor::calculateWaveFunction()",
				"Found unexpected index symbol.",
				"Use only positive numbers or '{IDX_ALL}'"
			);
			statesVector.push_back(*(states.begin() + n));
		}
	}

	Property::WaveFunction waveFunction(memoryLayout, statesVector);

	hint = new Property::WaveFunction*[1];
	((Property::WaveFunction**)hint)[0] = &waveFunction;

	calculate(
		calculateWaveFunctionCallback,
		allIndices,
		memoryLayout,
		waveFunction
	);

	delete [] (Property::WaveFunction**)hint;

	return waveFunction;
}

/*Property::GreensFunction* BPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	unsigned int numPoles = bSolver->getModel().getBasisSize();

	complex<double> *positions = new complex<double>[numPoles];
	complex<double> *amplitudes = new complex<double>[numPoles];
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		positions[n] = bSolver->getEigenValue(n);

		complex<double> uTo = bSolver->getAmplitude(n, to);
		complex<double> uFrom = bSolver->getAmplitude(n, from);

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

Property::DOS BPropertyExtractor::calculateDOS(){
//	const double *ev = bSolver->getEigenValues();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	double *data = dos.getDataRW();
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		int e = (int)(((bSolver->getEigenValue(n) - lowerBound)/(upperBound - lowerBound))*energyResolution);
//		int e = (int)(((ev[n] - lowerBound)/(upperBound - lowerBound))*energyResolution);
		if(e >= 0 && e < energyResolution){
			data[e] += 1./dE;
		}
	}

	return dos;
}

complex<double> BPropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Statistics statistics = bSolver->getModel().getStatistics();

	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				bSolver->getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				bSolver->getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
		}

		complex<double> u_to = bSolver->getAmplitude(n, to);
		complex<double> u_from = bSolver->getAmplitude(n, from);

		expectationValue += weight*conj(u_to)*u_from;
	}

	return expectationValue;
}

/*Property::Density DPropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density.getDataRW(), pattern, ranges, 0, 1);

	return density;
}*/

Property::Density BPropertyExtractor::calculateDensity(
	initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Density density(memoryLayout);

	calculate(
		calculateDensityCallback,
		allIndices,
		memoryLayout,
		density
	);

	return density;
}

/*Property::Magnetization DPropertyExtractor::calculateMagnetization(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		delete [] (int*)hint;
		TBTKExit(
			"DPropertyExtractor::calculateMagnetization()",
			"No spin index indiceated.",
			"Used IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization magnetization(lDimensions, lRanges);

	calculate(calculateMAGCallback, (void*)magnetization.getDataRW(), pattern, ranges, 0, 4);

	delete [] (int*)hint;

	return magnetization;
}*/

Property::Magnetization BPropertyExtractor::calculateMagnetization(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Magnetization magnetization(memoryLayout);

	hint = new int[1];
	calculate(
		calculateMagnetizationCallback,
		allIndices,
		memoryLayout,
		magnetization,
		(int*)hint
	);

	delete [] (int*)hint;

	return magnetization;
}

/*Property::LDOS BPropertyExtractor::calculateLDOS(
	Index pattern,
	Index ranges
){
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

	calculate(calculateLDOSCallback, (void*)ldos.getDataRW(), pattern, ranges, 0, energyResolution);

	return ldos;
}*/

Property::LDOS BPropertyExtractor::calculateLDOS(
	std::initializer_list<Index> patterns
){
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
		*bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
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

/*Property::SpinPolarizedLDOS DPropertyExtractor::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
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
	for(unsigned int n = 0; n < pattern.size(); n++){
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
			"DPropertyExtractor::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Used IDX_SPIN to indicate the position of the spin index."
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
		calculateSP_LDOSCallback,
		(void*)spinPolarizedLDOS.getDataRW(),
		pattern,
		ranges,
		0,
		4*energyResolution
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}*/

Property::SpinPolarizedLDOS BPropertyExtractor::calculateSpinPolarizedLDOS(
	std::initializer_list<Index> patterns
){
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
		*bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*bSolver->getModel().getHoppingAmplitudeSet(),
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
		calculateSP_LDOSCallback,
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

void BPropertyExtractor::calculateWaveFunctionCallback(
	PropertyExtractor *cb_this,
	void *waveFunction,
	const Index &index,
	int offset
){
	BPropertyExtractor *pe = (BPropertyExtractor*)cb_this;

	const vector<unsigned int> states = ((Property::WaveFunction**)pe->hint)[0]->getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		((complex<double>*)waveFunction)[offset + n] += pe->getAmplitude(states.at(n), index);
}

void BPropertyExtractor::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void* density,
	const Index &index,
	int offset
){
	BPropertyExtractor *pe = (BPropertyExtractor*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();
	Statistics statistics = pe->bSolver->getModel().getStatistics();
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}

		complex<double> u = pe->bSolver->getAmplitude(n, index);

		((double*)density)[offset] += pow(abs(u), 2)*weight;
	}
}

void BPropertyExtractor::calculateMagnetizationCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	BPropertyExtractor *pe = (BPropertyExtractor*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();
	Statistics statistics = pe->bSolver->getModel().getStatistics();

	int spin_index = ((int*)pe->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}

		complex<double> u_u = pe->bSolver->getAmplitude(n, index_u);
		complex<double> u_d = pe->bSolver->getAmplitude(n, index_d);

/*		((complex<double>*)mag)[offset + 0] += conj(u_u)*u_u*weight;
		((complex<double>*)mag)[offset + 1] += conj(u_u)*u_d*weight;
		((complex<double>*)mag)[offset + 2] += conj(u_d)*u_u*weight;
		((complex<double>*)mag)[offset + 3] += conj(u_d)*u_d*weight;*/
		((SpinMatrix*)mag)[offset].at(0, 0) += conj(u_u)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(0, 1) += conj(u_u)*u_d*weight;
		((SpinMatrix*)mag)[offset].at(1, 0) += conj(u_d)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(1, 1) += conj(u_d)*u_d*weight;
	}
}

void BPropertyExtractor::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	BPropertyExtractor *pe = (BPropertyExtractor*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];

	double step_size = (u_lim - l_lim)/(double)resolution;

	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > l_lim && eigenValue < u_lim){
			complex<double> u = pe->bSolver->getAmplitude(n, index);

			int e = (int)((eigenValue - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)ldos)[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void BPropertyExtractor::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	BPropertyExtractor *pe = (BPropertyExtractor*)cb_this;

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];
	int spin_index = ((int**)pe->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > l_lim && eigenValue < u_lim){
			complex<double> u_u = pe->bSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->bSolver->getAmplitude(n, index_d);

			int e = (int)((eigenValue - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace TBTK
