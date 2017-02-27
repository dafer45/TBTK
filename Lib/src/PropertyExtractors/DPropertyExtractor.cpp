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

#include "DPropertyExtractor.h"
#include "Functions.h"
#include "Streams.h"

using namespace std;

namespace TBTK{

namespace{
	complex<double> i(0,1);
}

DPropertyExtractor::DPropertyExtractor(DiagonalizationSolver *dSolver){
	this->dSolver = dSolver;
}

DPropertyExtractor::~DPropertyExtractor(){
}

void DPropertyExtractor::saveEigenValues(string path, string filename){
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
}

Property::EigenValues DPropertyExtractor::getEigenValues(){
	int size = dSolver->getModel()->getBasisSize();
	const double *ev = dSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	for(int n = 0; n < size; n++)
		eigenValues.data[n] = ev[n];

	return eigenValues;
}

Property::GreensFunction* DPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	unsigned int numPoles = dSolver->getModel()->getBasisSize();

	complex<double> *positions = new complex<double>[numPoles];
	complex<double> *amplitudes = new complex<double>[numPoles];
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		positions[n] = dSolver->getEigenValue(n);

		complex<double> uTo = dSolver->getAmplitude(n, to);
		complex<double> uFrom = dSolver->getAmplitude(n, from);

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
}

Property::DOS DPropertyExtractor::calculateDOS(){
	const double *ev = dSolver->getEigenValues();

//	Property::DOS *dos = new Property::DOS(lowerBound, upperBound, energyResolution);
	Property::DOS dos(lowerBound, upperBound, energyResolution);
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		int e = (int)(((ev[n] - lowerBound)/(upperBound - lowerBound))*energyResolution);
		if(e >= 0 && e < energyResolution){
//			dos->data[e] += 1.;
			dos.data[e] += 1.;
		}
	}

	return dos;
}

complex<double> DPropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Statistics statistics = dSolver->getModel()->getStatistics();

	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				dSolver->getEigenValue(n),
				dSolver->getModel()->getChemicalPotential(),
				dSolver->getModel()->getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				dSolver->getEigenValue(n),
				dSolver->getModel()->getChemicalPotential(),
				dSolver->getModel()->getTemperature()
			);
		}

		complex<double> u_to = dSolver->getAmplitude(n, to);
		complex<double> u_from = dSolver->getAmplitude(n, from);

		expectationValue += weight*conj(u_to)*u_from;
	}

	return expectationValue;
}

Property::Density DPropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density.data, pattern, ranges, 0, 1);

	return density;
}

Property::Magnetization DPropertyExtractor::calculateMagnetization(
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

	calculate(calculateMAGCallback, (void*)magnetization.data, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return magnetization;
}

Property::LDOS DPropertyExtractor::calculateLDOS(
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

	calculate(calculateLDOSCallback, (void*)ldos.data, pattern, ranges, 0, 1);

	return ldos;
}

Property::SpinPolarizedLDOS DPropertyExtractor::calculateSpinPolarizedLDOS(
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

	calculate(calculateSP_LDOSCallback, (void*)spinPolarizedLDOS.data, pattern, ranges, 0, 1);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}

void DPropertyExtractor::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void* density,
	const Index &index,
	int offset
){
	DPropertyExtractor *pe = (DPropertyExtractor*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();
	Statistics statistics = pe->dSolver->getModel()->getStatistics();
	for(int n = 0; n < pe->dSolver->getModel()->getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(eigen_values[n],
									pe->dSolver->getModel()->getChemicalPotential(),
									pe->dSolver->getModel()->getTemperature());
		}
		else{
			weight = Functions::boseEinsteinDistribution(eigen_values[n],
									pe->dSolver->getModel()->getChemicalPotential(),
									pe->dSolver->getModel()->getTemperature());
		}

		complex<double> u = pe->dSolver->getAmplitude(n, index);

		((double*)density)[offset] += pow(abs(u), 2)*weight;
	}
}

void DPropertyExtractor::calculateMAGCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	DPropertyExtractor *pe = (DPropertyExtractor*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();
	Statistics statistics = pe->dSolver->getModel()->getStatistics();

	int spin_index = ((int*)pe->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	for(int n = 0; n < pe->dSolver->getModel()->getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(eigen_values[n],
									pe->dSolver->getModel()->getChemicalPotential(),
									pe->dSolver->getModel()->getTemperature());
		}
		else{
			weight = Functions::boseEinsteinDistribution(eigen_values[n],
									pe->dSolver->getModel()->getChemicalPotential(),
									pe->dSolver->getModel()->getTemperature());
		}

		complex<double> u_u = pe->dSolver->getAmplitude(n, index_u);
		complex<double> u_d = pe->dSolver->getAmplitude(n, index_d);

		((complex<double>*)mag)[4*offset + 0] += conj(u_u)*u_u*weight;
		((complex<double>*)mag)[4*offset + 1] += conj(u_u)*u_d*weight;
		((complex<double>*)mag)[4*offset + 2] += conj(u_d)*u_u*weight;
		((complex<double>*)mag)[4*offset + 3] += conj(u_d)*u_d*weight;
	}
}

void DPropertyExtractor::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	DPropertyExtractor *pe = (DPropertyExtractor*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];

	double step_size = (u_lim - l_lim)/(double)resolution;

	for(int n = 0; n < pe->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u = pe->dSolver->getAmplitude(n, index);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)ldos)[resolution*offset + e] += real(conj(u)*u);
		}
	}
}

void DPropertyExtractor::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	DPropertyExtractor *pe = (DPropertyExtractor*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];
	int spin_index = ((int**)pe->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	for(int n = 0; n < pe->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u_u = pe->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->dSolver->getAmplitude(n, index_d);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((complex<double>*)sp_ldos)[4*resolution*offset + 4*e + 0] += conj(u_u)*u_u;
			((complex<double>*)sp_ldos)[4*resolution*offset + 4*e + 1] += conj(u_u)*u_d;
			((complex<double>*)sp_ldos)[4*resolution*offset + 4*e + 2] += conj(u_d)*u_u;
			((complex<double>*)sp_ldos)[4*resolution*offset + 4*e + 3] += conj(u_d)*u_d;
		}
	}
}

/*void DPropertyExtractor::calculate(
	void (*callback)(
		DPropertyExtractor *cb_this,
		void *memory,
		const Index &index,
		int offset
	),
	void *memory,
	Index pattern,
	const Index &ranges,
	int currentOffset,
	int offsetMultiplier
){
	int currentSubindex = pattern.size()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.at(currentSubindex); n++){
			pattern.at(currentSubindex) = n;
			calculate(callback,
					memory,
					pattern,
					ranges,
					currentOffset,
					nextOffsetMultiplier
			);
			if(!isSumIndex)
				currentOffset += offsetMultiplier;
		}
	}
}

void DPropertyExtractor::ensureCompliantRanges(
	const Index &pattern,
	Index &ranges
){
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) >= 0)
			ranges.at(n) = 1;
	}
}

void DPropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges,
	int *lDimensions,
	int **lRanges
){
	*lDimensions = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lDimensions)++;
	}

	(*lRanges) = new int[*lDimensions];
	int counter = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lRanges)[counter++] = ranges.at(n);
	}
}*/

};	//End of namespace TBTK
