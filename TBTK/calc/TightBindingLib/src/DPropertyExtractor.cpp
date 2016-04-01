/** @file DPropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/DPropertyExtractor.h"
#include <iostream>

using namespace std;

namespace TBTK{

complex<double> i(0,1);

DPropertyExtractor::DPropertyExtractor(DiagonalizationSolver *dSolver){
	this->dSolver = dSolver;
}

DPropertyExtractor::~DPropertyExtractor(){
}

void DPropertyExtractor::save(int *memory, int rows, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int r = 0; r < rows; r++){
		for(int c = 0; c < columns; c++){
			if(c != 0)
				fout << "\t";
			fout << memory[columns*r + c];
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::save(double *memory, int rows, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int r = 0; r < rows; r++){
		for(int c = 0; c < columns; c++){
			if(c != 0)
				fout << "\t";
			fout << memory[columns*r + c];
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::save(complex<double> *memory, int rows, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int r = 0; r < rows; r++){
		for(int c = 0; c < columns; c++){
			if(c != 0)
				fout << "\t";
			fout << abs(memory[columns*r + c]) << " " << arg(memory[columns*r + c]);
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::save2D(int *memory, int size_x, int size_y, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save2D: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int y = 0; y < size_y; y++){
		for(int x = 0; x < size_x; x++){
			for(int c = 0; c < columns; c++){
				if(c != 0)
					fout << "\t";
				fout << memory[columns*(x + size_x*y) + c];
			}
			fout << "\n";
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::save2D(double *memory, int size_x, int size_y, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save2D: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int y = 0; y < size_y; y++){
		for(int x = 0; x < size_x; x++){
			for(int c = 0; c < columns; c++){
				if(c != 0)
					fout << "\t";
				fout << memory[columns*(x + size_x*y) + c];
			}
			fout << "\n";
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::save2D(complex<double> *memory, int size_x, int size_y, int columns, string filename, string path){
	if(memory == NULL){
		cout << "Error in PropertyExtractor::save2D: memory is NULL.\n";
		return;
	}

	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int y = 0; y < size_y; y++){
		for(int x = 0; x < size_x; x++){
			for(int c = 0; c < columns; c++){
				if(c != 0)
					fout << "\t";
				fout << abs(memory[columns*(x + size_x*y) + c]) << " " << arg(memory[columns*(x + size_x*y) + c]);
			}
			fout << "\n";
		}
		fout << "\n";
	}
	fout.close();
}

void DPropertyExtractor::saveEV(string path, string filename){
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

void DPropertyExtractor::getTabulatedAmplitudeSet(int **table, int *dims){
	dSolver->getModel()->amplitudeSet.tabulate(table, dims);
}

double* DPropertyExtractor::getEV(){
	double *ev = new double[dSolver->getModel()->getBasisSize()];
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++)
		ev[n] = dSolver->getEigenValues()[n];
	return ev;
}

double* DPropertyExtractor::calculateDOS(double u_lim, double l_lim, int resolution){
	const double *ev = dSolver->getEigenValues();

	double *dos = new double[resolution];
	for(int n = 0; n < resolution; n++)
		dos[n] = 0.;
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		int e = (int)(((ev[n] - l_lim)/(u_lim - l_lim))*resolution);
		if(e >= 0 && e < resolution){
			dos[e] += 1.;
		}
	}

	return dos;
}

double* DPropertyExtractor::calculateDensity(Index pattern, Index ranges){
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int densityArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			densityArraySize *= ranges.indices.at(n);
	}
	double *density = new double[densityArraySize];
	for(int n = 0; n < densityArraySize; n++)
		density[n] = 0.;
	calculate(calculateDensityCallback, (void*)density, pattern, ranges, 0, 1);

	return density;
}

/*double* DPropertyExtractor::calculateMAG(Index pattern, Index ranges){
	hint = new int[1];
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		cout << "Error in PropertyExtractor::calculateMAG: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int magArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			magArraySize *= ranges.indices.at(n);
	}
	double *mag = new double[3*magArraySize];
	for(int n = 0; n < 3*magArraySize; n++)
		mag[n] = 0;
	calculate(calculateMAGCallback, (void*)mag, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return mag;
}*/

complex<double>* DPropertyExtractor::calculateMAG(Index pattern, Index ranges){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		cout << "Error in PropertyExtractor::calculateMAG: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int magArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			magArraySize *= ranges.indices.at(n);
	}
	complex<double> *mag = new complex<double>[4*magArraySize];
	for(int n = 0; n < 4*magArraySize; n++)
		mag[n] = 0;
	calculate(calculateMAGCallback, (void*)mag, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return mag;
}

/*double* DPropertyExtractor::calculateSP_LDOS(Index pattern, Index ranges, double u_lim, double l_lim, int resolution){
	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: u_lim
	//hint[0][1]: l_lim
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = u_lim;
	((double**)hint)[0][1] = l_lim;
	((int**)hint)[1][0] = resolution;

	((int**)hint)[1][1] = -1;
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int**)hint)[1][1] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int**)hint)[1][1] == -1){
		cout << "Error in PropertyExtractor::calculateSP_LDOS_E: No spin index indicated.\n";
		delete [] ((double**)hint)[0];
		delete [] ((int**)hint)[1];
		delete [] (void**)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int sp_ldosArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			sp_ldosArraySize *= ranges.indices.at(n);
	}
	double *sp_ldos = new double[6*resolution*sp_ldosArraySize];
	for(int n = 0; n < 6*resolution*sp_ldosArraySize; n++)
		sp_ldos[n] = 0;
	calculate(calculateSP_LDOSCallback, (void*)sp_ldos, pattern, ranges, 0, 1);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return sp_ldos;
}*/

complex<double>* DPropertyExtractor::calculateSP_LDOS(Index pattern, Index ranges, double u_lim, double l_lim, int resolution){
	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: u_lim
	//hint[0][1]: l_lim
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = u_lim;
	((double**)hint)[0][1] = l_lim;
	((int**)hint)[1][0] = resolution;

	((int**)hint)[1][1] = -1;
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int**)hint)[1][1] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int**)hint)[1][1] == -1){
		cout << "Error in PropertyExtractor::calculateSP_LDOS_E: No spin index indicated.\n";
		delete [] ((double**)hint)[0];
		delete [] ((int**)hint)[1];
		delete [] (void**)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int sp_ldosArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			sp_ldosArraySize *= ranges.indices.at(n);
	}
	complex<double> *sp_ldos = new complex<double>[4*resolution*sp_ldosArraySize];
	for(int n = 0; n < 4*resolution*sp_ldosArraySize; n++)
		sp_ldos[n] = 0;
	calculate(calculateSP_LDOSCallback, (void*)sp_ldos, pattern, ranges, 0, 1);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return sp_ldos;
}

void DPropertyExtractor::calculateDensityCallback(DPropertyExtractor *cb_this, void* density, const Index &index, int offset){
	const double *eigen_values = cb_this->dSolver->getEigenValues();
	for(int n = 0; n < cb_this->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] < 0){
			complex<double> u = cb_this->dSolver->getAmplitude(n, index);
			((double*)density)[offset] += pow(abs(u), 2);
		}
	}
}

/*void DPropertyExtractor::calculateMAGCallback(DPropertyExtractor *cb_this, void *mag, const Index &index, int offset){
	const double *eigen_values = cb_this->dSolver->getEigenValues();

	int spin_index = ((int*)cb_this->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] < 0){
			complex<double> u_u = cb_this->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->dSolver->getAmplitude(n, index_d);

			((double*)mag)[3*offset + 0] += real(conj(u_u)*u_d + conj(u_d)*u_u);
			((double*)mag)[3*offset + 1] += imag(-conj(u_u)*u_d + conj(u_d)*u_u);
			((double*)mag)[3*offset + 2] += real(conj(u_u)*u_u - conj(u_d)*u_d);
		}
	}
}*/

void DPropertyExtractor::calculateMAGCallback(DPropertyExtractor *cb_this, void *mag, const Index &index, int offset){
	const double *eigen_values = cb_this->dSolver->getEigenValues();

	int spin_index = ((int*)cb_this->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] < 0){
			complex<double> u_u = cb_this->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->dSolver->getAmplitude(n, index_d);

			((complex<double>*)mag)[4*offset + 0] += conj(u_u)*u_u;
			((complex<double>*)mag)[4*offset + 1] += conj(u_u)*u_d;
			((complex<double>*)mag)[4*offset + 2] += conj(u_d)*u_u;
			((complex<double>*)mag)[4*offset + 3] += conj(u_d)*u_d;
		}
	}
}

/*void DPropertyExtractor::calculateSP_LDOSCallback(DPropertyExtractor *cb_this, void *sp_ldos, const Index &index, int offset){
	const double *eigen_values = cb_this->dSolver->getEigenValues();

	double u_lim = ((double**)cb_this->hint)[0][0];
	double l_lim = ((double**)cb_this->hint)[0][1];
	int resolution = ((int**)cb_this->hint)[1][0];
	int spin_index = ((int**)cb_this->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u_u = cb_this->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->dSolver->getAmplitude(n, index_d);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)sp_ldos)[6*resolution*offset + 6*e + 0] += abs(u_u+u_d)*abs(u_u+u_d)/2.;
			((double*)sp_ldos)[6*resolution*offset + 6*e + 1] += abs(u_u-u_d)*abs(u_u-u_d)/2.;
			((double*)sp_ldos)[6*resolution*offset + 6*e + 2] += abs(u_u-i*u_d)*abs(u_u-i*u_d)/2.;
			((double*)sp_ldos)[6*resolution*offset + 6*e + 3] += abs(u_u+i*u_d)*abs(u_u+i*u_d)/2.;
			((double*)sp_ldos)[6*resolution*offset + 6*e + 4] += real(conj(u_u)*u_u);
			((double*)sp_ldos)[6*resolution*offset + 6*e + 5] += real(conj(u_d)*u_d);
		}
	}
}*/

void DPropertyExtractor::calculateSP_LDOSCallback(DPropertyExtractor *cb_this, void *sp_ldos, const Index &index, int offset){
	const double *eigen_values = cb_this->dSolver->getEigenValues();

	double u_lim = ((double**)cb_this->hint)[0][0];
	double l_lim = ((double**)cb_this->hint)[0][1];
	int resolution = ((int**)cb_this->hint)[1][0];
	int spin_index = ((int**)cb_this->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->dSolver->getModel()->getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u_u = cb_this->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->dSolver->getAmplitude(n, index_d);

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

/*void DPropertyExtractor::calculate(void (*callback)(DPropertyExtractor *cb_this, void *memory, const Index &index, int offset),
			void *memory, Index pattern, const Index &ranges, int currentOffset, int offsetMultiplier){
	unsigned int currentSubindex = 0;
	for(; currentSubindex < pattern.indices.size(); currentSubindex++){
		if(pattern.indices.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == pattern.indices.size()){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.indices.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.indices.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.indices.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.indices.at(currentSubindex); n++){
			pattern.indices.at(currentSubindex) = n;
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
}*/

void DPropertyExtractor::calculate(void (*callback)(DPropertyExtractor *cb_this, void *memory, const Index &index, int offset),
			void *memory, Index pattern, const Index &ranges, int currentOffset, int offsetMultiplier){
	int currentSubindex = pattern.indices.size()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.indices.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.indices.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.indices.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.indices.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.indices.at(currentSubindex); n++){
			pattern.indices.at(currentSubindex) = n;
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

};
