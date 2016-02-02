#include "../include/PropertyExtractor.h"
#include <iostream>

using namespace std;

complex<double> i(0,1);

PropertyExtractor::PropertyExtractor(System *system){
	this->system = system;
}

PropertyExtractor::~PropertyExtractor(){
}

void PropertyExtractor::save(int *memory, int rows, int columns, string filename, string path){
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

void PropertyExtractor::save(double *memory, int rows, int columns, string filename, string path){
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

void PropertyExtractor::save(complex<double> *memory, int rows, int columns, string filename, string path){
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

void PropertyExtractor::save2D(int *memory, int size_x, int size_y, int columns, string filename, string path){
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

void PropertyExtractor::save2D(double *memory, int size_x, int size_y, int columns, string filename, string path){
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

void PropertyExtractor::save2D(complex<double> *memory, int size_x, int size_y, int columns, string filename, string path){
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

void PropertyExtractor::saveEV(string path, string filename){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int n = 0; n < system->getBasisSize(); n++){
		fout << system->getEigenValues()[n] << "\n";
	}
	fout.close();
}

void PropertyExtractor::getTabulatedAmplitudeSet(int **table, int *dims){
	system->amplitudeSet.tabulate(table, dims);
}

double* PropertyExtractor::getEV(){
	double *ev = new double[system->getBasisSize()];
	for(int n = 0; n < system->getBasisSize(); n++)
		ev[n] = system->getEigenValues()[n];
	return ev;
}

double* PropertyExtractor::calculateDOS(double u_lim, double l_lim, int resolution){
	const double *ev = system->getEigenValues();

	double *dos = new double[resolution];
	for(int n = 0; n < resolution; n++)
		dos[n] = 0.;
	for(int n = 0; n < system->getBasisSize(); n++){
		int e = (int)(((ev[n] - l_lim)/(u_lim - l_lim))*resolution);
		if(e >= 0 && e < resolution){
			dos[e] += 1.;
		}
	}

	return dos;
}

double* PropertyExtractor::calculateLDOS(Index pattern, Index ranges){
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int ldosArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			ldosArraySize *= ranges.indices.at(n);
	}
	double *ldos = new double[ldosArraySize];
	for(int n = 0; n < ldosArraySize; n++)
		ldos[n] = 0.;
	calculate(calculateLDOSCallback, (void*)ldos, pattern, ranges, 0, 1);

	return ldos;
}

double* PropertyExtractor::calculateMAG(Index pattern, Index ranges){
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
}

double* PropertyExtractor::calculateSP_LDOS_E(Index pattern, Index ranges, double u_lim, double l_lim, int resolution){
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

	int sp_ldos_eArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			sp_ldos_eArraySize *= ranges.indices.at(n);
	}
	double *sp_ldos_e = new double[6*resolution*sp_ldos_eArraySize];
	for(int n = 0; n < 6*resolution*sp_ldos_eArraySize; n++)
		sp_ldos_e[n] = 0;
	calculate(calculateSP_LDOS_ECallback, (void*)sp_ldos_e, pattern, ranges, 0, 1);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return sp_ldos_e;
}

void PropertyExtractor::calculateLDOSCallback(PropertyExtractor *cb_this, void* ldos, const Index &index, int offset){
	if(index.indices.back() > 1)
		return;
	const double *eigen_values = cb_this->system->getEigenValues();
	for(int n = 0; n < cb_this->system->getBasisSize(); n++){
		if(eigen_values[n] < 0){
			complex<double> u = cb_this->system->getAmplitude(n, index);
			((double*)ldos)[offset] += pow(abs(u), 2);
		}
	}
}

void PropertyExtractor::calculateMAGCallback(PropertyExtractor *cb_this, void *mag, const Index &index, int offset){
	const double *eigen_values = cb_this->system->getEigenValues();

	int spin_index = ((int*)cb_this->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->system->getBasisSize(); n++){
		if(eigen_values[n] < 0){
			complex<double> u_u = cb_this->system->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->system->getAmplitude(n, index_d);

			((double*)mag)[3*offset + 0] += real(conj(u_u)*u_d + conj(u_d)*u_u);
			((double*)mag)[3*offset + 1] += imag(-conj(u_u)*u_d + conj(u_d)*u_u);
			((double*)mag)[3*offset + 2] += real(conj(u_u)*u_u - conj(u_d)*u_d);
		}
	}
}

void PropertyExtractor::calculateSP_LDOS_ECallback(PropertyExtractor *cb_this, void *sp_ldos_e, const Index &index, int offset){
	const double *eigen_values = cb_this->system->getEigenValues();

	double u_lim = ((double**)cb_this->hint)[0][0];
	double l_lim = ((double**)cb_this->hint)[0][1];
	int resolution = ((int**)cb_this->hint)[1][0];
	int spin_index = ((int**)cb_this->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.indices.at(spin_index) = 0;
	index_d.indices.at(spin_index) = 1;
	for(int n = 0; n < cb_this->system->getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u_u = cb_this->system->getAmplitude(n, index_u);
			complex<double> u_d = cb_this->system->getAmplitude(n, index_d);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 0] += abs(u_u+u_d)*abs(u_u+u_d)/2.;
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 1] += abs(u_u-u_d)*abs(u_u-u_d)/2.;
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 2] += abs(u_u-i*u_d)*abs(u_u-i*u_d)/2.;
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 3] += abs(u_u+i*u_d)*abs(u_u+i*u_d)/2.;
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 4] += real(conj(u_u)*u_u);
			((double*)sp_ldos_e)[6*resolution*offset + 6*e + 5] += real(conj(u_d)*u_d);
		}
	}
}

void PropertyExtractor::calculate(void (*callback)(PropertyExtractor *cb_this, void *memory, const Index &index, int offset),
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
}

