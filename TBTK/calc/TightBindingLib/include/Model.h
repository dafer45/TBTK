#ifndef COM_DAFER45_BAND_STRUCTURE_SYSTEM
#define COM_DAFER45_BAND_STRUCTURE_SYSTEM

#include <complex>
#include "AmplitudeSet.h"
#include <string>
#include <fstream>

enum {IDX_SUM_ALL = -1, IDX_ALL = -1,
	IDX_X = -2,
	IDX_Y = -3,
	IDX_Z = -4,
	IDX_SPIN = -5};

class Model{
public:
	enum {MODE_ALL_EIGENVECTORS, MODE_SOME_EIGENVECTORS, MODE_CHEBYSHEV};

	Model(int mode = MODE_ALL_EIGENVECTORS, int numEigenstates = 0);
	~Model();

	void addHA(HoppingAmplitude ha);
	void addHAAndHC(HoppingAmplitude ha);
	int getBasisIndex(Index index);
	int getBasisSize();

	void setSCCallback(bool (*scCallback)(Model *model));
	void setMaxIterations(int maxIterations);

	void construct();
	void update();
	void solve();

	void run();

	const double* getEigenValues();//{	return eigen_values;	}
	const std::complex<double>* getEigenVectors();//{	return eigen_vectors;	};
	const std::complex<double> getAmplitude(int state, const Index &index);

	void print();
	AmplitudeSet amplitudeSet;

	void saveEV(std::string path = "./", std::string filename = "EV.dat");
/*	void saveDOS(Index pattern, Index range, std::string path = "./", std::string filename = "DOS.dat");
	void saveDOS2D(Index pattern, Index range, std::string path = "./", std::string filename = "DOS.dat");
	void saveCustom(void (*callback)(System *system, double *memory, Index index, int offset),
			Index pattern, Index range, std::string path = "./", std::string filename = "custom.dat");
	void saveCustom2D(void (*callback)(System *system, double *memory, Index index, int offset),
			Index pattern, Index range, std::string path = "./", std::string filename = "custom.dat");*/
private:
	int mode;
	int numEigenstates;

	std::complex<double>* hamiltonian;
	double* eigen_values;
	std::complex<double>* eigen_vectors;

	int maxIterations;
	bool (*scCallback)(Model *model);

/*	void calculateDOS(double *dos, Index pattern, Index limits, int currentOffset, int offsetMultiplier);
	void printDOS(std::ofstream &fout, Index pattern, Index limits);*/

/*	void calculate(void (*callback)(System *system, double *memory, Index index, int offset),
			double *dos, Index pattern, Index limits, int currentOffset, int offsetMultiplier);
	void save(void (*callback)(System *system, double *memory, Index index, int offset),
			Index pattern, Index range, std::string path, std::string filename);
	void save2D(void (*callback)(System *system, double *memory, Index index, int offset),
			Index pattern, Index range, std::string path, std::string filename);
	static void calculateDOSCallback(System *system, double* dos, Index index, int offset);*/
};

inline int Model::getBasisSize(){
	return amplitudeSet.getBasisSize();
}

inline void Model::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

inline const std::complex<double> Model::getAmplitude(int state, const Index &index){
	return eigen_vectors[getBasisSize()*state + getBasisIndex(index)];
}

inline int Model::getBasisIndex(Index index){
	return amplitudeSet.getBasisIndex(index);
}

inline const double* Model::getEigenValues(){
	if(mode == MODE_ALL_EIGENVECTORS){
		return eigen_values;
	}
	else{
		std::cout << "Error in System::getEigenValues(): Eigenvalues cannot be accessed with this method in MODE_SOME_EIGENVECTORS\n";
		exit(1);
	}
}
inline const std::complex<double>* Model::getEigenVectors(){
	if(mode == MODE_ALL_EIGENVECTORS){
		return eigen_vectors;
	}
	else{
		std::cout << "Error in System::getEigenValues(): Eigenvectors cannot be accessed with this method in MODE_SOME_EIGENVECTORS";
		exit(1);
	}
}

#endif

