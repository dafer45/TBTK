#ifndef COM_DAFER45_TBTK_MODEL
#define COM_DAFER45_TBTK_MODEL

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

	void construct();

/*	void setSCCallback(bool (*scCallback)(Model *model));
	void setMaxIterations(int maxIterations);

	void update();
	void solve();

	void run();

	const double* getEigenValues();//{	return eigen_values;	}
	const std::complex<double>* getEigenVectors();//{	return eigen_vectors;	};
	const std::complex<double> getAmplitude(int state, const Index &index);*/

	void print();
	AmplitudeSet amplitudeSet;

	void saveEV(std::string path = "./", std::string filename = "EV.dat");
private:
	int mode;
	int numEigenstates;

/*	std::complex<double>* hamiltonian;
	double* eigen_values;
	std::complex<double>* eigen_vectors;

	int maxIterations;
	bool (*scCallback)(Model *model);*/
};

inline int Model::getBasisSize(){
	return amplitudeSet.getBasisSize();
}

/*inline void Model::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

inline const std::complex<double> Model::getAmplitude(int state, const Index &index){
	return eigen_vectors[getBasisSize()*state + getBasisIndex(index)];
}*/

inline int Model::getBasisIndex(Index index){
	return amplitudeSet.getBasisIndex(index);
}

/*inline const double* Model::getEigenValues(){
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
}*/

#endif

