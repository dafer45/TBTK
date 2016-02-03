#ifndef COM_DAFER45_TBTK_DIAGONALIZATION_SOLVER
#define COM_DAFER45_TBTK_DIAGONALIZATION_SOLVER

#include "Model.h"
#include <complex>

class DiagonalizationSolver{
public:
	DiagonalizationSolver();
	~DiagonalizationSolver();

	void setModel(Model *model);

	void setSCCallback(bool (*scCallback)(DiagonalizationSolver *diagonalizationSolver));
	void setMaxIterations(int maxIterations);

	void run();

	const double* getEigenValues();
	const std::complex<double>* getEigenVectors();
	const std::complex<double> getAmplitude(int state, const Index &index);

	Model *getModel();
private:
	Model *model;

	std::complex<double> *hamiltonian;
	double *eigenValues;
	std::complex<double> *eigenVectors;

	int maxIterations;
	bool (*scCallback)(DiagonalizationSolver *diagonalizationSolver);

	void init();
	void update();
	void solve();
};

inline void DiagonalizationSolver::setModel(Model *model){
	this->model = model;
}

inline void DiagonalizationSolver::setSCCallback(bool (*scCallback)(DiagonalizationSolver *diagonalizationSolver)){
	this->scCallback = scCallback;
}

inline void DiagonalizationSolver::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

inline const double* DiagonalizationSolver::getEigenValues(){
	return eigenValues;
}

inline const std::complex<double>* DiagonalizationSolver::getEigenVectors(){
	return eigenVectors;
}

inline const std::complex<double> DiagonalizationSolver::getAmplitude(int state, const Index &index){
	return eigenVectors[model->getBasisSize()*state + model->getBasisIndex(index)];
}

inline Model* DiagonalizationSolver::getModel(){
	return model;
}

#endif

