#ifndef COM_DAFER45_ARNOLDI_SOLVER
#define COM_DAFER45_ARNOLDI_SOLVER

#include "Model.h"

#include "slu_zdefs.h"

#include <complex>

namespace TBTK{

class ArnoldiSolver{
public:
	ArnoldiSolver();
	~ArnoldiSolver();

	void setModel(Model *model);

	void setNumEigenValues(int numEigenValues);

	void setNumLanczosVectors(int numLanczosVectors);

	void setTolerance(double tolerance);

	void setMaxIterations(int maxIterations);

	void run();
private:
	Model *model;

	//Arnoldi variables
	int numEigenValues;

	bool calculateEigenVectors;

	int numLanczosVectors;

	double shift;

	//Machine tolerance is used if tolerance <= 0
	double tolerance;

	int maxIterations;

	std::complex<double> *residuals;

	std::complex<double> *eigenValues;

	std::complex<double> *eigenVectors;

	//SuperLU variables
	SuperMatrix *hamiltonian;

	SuperMatrix *vector;		//b and x (in Ax = b)

	SuperMatrix *lowerTriangular;	//L (in LU decomposition of A)

	SuperMatrix *upperTriangular;	//U (in LU decomposition of A)

	int *rowPermutations;

	int *colPermutations;

	superlu_options_t *options;

	SuperLUStat_t *stat;

	int info;

	void init();

	void performLUFactorization();

	void arnoldiLoop();
};

inline void ArnoldiSolver::setModel(Model *model){
	this->model = model;
}

inline void ArnoldiSolver::setNumEigenValues(int numEigenValues){
	this->numEigenValues = numEigenValues;
}

inline void ArnoldiSolver::setNumLanczosVectors(int numLanczosVectors){
	this->numLanczosVectors = numLanczosVectors;
}

inline void ArnoldiSolver::setTolerance(double tolerance){
	this->tolerance = tolerance;
}

inline void ArnoldiSolver::setMaxIterations(int maxIterations){
	this->maxIterations = maxIterations;
}

};	//End of namesapce TBTK

#endif
