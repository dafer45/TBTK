/** @package TBTKcalc
 *  @file Model.h
 *  @brief Model Hamiltonian
 *
 *  @author Kristofer Björnson
 */

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

	void print();
	AmplitudeSet amplitudeSet;

	void saveEV(std::string path = "./", std::string filename = "EV.dat");
private:
	int mode;
	int numEigenstates;
};

inline int Model::getBasisSize(){
	return amplitudeSet.getBasisSize();
}

inline int Model::getBasisIndex(Index index){
	return amplitudeSet.getBasisIndex(index);
}

#endif

