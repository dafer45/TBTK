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

/** The Model conatins all information about the Hamiltonian. It is currently a
 *  wrapper for AmplitudeSet, but can in the future come to be extended with
 *  further properties.
 */
class Model{
public:
	enum {MODE_ALL_EIGENVECTORS, MODE_SOME_EIGENVECTORS, MODE_CHEBYSHEV};

	/** Constructor. */
	Model(int mode = MODE_ALL_EIGENVECTORS, int numEigenstates = 0);

	/** Destructor. */
	~Model();

	/** Add a HoppingAmplitude. */
	void addHA(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate. */
	void addHAAndHC(HoppingAmplitude ha);

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *  @param index 'From'-index to get Hilbert space index for. */
	int getBasisIndex(Index index);

	/** Get size of Hilbert space. */
	int getBasisSize();

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** AmplitudeSet containing @link HoppingAmplitude HoppingAmplitudes
	 *  @endlink.*/
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

