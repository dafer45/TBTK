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

namespace TBTK{

class Geometry;

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
	/** Constructor. */
	Model();

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

	/** Returns true if the Hilbert space basis has been constructed. */
	bool getIsConstructed();

	/** Construct Hamiltonian on COO format. */
	void constructCOO();

	/** Destruct Hamiltonian on COO format. */
	void destructCOO();

	/** To be called when HoppingAmplitudes need to be reevaluated. This is
	 *  required if the AmplitudeSet in addition to its standard storage
	 *  format also utilizes a more effective format such as COO format and
	 *  some HoppingAmplitudes are evaluated through the use of callbacks. */
	void reconstructCOO();

	/** Set temperature. */
	void setTemperature(double temperature);

	/** Get temperature. */
	double getTemperature();

	/** Set chemical potential. */
	void setChemicalPotential(double chemicalPotential);

	/** Get chemical potential. */
	double getChemicalPotential();

	/** AmplitudeSet containing @link HoppingAmplitude HoppingAmplitudes
	 *  @endlink.*/
	AmplitudeSet amplitudeSet;

	/** Enums for Fermi-Dirac and Bose-Einstein statistics. */
	enum Statistics {FermiDirac, BoseEinstein};

	/** Set statistics. */
	void setStatistics(Statistics statistics);

	/** Get statistics. */
	Statistics getStatistics();

	/** Create geometry. */
	void createGeometry(int dimensions, int numSpecifiers = 0);

	/** Get geometry. */
	Geometry* getGeometry();

	void saveEV(std::string path = "./", std::string filename = "EV.dat");

	void setTalkative(bool isTalkative);
private:
	/** Temperature. */
	double temperature;

	/** Chemical potential. */
	double chemicalPotential;

	/** Statistics (Fermi-Dirac or Bose-Einstein). */
	Statistics statistics;

	/** Geometry. */
	Geometry *geometry;

	/** Flag indicating whether to write information to standard output or
	 *  not. */
	bool isTalkative;
};

inline void Model::addHA(HoppingAmplitude ha){
	amplitudeSet.addHA(ha);
}

inline void Model::addHAAndHC(HoppingAmplitude ha){
	amplitudeSet.addHAAndHC(ha);
}

inline int Model::getBasisSize(){
	return amplitudeSet.getBasisSize();
}

inline int Model::getBasisIndex(Index index){
	return amplitudeSet.getBasisIndex(index);
}

inline bool Model::getIsConstructed(){
	return amplitudeSet.getIsConstructed();
}

inline void Model::constructCOO(){
	amplitudeSet.sort();
	amplitudeSet.constructCOO();
}

inline void Model::destructCOO(){
	amplitudeSet.destructCOO();
}

inline void Model::reconstructCOO(){
	amplitudeSet.reconstructCOO();
}

inline void Model::setTemperature(double temperature){
	this->temperature = temperature;
}

inline double Model::getTemperature(){
	return temperature;
}

inline void Model::setChemicalPotential(double chemicalPotential){
	this->chemicalPotential = chemicalPotential;
}

inline double Model::getChemicalPotential(){
	return chemicalPotential;
}

inline void Model::setStatistics(Statistics statistics){
	this->statistics = statistics;
}

inline Model::Statistics Model::getStatistics(){
	return statistics;
}

inline Geometry* Model::getGeometry(){
	return geometry;
}

inline void Model::setTalkative(bool isTalkative){
	this->isTalkative = isTalkative;
}

};	//End of namespace TBTK

#endif

