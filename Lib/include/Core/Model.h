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

/** @package TBTKcalc
 *  @file Model.h
 *  @brief Model Hamiltonian
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MODEL
#define COM_DAFER45_TBTK_MODEL

#include "Geometry.h"
#include "HoppingAmplitudeSet.h"
#include "SingleParticleContext.h"
#include "ManyBodyContext.h"
#include "Statistics.h"

#include <complex>
#include <fstream>
#include <string>

namespace TBTK{

class FileReader;

/** The Model conatins all information about the Hamiltonian. It is currently a
 *  wrapper for HoppingAmplitudeSet, but can in the future come to be extended
 *  with further properties.
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

	/** Sort HoppingAmplitudes. */
	void sortHoppingAmplitudes();

	/** Construct Hamiltonian on COO format. */
	void constructCOO();

	/** Destruct Hamiltonian on COO format. */
	void destructCOO();

	/** To be called when HoppingAmplitudes need to be reevaluated. This is
	 *  required if the HoppingAmplitudeSet in addition to its standard
	 *  storage format also utilizes a more effective format such as COO
	 *  format and some HoppingAmplitudes are evaluated through the use of
	 *  callbacks. */
	void reconstructCOO();

	/** Set temperature. */
	void setTemperature(double temperature);

	/** Get temperature. */
	double getTemperature();

	/** Set chemical potential. */
	void setChemicalPotential(double chemicalPotential);

	/** Get chemical potential. */
	double getChemicalPotential();

	/** Set statistics. */
	void setStatistics(Statistics statistics);

	/** Get statistics. */
	Statistics getStatistics();

	/** Get amplitude set. */
	const HoppingAmplitudeSet* getHoppingAmplitudeSet() const;

	/** Create geometry. */
	void createGeometry(int dimensions, int numSpecifiers = 0);

	/** Get geometry. */
	Geometry* getGeometry();

	/** Create ManyBodyContext. */
	void createManyBodyContext();

	/** Get ManyBodyContext. */
	ManyBodyContext* getManyBodyContext();

	void saveEV(std::string path = "./", std::string filename = "EV.dat");

	void setTalkative(bool isTalkative);
private:
	/** Temperature. */
	double temperature;

	/** Chemical potential. */
	double chemicalPotential;

	/** Single particle context. */
	SingleParticleContext *singleParticleContext;

	/** Many-body context. */
	ManyBodyContext *manyBodyContext;

	/** Flag indicating whether to write information to standard output or
	 *  not. */
	bool isTalkative;

	/** FileReader is a friend class to allow it to write Model data. */
	friend class FileReader;
};

inline void Model::addHA(HoppingAmplitude ha){
	singleParticleContext->addHA(ha);
}

inline void Model::addHAAndHC(HoppingAmplitude ha){
	singleParticleContext->addHAAndHC(ha);
}

inline int Model::getBasisSize(){
	return singleParticleContext->getBasisSize();
}

inline int Model::getBasisIndex(Index index){
	return singleParticleContext->getBasisIndex(index);
}

inline bool Model::getIsConstructed(){
	return singleParticleContext->getIsConstructed();
}

inline void Model::sortHoppingAmplitudes(){
	singleParticleContext->sortHoppingAmplitudes();
}

inline void Model::constructCOO(){
	singleParticleContext->constructCOO();
}

inline void Model::destructCOO(){
	singleParticleContext->destructCOO();
}

inline void Model::reconstructCOO(){
	singleParticleContext->reconstructCOO();
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
	singleParticleContext->setStatistics(statistics);
}

inline Statistics Model::getStatistics(){
	return singleParticleContext->getStatistics();
}

inline const HoppingAmplitudeSet* Model::getHoppingAmplitudeSet() const{
	return singleParticleContext->getHoppingAmplitudeSet();
}

inline void Model::createGeometry(int dimensions, int numSpecifiers){
	singleParticleContext->createGeometry(dimensions, numSpecifiers);
}

inline Geometry* Model::getGeometry(){
	return singleParticleContext->getGeometry();
}

inline void Model::createManyBodyContext(){
	manyBodyContext = new ManyBodyContext(singleParticleContext);
}

inline ManyBodyContext* Model::getManyBodyContext(){
	return manyBodyContext;
}

inline void Model::setTalkative(bool isTalkative){
	this->isTalkative = isTalkative;
}

};	//End of namespace TBTK

#endif
