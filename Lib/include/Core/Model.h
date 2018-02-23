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
 *  @brief Container of Model related information.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MODEL
#define COM_DAFER45_TBTK_MODEL

#include "AbstractHoppingAmplitudeFilter.h"
#include "AbstractIndexFilter.h"
#include "Communicator.h"
#include "Geometry.h"
#include "HoppingAmplitudeSet.h"
#include "IndexBasedHoppingAmplitudeFilter.h"
#include "SingleParticleContext.h"
#include "ManyBodyContext.h"
#include "Serializeable.h"
#include "Statistics.h"

#include <complex>
#include <fstream>
#include <string>
#include <tuple>

namespace TBTK{

class FileReader;

/** @brief Container of Model related information.
 *
 *  The Model conatins all model related information such as the Hamiltonian,
 *  temperature, and chemical potential.
 */
class Model : public Serializeable, public Communicator{
public:
	/** Constructor. */
	Model();

	/** Constructor. */
	Model(const std::vector<unsigned int> &capacity);

	/** Copy constructor. */
	Model(const Model &model);

	/** Move constructor. */
	Model(Model &&model);

	/** Constructor. Constructs the Model from a serialization string. Note
	 *  that the ManyBodyContext is not yet serialized. */
	Model(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~Model();

	/** Assignment operator. */
	Model& operator=(const Model &model);

	/** Move assignment operator. */
	Model& operator=(Model &&model);

	/** Add a HoppingAmplitude. */
	void addHoppingAmplitude(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate. */
	void addHoppingAmplitudeAndHermitianConjugate(HoppingAmplitude ha);

	/** Add a Model as a subsystem. */
	void addModel(const Model &model, const Index &subsytemIndex);

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *  @param index 'From'-index to get Hilbert space index for. */
	int getBasisIndex(const Index &index) const;

	/** Get size of Hilbert space. */
	int getBasisSize() const;

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
	double getTemperature() const;

	/** Set chemical potential. */
	void setChemicalPotential(double chemicalPotential);

	/** Get chemical potential. */
	double getChemicalPotential() const;

	/** Set statistics. */
	void setStatistics(Statistics statistics);

	/** Get statistics. */
	Statistics getStatistics() const;

	/** Get amplitude set. */
	const HoppingAmplitudeSet* getHoppingAmplitudeSet() const;

	/** Create geometry. */
	void createGeometry(int dimensions, int numSpecifiers = 0);

	/** Get geometry. */
	Geometry* getGeometry();

	/** Get geometry. */
	const Geometry* getGeometry() const;

	/** Create ManyBodyContext. */
	void createManyBodyContext();

	/** Get ManyBodyContext. */
	ManyBodyContext* getManyBodyContext();

	/** Set HoppingAmplitude filter. */
	void setFilter(
		const AbstractHoppingAmplitudeFilter &hoppingAmplitudeFilter
	);

	/** Set Index filter. */
	void setFilter(
		const AbstractIndexFilter &indexFilter
	);

//	void saveEV(std::string path = "./", std::string filename = "EV.dat");

	/** Operator<<. */
	Model& operator<<(const HoppingAmplitude& hoppingAmplitude);

	/** Operator<<. */
	Model& operator<<(
		const std::tuple<HoppingAmplitude, HoppingAmplitude> &hoppingAmplitudes
	);

	/** Implements Serializeable::serialize(). Note that the
	 *  ManyBodyContext is not yet serialized. */
	std::string serialize(Mode mode) const;
private:
	/** Temperature. */
	double temperature;

	/** Chemical potential. */
	double chemicalPotential;

	/** Single particle context. */
	SingleParticleContext *singleParticleContext;

	/** Many-body context. */
	ManyBodyContext *manyBodyContext;

	/** Hopping amplitude filter. */
	AbstractHoppingAmplitudeFilter *hoppingAmplitudeFilter;

	/** FileReader is a friend class to allow it to write Model data. */
	friend class FileReader;
};

inline void Model::addHoppingAmplitude(HoppingAmplitude ha){
	singleParticleContext->addHoppingAmplitude(ha);
}

inline void Model::addHoppingAmplitudeAndHermitianConjugate(
	HoppingAmplitude ha
){
	singleParticleContext->addHoppingAmplitudeAndHermitianConjugate(ha);
}

inline int Model::getBasisSize() const{
	return singleParticleContext->getBasisSize();
}

inline int Model::getBasisIndex(const Index &index) const{
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

inline double Model::getTemperature() const{
	return temperature;
}

inline void Model::setChemicalPotential(double chemicalPotential){
	this->chemicalPotential = chemicalPotential;
}

inline double Model::getChemicalPotential() const{
	return chemicalPotential;
}

inline void Model::setStatistics(Statistics statistics){
	singleParticleContext->setStatistics(statistics);
}

inline Statistics Model::getStatistics() const{
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

inline const Geometry* Model::getGeometry() const{
	return singleParticleContext->getGeometry();
}

inline void Model::createManyBodyContext(){
	manyBodyContext = new ManyBodyContext(singleParticleContext);
}

inline ManyBodyContext* Model::getManyBodyContext(){
	return manyBodyContext;
}

inline void Model::setFilter(
	const AbstractHoppingAmplitudeFilter &hoppingAmplitudeFilter
){
	if(this->hoppingAmplitudeFilter != nullptr)
		delete this->hoppingAmplitudeFilter;

	this->hoppingAmplitudeFilter = hoppingAmplitudeFilter.clone();
}

inline void Model::setFilter(
	const AbstractIndexFilter &indexFilter
){
	if(this->hoppingAmplitudeFilter != nullptr)
		delete this->hoppingAmplitudeFilter;

	this->hoppingAmplitudeFilter = new IndexBasedHoppingAmplitudeFilter(
		indexFilter
	);
}

inline Model& Model::operator<<(const HoppingAmplitude &hoppingAmplitude){
	if(
		hoppingAmplitudeFilter == nullptr
		|| hoppingAmplitudeFilter->isIncluded(hoppingAmplitude)
	){
		addHoppingAmplitude(hoppingAmplitude);
	}

	return *this;
}

inline Model& Model::operator<<(const std::tuple<HoppingAmplitude, HoppingAmplitude> &hoppingAmplitudes){
	if(
		hoppingAmplitudeFilter == nullptr
	){
		addHoppingAmplitude(std::get<0>(hoppingAmplitudes));
		addHoppingAmplitude(std::get<1>(hoppingAmplitudes));
	}
	else{
		if(
			hoppingAmplitudeFilter->isIncluded(
				std::get<0>(hoppingAmplitudes)
			)
		){
			addHoppingAmplitude(std::get<0>(hoppingAmplitudes));
		}
		if(
			hoppingAmplitudeFilter->isIncluded(
				std::get<1>(hoppingAmplitudes)
			)
		){
			addHoppingAmplitude(std::get<1>(hoppingAmplitudes));
		}
	}

	return *this;
}

};	//End of namespace TBTK

#endif
