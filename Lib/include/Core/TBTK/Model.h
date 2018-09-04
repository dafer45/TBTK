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

#include "TBTK/AbstractHoppingAmplitudeFilter.h"
#include "TBTK/AbstractIndexFilter.h"
#include "TBTK/Communicator.h"
#include "TBTK/Geometry.h"
#include "TBTK/HoppingAmplitudeList.h"
#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/IndexBasedHoppingAmplitudeFilter.h"
#include "TBTK/SingleParticleContext.h"
#include "TBTK/ManyParticleContext.h"
#include "TBTK/Serializable.h"
#include "TBTK/Statistics.h"

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
class Model : public Serializable, public Communicator{
public:
	/** Constructor. */
	Model();

	/** Constructs a Model with a preallocated storage structure such that
	 *  the addition of HoppingAmplitudes with indices that have the same
	 *  subindex structure as 'capacity', but with smaller subindices will
	 *  not cause reallocation for the main storage structure. Internal
	 *  containers for HoppingAmplitudes may still be reallocated.
	 *
	 *  @param capacity The 'Index capacity'. */
	Model(const std::vector<unsigned int> &capacity);

	/** Copy constructor.
	 *
	 *  @param model Model to copy. */
	Model(const Model &model);

	/** Move constructor.
	 *
	 *  @param model Model to move. */
	Model(Model &&model);

	/** Constructor. Constructs the Model from a serialization string. Note
	 *  that the ManyParticleContext is not yet serialized.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Index.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Model(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~Model();

	/** Assignment operator.
	 *
	 *  @param rhs Model to assign to the left hand side.
	 *
	 *  @return Reference to the assigned Model. */
	Model& operator=(const Model &rhs);

	/** Move assignment operator.
	 *
	 *  @param rhs Model to assign to the left hand side.
	 *
	 *  @return Reference to the assigned Model. */
	Model& operator=(Model &&rhs);

	/** Add a HoppingAmplitude.
	 *
	 @param ha HoppingAmplitude to add. */
	void add(HoppingAmplitude ha);

	/** Add a Model as a subsystem.
	 *
	 *  @param model Model to include as subsystem.
	 *  @param subsystemIndex Index that will be prepended to each Index in
	 *  the model. */
	void addModel(const Model &model, const Index &subsytemIndex);

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *
	 *  @param index Physical Index for which to obtain the Hilbert space
	 *  index.
	 *
	 *  @return The Hilbert space index corresponding to the given Physical
	 *  Index. Returns -1 if Model::construct() has not been called. */
	int getBasisIndex(const Index &index) const;

	/** Get size of Hilbert space.
	 *
	 *  @return The basis size if the basis has been constructed using the
	 *  call to Model::construct(), otherwise -1. */
	int getBasisSize() const;

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/** Check whether the Hilbert space basis has been constructed.
	 *
	 *  @return True if the Hilbert space basis has been constructed. */
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

	/** Set temperature.
	 *
	 *  @param temperature The temperature. */
	void setTemperature(double temperature);

	/** Get temperature.
	 *
	 *  @return The temperature. */
	double getTemperature() const;

	/** Set chemical potential.
	 *
	 *  @param chemicalPotential The chemical potential. */
	void setChemicalPotential(double chemicalPotential);

	/** Get chemical potential.
	 *
	 *  @return The chemical potential. */
	double getChemicalPotential() const;

	/** Set statistics.
	 *
	 *  @param statistics The statistics to use. */
	void setStatistics(Statistics statistics);

	/** Get statistics.
	 *
	 *  @return The currently set Statistics. */
	Statistics getStatistics() const;

	/** Get amplitude set.
	 *
	 *  @return Reference to the contained HoppingAmplitudeSet. */
	const HoppingAmplitudeSet& getHoppingAmplitudeSet() const;

	/** Get SourceAmplitudeSet.
	 *
	 *  @return Reference to the contained SourceAmplitudeSet. */
	const SourceAmplitudeSet& getSourceAmplitudeSet() const;

	/** Get geometry.
	 *
	 *  @return Pointer to the contained Geometry. */
	Geometry& getGeometry();

	/** Get geometry.
	 *
	 *  @return Pointer to the contained Geometry. */
	const Geometry& getGeometry() const;

	/** Create ManyParticleContext. */
	void createManyParticleContext();

	/** Get ManyParticleContext.
	 *
	 *  @return Pointer to the contained ManyParticleContext. */
	ManyParticleContext* getManyParticleContext();

	/** Set a HoppingAmplitudeFilter. The HoppingAmplitudeFilter will be
	 *  used by the Model to determine whether a given HoppingAmplitude
	 *  that is passed to the Model actually should be added or not. If no
	 *  HoppingAmplitudeFilter is set, all @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink are added. But if a
	 *  HoppingAmplitudeFilter is set, only those @link HoppingAmplitud
	 *  HoppingAmplitudes @endlink that the filter returns true for are
	 *  added.
	 *
	 *  @param hoppingAmplitudeFilter The HoppingAmplitudeFilter to use. */
	void setFilter(
		const AbstractHoppingAmplitudeFilter &hoppingAmplitudeFilter
	);

	/** Set an IndexFilter. The IndexFilter will be used by the Model to
	 *  determine whether a given HoppingAmplitude that is passed to the
	 *  Model actually should be added or not. If no IndexFilter is set,
	 *  all @link HoppingAmplitude HoppingAmplitudes @endlink are added.
	 *  But if an IndexFilter is set, only those @link HoppingAmplitud
	 *  HoppingAmplitudes @endlink for which the filter returns true for
	 *  both @link Index Indices @endlink are added.
	 *
	 *  @param indexFilter The IndexFilter to use. */
	void setFilter(
		const AbstractIndexFilter &indexFilter
	);

	/** Operator<<. */
	Model& operator<<(const HoppingAmplitude& hoppingAmplitude);

	/** Operator<<. */
	Model& operator<<(
		const std::tuple<HoppingAmplitude, HoppingAmplitude> &hoppingAmplitudes
	);

	/** Operator<<. */
	Model& operator<<(const HoppingAmplitudeList& hoppingAmplitudeList);

	/** Operator<<. */
	Model& operator<<(const SourceAmplitude& sourceAmplitude);

	/** Implements Serializable::serialize(). Note that the
	 *  ManyParticleContext is not yet serialized. */
	std::string serialize(Mode mode) const;
private:
	/** Temperature. */
	double temperature;

	/** Chemical potential. */
	double chemicalPotential;

	/** Single particle context. */
	SingleParticleContext singleParticleContext;

	/** Many-body context. */
	ManyParticleContext *manyParticleContext;

	/** Index filter. */
	AbstractIndexFilter *indexFilter;

	/** Hopping amplitude filter. */
	AbstractHoppingAmplitudeFilter *hoppingAmplitudeFilter;

	/** FileReader is a friend class to allow it to write Model data. */
	friend class FileReader;
};

inline void Model::add(HoppingAmplitude ha){
	singleParticleContext.getHoppingAmplitudeSet().add(ha);
}

inline int Model::getBasisSize() const{
	return singleParticleContext.getHoppingAmplitudeSet().getBasisSize();
}

inline int Model::getBasisIndex(const Index &index) const{
	return singleParticleContext.getHoppingAmplitudeSet().getBasisIndex(index);
}

inline bool Model::getIsConstructed(){
	return singleParticleContext.getHoppingAmplitudeSet().getIsConstructed();
}

inline void Model::sortHoppingAmplitudes(){
	singleParticleContext.sortHoppingAmplitudes();
}

inline void Model::constructCOO(){
	singleParticleContext.constructCOO();
}

inline void Model::destructCOO(){
	singleParticleContext.getHoppingAmplitudeSet().destructCOO();
}

inline void Model::reconstructCOO(){
	singleParticleContext.getHoppingAmplitudeSet().reconstructCOO();
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
	singleParticleContext.setStatistics(statistics);
}

inline Statistics Model::getStatistics() const{
	return singleParticleContext.getStatistics();
}

inline const HoppingAmplitudeSet& Model::getHoppingAmplitudeSet() const{
	return singleParticleContext.getHoppingAmplitudeSet();
}

inline const SourceAmplitudeSet& Model::getSourceAmplitudeSet() const{
	return singleParticleContext.getSourceAmplitudeSet();
}

inline Geometry& Model::getGeometry(){
	return singleParticleContext.getGeometry();
}

inline const Geometry& Model::getGeometry() const{
	return singleParticleContext.getGeometry();
}

inline void Model::createManyParticleContext(){
	manyParticleContext = new ManyParticleContext(&singleParticleContext);
}

inline ManyParticleContext* Model::getManyParticleContext(){
	return manyParticleContext;
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
	if(this->indexFilter != nullptr)
		delete this->indexFilter;
	this->indexFilter = indexFilter.clone();

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
		add(hoppingAmplitude);
	}

	return *this;
}

inline Model& Model::operator<<(const std::tuple<HoppingAmplitude, HoppingAmplitude> &hoppingAmplitudes){
	if(
		hoppingAmplitudeFilter == nullptr
	){
		add(std::get<0>(hoppingAmplitudes));
		add(std::get<1>(hoppingAmplitudes));
	}
	else{
		if(
			hoppingAmplitudeFilter->isIncluded(
				std::get<0>(hoppingAmplitudes)
			)
		){
			add(std::get<0>(hoppingAmplitudes));
		}
		if(
			hoppingAmplitudeFilter->isIncluded(
				std::get<1>(hoppingAmplitudes)
			)
		){
			add(std::get<1>(hoppingAmplitudes));
		}
	}

	return *this;
}

inline Model& Model::operator<<(const HoppingAmplitudeList &hoppingAmplitudeList){
	for(unsigned int n = 0; n < hoppingAmplitudeList.getSize(); n++)
		add(hoppingAmplitudeList[n]);

	return *this;
}

inline Model& Model::operator<<(const SourceAmplitude &sourceAmplitude){
	if(
		indexFilter == nullptr
		|| indexFilter->isIncluded(sourceAmplitude.getIndex())
	){
		singleParticleContext.getSourceAmplitudeSet().add(
			sourceAmplitude
		);
	}

	return *this;
}

};	//End of namespace TBTK

#endif
