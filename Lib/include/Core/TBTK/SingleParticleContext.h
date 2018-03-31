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
 *  @file SingleParticleContext.h
 *  @brief The context for the single particle part of a Model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SINGLE_PARTICLE_CONTEXT
#define COM_DAFER45_TBTK_SINGLE_PARTICLE_CONTEXT

#include "TBTK/Geometry.h"
#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/Serializable.h"
#include "TBTK/Statistics.h"

namespace TBTK{

class FileReader;

/** @brief The context for the single particle part of a Model. */
class SingleParticleContext : public Serializable{
public:
	/** Constructor. */
	SingleParticleContext();

	/** Constructor. */
	SingleParticleContext(const std::vector<unsigned int> &capacity);

	/** Copy constructor. */
	SingleParticleContext(
		const SingleParticleContext &singleParticleContext
	);

	/** Move constructor. */
	SingleParticleContext(
		SingleParticleContext &&singleParticleContext
	);

	/** Constructor. Constructs the SingleParticleContext from a
	 *  serializeation string. */
	SingleParticleContext(const std::string &serialization, Mode mode);

	/**Destructor. */
	virtual ~SingleParticleContext();

	/** Assignment operator. */
	SingleParticleContext& operator=(const SingleParticleContext &rhs);

	/** Move assignment operator. */
	SingleParticleContext& operator=(SingleParticleContext &&rhs);

	/** Set statistics.
	 *
	 *  @param statistics The Statistics to use.*/
	void setStatistics(Statistics statistics);

	/** Get statistics.
	 *
	 *  @return The currently set Statistics. */
	Statistics getStatistics() const;

	/** Add a HoppingAmplitude. */
	void addHoppingAmplitude(HoppingAmplitude ha);

	/** Add a HoppingAmplitude and its Hermitian conjugate. */
	void addHoppingAmplitudeAndHermitianConjugate(HoppingAmplitude ha);

	/** Get Hilbert space index corresponding to given 'from'-index.
	 *  @param index 'from'-index to get Hilbert space index for. */
	int getBasisIndex(const Index &index) const;

	/** Get size of Hilbert space. */
	int getBasisSize() const;

	/** Construct Hilbert space. No more @link HoppingAmplitude
	 *  HoppingAmplitudes @endlink should be added after this call. */
	void construct();

	/*** Sort HoppingAmplitudes. */
	void sortHoppingAmplitudes();

	/** Returns true if the Hilbert space basis has been constructed. */
	bool getIsConstructed() const;

	/** Construct Hamiltonian on COO format. */
	void constructCOO();

	/** Destruct Hamiltonian on COO format. */
	void destructCOO();

	/** To be called when HoppingAmplitudes need to be reevaluated. This is
	 *  required if the HoppingAmplitudeSet in addition to its standard
	 *  storage format also utilizes a more effective format such as COO
	 *  format and some HoppingAMplitudes are evaluated through the use of
	 *  callbacks. */
	void reconstructCOO();

	/** Get HoppingAMplitudeSet. */
	const HoppingAmplitudeSet* getHoppingAmplitudeSet() const;

	/** Create Geometry. */
	void createGeometry(int dimensions, int numSpecifiers = 0);

	/** Get Geometry. */
	Geometry* getGeometry();

	/** Implements Serializable::serialize(). */
	std::string serialize(Mode mode) const;
private:
	/** Statistics (Fermi-Dirac or Bose-Einstein).*/
	Statistics statistics;

	/** HoppingAmplitudeSet containing @ling HoppingAmplitude
	 *  HoppingAmplitudes @endlink. */
	HoppingAmplitudeSet *hoppingAmplitudeSet;

	/** Geometry. */
	Geometry *geometry;

	/** FileReader is a friend class to allow it to write Model data. */
	friend class FileReader;
};

inline void SingleParticleContext::setStatistics(Statistics statistics){
	this->statistics = statistics;
}

inline Statistics SingleParticleContext::getStatistics() const{
	return statistics;
}

inline void SingleParticleContext::addHoppingAmplitude(HoppingAmplitude ha){
	hoppingAmplitudeSet->addHoppingAmplitude(ha);
}

inline void SingleParticleContext::addHoppingAmplitudeAndHermitianConjugate(
	HoppingAmplitude ha
){
	hoppingAmplitudeSet->addHoppingAmplitudeAndHermitianConjugate(ha);
}

inline int SingleParticleContext::getBasisIndex(const Index &index) const{
	return hoppingAmplitudeSet->getBasisIndex(index);
}

inline int SingleParticleContext::getBasisSize() const{
	return hoppingAmplitudeSet->getBasisSize();
}

inline bool SingleParticleContext::getIsConstructed() const{
	return hoppingAmplitudeSet->getIsConstructed();
}

inline void SingleParticleContext::sortHoppingAmplitudes(){
	hoppingAmplitudeSet->sort();
}

inline void SingleParticleContext::constructCOO(){
	hoppingAmplitudeSet->sort();
	hoppingAmplitudeSet->constructCOO();
}

inline void SingleParticleContext::destructCOO(){
	hoppingAmplitudeSet->destructCOO();
}

inline void SingleParticleContext::reconstructCOO(){
	hoppingAmplitudeSet->reconstructCOO();
}

inline const HoppingAmplitudeSet* SingleParticleContext::getHoppingAmplitudeSet() const{
	return hoppingAmplitudeSet;
}

inline Geometry* SingleParticleContext::getGeometry(){
	return geometry;
}

};	//End of namespace TBTK

#endif
