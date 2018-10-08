/* Copyright 2017 Kristofer Björnson
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
 *  @file MomentumSpaceContext.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RPA_MOMENTUM_SPACE_CONTEXT
#define COM_DAFER45_TBTK_RPA_MOMENTUM_SPACE_CONTEXT

#include "TBTK/BrillouinZone.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"

namespace TBTK{

class MomentumSpaceContext{
public:
	/** Constructor. */
	MomentumSpaceContext();

	/** Destructor. */
	~MomentumSpaceContext();

	/** Set model. */
	void setModel(Model &model);

	/** Get model. */
	const Model& getModel() const;

	/** Set Brillouin zone. */
	void setBrillouinZone(const BrillouinZone &brillouinZone);

	/** Get Brillouin zone. */
	const BrillouinZone& getBrillouinZone() const;

	/** Set number of mesh points. */
	void setNumMeshPoints(const std::vector<unsigned int> &numMeshPoints);

	/** Get number of mesh points. */
	const std::vector<unsigned int>& getNumMeshPoints() const;

	/** Get mesh. */
	const std::vector<std::vector<double>>& getMesh() const;

	/** Set number of orbitals. */
	void setNumOrbitals(unsigned int numOrbitals);

	/** Get number of orbitals. */
	unsigned int getNumOrbitals() const;

	/** Initialize the SusceptibilityCalculator. */
	void init();

	/** Get energy using global state index. */
	double getEnergy(unsigned int state) const;

	/** Get energy using block local state index. */
	double getEnergy(unsigned int block, unsigned int state) const;

	/** Get amplitude. */
	std::complex<double> getAmplitude(
		unsigned int block,
		unsigned int state,
		unsigned int amplitude
	) const;

	/** Get Index corresponding to given k-vector. */
	Index getKIndex(const std::vector<double> &k) const;

	/** Get property extractor. */
	const PropertyExtractor::BlockDiagonalizer& getPropertyExtractorBlockDiagonalizer() const;
private:
	/** Model to work on. */
	Model *model;

	/** BrillouinZone. */
	const BrillouinZone *brillouinZone;

	/** Number of mesh points. */
	std::vector<unsigned int> numMeshPoints;

	/** Mesh. */
	std::vector<std::vector<double>> mesh;

	/** Number of orbitals. */
	unsigned int numOrbitals;

	/** Solver. */
	Solver::BlockDiagonalizer solver;

	/** Property extractor. */
	PropertyExtractor::BlockDiagonalizer *propertyExtractor;

	/** Energies. */
	double *energies;

	/** Amplitudes. */
	std::complex<double> *amplitudes;

	/** Flag indicating whether the SusceptibilityCalculator is
	 *  initialized. */
	bool isInitialized;

};

inline void MomentumSpaceContext::setModel(Model &model){
	this->model = &model;

	isInitialized = false;
}

inline const Model& MomentumSpaceContext::getModel() const{
	return *model;
}

inline void MomentumSpaceContext::setBrillouinZone(
	const BrillouinZone &brillouinZone
){
	this->brillouinZone = &brillouinZone;

	isInitialized = false;
}

inline const BrillouinZone& MomentumSpaceContext::getBrillouinZone() const{
	return *brillouinZone;
}

inline void MomentumSpaceContext::setNumMeshPoints(
	const std::vector<unsigned int> &numMeshPoints
){
	this->numMeshPoints = numMeshPoints;
	TBTKAssert(
		brillouinZone != nullptr,
		"MomentumSpaceContext::setNumMeshPoints()",
		"BrillouinZone not set.",
		"First set the BrillouinZone using"
		<< " MomentumSpaceContext::setBrillouinZone()"
	);
	mesh = brillouinZone->getMinorMesh(numMeshPoints);

	isInitialized = false;
}

inline const std::vector<unsigned int>& MomentumSpaceContext::getNumMeshPoints(
) const{
	return numMeshPoints;
}

inline const std::vector<std::vector<double>>& MomentumSpaceContext::getMesh(
) const{
	return mesh;
}

inline void MomentumSpaceContext::setNumOrbitals(unsigned int numOrbitals){
	this->numOrbitals = numOrbitals;

	isInitialized = false;
}

inline unsigned int MomentumSpaceContext::getNumOrbitals() const{
	return numOrbitals;
}

inline double MomentumSpaceContext::getEnergy(unsigned int state) const{
	return energies[state];
}

inline double MomentumSpaceContext::getEnergy(
	unsigned int block,
	unsigned int state
) const{
	return energies[block*numOrbitals + state];
}

inline std::complex<double> MomentumSpaceContext::getAmplitude(
	unsigned int block,
	unsigned int state,
	unsigned int amplitude
) const{
	return amplitudes[numOrbitals*(numOrbitals*block + state) + amplitude];
}

inline Index MomentumSpaceContext::getKIndex(
	const std::vector<double> &k
) const{
	return brillouinZone->getMinorCellIndex(
		k,
		numMeshPoints
	);
}

inline const PropertyExtractor::BlockDiagonalizer& MomentumSpaceContext::getPropertyExtractorBlockDiagonalizer(
) const{
	return *propertyExtractor;
}

};	//End of namespace TBTK

#endif
