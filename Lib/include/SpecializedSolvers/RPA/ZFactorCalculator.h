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
 *  @file ZFactorCalculator.h
 *  @brief Calculates the Z-factor using the RPA approximation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_Z_FACTOR_CALCULATOR
#define COM_DAFER45_TBTK_Z_FACTOR_CALCULATOR

#include "BrillouinZone.h"
#include "BlockDiagonalizationSolver.h"
#include "BPropertyExtractor.h"
#include "IndexedDataTree.h"
#include "SelfEnergyCalculator.h"

namespace TBTK{

class ZFactorCalculator{
public:
	/** Constructor. */
	ZFactorCalculator(const MomentumSpaceContext &momentumSpaceContext);

	/** Destructor. */
	~ZFactorCalculator();

	/** Set number of energies. */
	void setNumSummationEnergies(unsigned int numSummationEnergies);

	/** Initialize the ZFactorCalculator. */
	void init();

	/** Calculate Z-factor. */
	std::vector<std::complex<double>> calculateZFactor(
		const std::vector<double> &k
	);

	/** Calculate Z-factor. */
	std::vector<std::complex<double>> calculateZFactor2(
		const std::vector<double> &k
	);

	/** Set U. */
	void setU(std::complex<double> U);

	/** Set Up. */
	void setUp(std::complex<double> Up);

	/** Set J. */
	void setJ(std::complex<double> J);

	/** Set Jp. */
	void setJp(std::complex<double> Jp);

	/** Precompute susceptibilities. */
//	void precomputeSusceptibilities(unsigned int numWorkers = 128);

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** SusceptibilityCalculator. */
	SelfEnergyCalculator selfEnergyCalculator;

	/** Number of summationEnergies. */
	unsigned int numSummationEnergies;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** Invert matix. */
	void invertMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** Multiply matrices. */
	void multiplyMatrices(
		std::complex<double> *matrix1,
		std::complex<double> *matrix2,
		std::complex<double> *result,
		unsigned int dimensions
	);

	/** Print matrix. */
	void printMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;
};

inline void ZFactorCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies
){
	this->numSummationEnergies = numSummationEnergies;
}

inline void ZFactorCalculator::setU(std::complex<double> U){
	this->U = U;
	selfEnergyCalculator.setU(U);
}

inline void ZFactorCalculator::setUp(std::complex<double> Up){
	this->Up = Up;
	selfEnergyCalculator.setUp(Up);
}

inline void ZFactorCalculator::setJ(std::complex<double> J){
	this->J = J;
	selfEnergyCalculator.setJ(J);
}

inline void ZFactorCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	selfEnergyCalculator.setJp(Jp);
}

/*inline void ZFactorCalculator::precomputeSusceptibilities(
	unsigned int numWorkers
){
	selfEnergyCalculator.precomputeSusceptibilities(numWorkers);
}*/

inline void ZFactorCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	selfEnergyCalculator.saveSusceptibilities(filename);
}

inline void ZFactorCalculator::loadSusceptibilities(
	const std::string &filename
){
	selfEnergyCalculator.loadSusceptibilities(filename);
}

};	//End of namespace TBTK

#endif
