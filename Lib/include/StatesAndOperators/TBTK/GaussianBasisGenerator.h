/* Copyright 2019 Kristofer Björnson
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
 *  @file GaussianBasisGenerator.h
 *  @brief Generates a basis of @link GaussianState GaussianStates @endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GAUSSIAN_BASIS_GENERATOR
#define COM_DAFER45_TBTK_GAUSSIAN_BASIS_GENERATOR

#include "TBTK/GaussianState.h"

#include <unordered_map>
#include <vector>

#include <libint2.hpp>

namespace TBTK{

class GaussianBasisGenerator{
public:
	/** Constructor. */
	GaussianBasisGenerator();

	/** Deleted copy constructor. */
	GaussianBasisGenerator(
		const GaussianBasisGenerator &gaussianBasisGenerator
	) = delete;

	/** Destructor. */
	~GaussianBasisGenerator();

	/** Deleted assignment operator. */
	GaussianBasisGenerator& operator=(
		const GaussianBasisGenerator &rhs
	) = delete;

	/** Generate a basis of @link GaussianState GaussianStates@endling. */
	std::vector<GaussianState> generateBasis();
private:
	std::vector<GaussianState> basisSet;

	libint2::BasisSet libintBasisSet;

	std::unordered_map<size_t, std::vector<size_t>> libintShellPairList;

	std::vector<
		std::vector<std::shared_ptr<libint2::ShellPair>>
	> libintShellPairData;

	std::vector<libint2::Atom> atoms;

	static unsigned int instanceCounter;

	/** Initialize the basis state generation algorithm. */
	void initialize();

	/** Initialize the basis set. */
	void initializeBasisSet();

	/** Calculate shell pairs. */
	void computeShellPairs(double threshold = 1e-12);

	/** Calculate single particle integral. */
	void calculateSingleParticleTerms(
		libint2::Operator o,
		std::function<
			void(
				std::complex<double> value,
				unsigned int linearBraIndex,
				unsigned int linearKetIndex
			)
		> &&lambdaStoreResult
	);

	/** Calculate the two-body Fock integrals. */
	void calculateTwoBodyFockTerms();
};

};	//End of namespace TBTK

#endif
