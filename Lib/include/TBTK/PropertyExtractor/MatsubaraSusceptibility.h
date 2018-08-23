/* Copyright 2018 Kristofer Björnson
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
 *  @file MatsubaraSusceptibility.h
 *  @brief Extracts physical properties from the
 *  Solver::MatsubaraSusceptibility.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_MATSUBARA_SUSCEPTIBILITY
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_MATSUBARA_SUSCEPTIBILITY

#include "TBTK/Solver/MatsubaraSusceptibility.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::MatsubaraSusceptibility extracts the Susceptibility
 *  from Solver::MatsubaraSusceptibility. */
class MatsubaraSusceptibility : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::MatsubaraSusceptibility.
	 *
	 *  @param solver The Solver to use. */
	MatsubaraSusceptibility(Solver::MatsubaraSusceptibility &solver);

	/** Calculates the Susceptibility. */
	virtual Property::Susceptibility calculateSusceptibility(
		std::vector<Index> patterns
	);
private:
	/** Information class for passing information about the block structure
	 *  when calculating the susceptibility. */
	class MatsubaraInformation : public Information{
	public:
		/** Constructs a
		 *  PropertyExtractor::MatsubaraSusceptibility::MatsubaraInfomration.
		 */
		MatsubaraInformation();

		/** Set whether the susceptibility should be calculated for all
		 *  block indices. */
		void setCalculateSusceptibilityForAllBlocks(
			bool calculateSusceptibilityForAllBlocks
		);

		/** Get whether the susceptibility should be calculated for all
		 *  block indices. */
		bool getCalculateSusceptibilityForAllBlocks() const;
	private:
		/** Flag indicating whether the susceptibility should be
		 *  calculated for all block indices. */
		bool calculateSusceptibilityForAllBlocks;
	};

	/** Calback for callculating susceptibility. */
	static void calculateSusceptibilityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Solver::Diagonalizer to work on. */
	Solver::MatsubaraSusceptibility *solver;

	/** Energies. */
	std::vector<std::complex<double>> energies;
};

inline void MatsubaraSusceptibility::MatsubaraInformation::setCalculateSusceptibilityForAllBlocks(
	bool calculateSusceptibilityForAllBlocks
){
	this->calculateSusceptibilityForAllBlocks
		= calculateSusceptibilityForAllBlocks;
}

inline bool MatsubaraSusceptibility::MatsubaraInformation::getCalculateSusceptibilityForAllBlocks(
) const{
	return calculateSusceptibilityForAllBlocks;
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
