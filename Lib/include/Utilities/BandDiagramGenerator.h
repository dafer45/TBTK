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
 *  @file BandDiagramGenerator.h
 *  @brief Generator of band diagrams.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BAND_DIAGRAM_GENERATOR
#define COM_DAFER45_TBTK_BAND_DIAGRAM_GENERATOR

#include "ReciprocalLattice.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

class BandDiagramGenerator{
public:
	/** Constructor. */
	BandDiagramGenerator();

	/** Set ReciprocalLattice. */
	void setReciprocalLattice(const ReciprocalLattice &reciprocalLattice);

	/** Generate band diagram. */
	std::vector<std::vector<double>> generateBandDiagram(
		std::initializer_list<std::initializer_list<double>> kPoints,
		unsigned int resolution,
		std::initializer_list<std::initializer_list<double>> nestingVectors = {}
	) const;
private:
	/** Pointer to ReciprocalLattice. */
	const ReciprocalLattice *reciprocalLattice;
};

inline void BandDiagramGenerator::setReciprocalLattice(
	const ReciprocalLattice &reciprocalLattice
){
	this->reciprocalLattice = &reciprocalLattice;
}

};	//End of namespace TBTK

#endif
