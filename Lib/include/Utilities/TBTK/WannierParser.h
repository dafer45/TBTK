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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file WannierParser.h
 *  @brief Parses Wannier files.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WANNIER_PARSER
#define COM_DAFER45_TBTK_WANNIER_PARSER

#include "TBTK/Model.h"
#include "TBTK/ParallelepipedArrayState.h"
#include "TBTK/ReciprocalLattice.h"
#include "TBTK/Resource.h"

#include <initializer_list>
#include <string>
#include <vector>

namespace TBTK{

class WannierParser{
public:
	/** Parse file. */
//	UnitCell* parseMatrixElements(std::string filename);
	UnitCell* parseMatrixElements(Resource &resource);

	/** Parse Wannier functions. */
	std::vector<ParallelepipedArrayState*> parseWannierFunctions(
//		std::string filename,
		Resource &resource,
		unsigned int resolutionX,
		unsigned int resolutionY,
		unsigned int resolutionZ,
		unsigned int numStates,
		const std::vector<std::vector<double>> &basisVectors
	);

	/** Parse WannierFunctions. */
	std::vector<ParallelepipedArrayState*> parseWannierFunctions(
		Resource &resource
	);

	/** Get reciprocal lattice. */
	ReciprocalLattice* getReciprocalLattice();

	/** Get matrix dimension. */
//	unsigned int getMatrixDimension() const;
private:
	/** Reciprocal lattice. */
//	ReciprocalLattice *reciprocalLattice;

	/** Matrix dimension. */
//	unsigned int matrixDimension;
};

/*inline ReciprocalLattice* WannierParser::getReciprocalLattice(){
	return reciprocalLattice;
}*/

/*inline unsigned int WannierParser::getMatrixDimension() const{
	return matrixDimension;
}*/

};	//End of namespace TBTK

#endif
/// @endcond
