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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file UnitCell.h
 *  @brief Unit cell that act as container of States.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_CELL
#define COM_DAFER45_TBTK_UNIT_CELL

#include "TBTK/StateSet.h"

namespace TBTK{

class UnitCell : public StateSet{
public:
	/** Constructor. */
	UnitCell(std::initializer_list<std::initializer_list<double>> latticeVectors, bool isOwner = true);

	/** Constructor. */
	UnitCell(const std::vector<std::vector<double>> &latticeVectors, bool isOwner = true);

	/** Destructor. */
	~UnitCell();

	/** Get number of lattice vectors. */
	int getNumLatticeVectors() const;

	/** Get lattice vector. */
	const std::vector<double>& getLatticeVector(int n) const;

	/** Get lattice vectors. */
	const std::vector<std::vector<double>>& getLatticeVectors() const;
private:
	/** Lattice vectors. */
	std::vector<std::vector<double>> latticeVectors;
};

inline int UnitCell::getNumLatticeVectors() const{
	return latticeVectors.size();
}

inline const std::vector<double>& UnitCell::getLatticeVector(int n) const{
	return latticeVectors.at(n);
}

inline const std::vector<std::vector<double>>& UnitCell::getLatticeVectors() const{
	return latticeVectors;
}

};	//End of namespace TBTK

#endif
/// @endcond
