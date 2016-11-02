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
 *  @file Lattice.h
 *  @brief A Lattice allows for repeated replication of UnitCells.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LATTICE
#define COM_DAFER45_TBTK_LATTICE

#include "UnitCell.h"
#include "Index.h"

#include <vector>

namespace TBTK{

class Lattice{
public:
	/** Constructor. */
	Lattice(UnitCell *unitCell);

	/** Destructor. */
	~Lattice();

	/** Add lattice point to the lattice. */
	void addLatticePoint(const Index &latticePoint);

	/** Genearates a state set from the Lattice. */
	StateSet* generateStateSet();
private:
	/** Unit cell that is to be replicated throughout the lattice. */
	UnitCell *unitCell;

	/** Lattice points that are included in the lattice. */
	std::vector<Index> latticePoints;
};

};	//End of namespace TBTK

#endif
