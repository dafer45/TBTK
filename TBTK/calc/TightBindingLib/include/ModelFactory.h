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
 *  @file ModelFactor.h
 *  @brief Class for generating common models.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MODEL_FACTORY
#define COM_DAFER45_TBTK_MODEL_FACTORY

#include "Model.h"
#include "Geometry.h"
#include "StateSet.h"
#include "UnitCell.h"

#include <initializer_list>
#include <complex>

namespace TBTK{
namespace Util{

class ModelFactory{
public:
	/** Create square lattice with two spins per site and nearest neighbor
	 *  hopping amplitude t.
	 *
	 *  @param size List of ranges. {10}, {10, 20}, {10, 20, 30} creates a
	 *  square lattice of size 10, 10x20, and 10x20x30, respectively.
	 *
	 *  @param periodic Specifies whether given dimension should have
	 *  periodic boundary conditions or not.
	 *
	 *  @param t Nearest neighbor hopping amplitude. */
	static Model* createSquareLattice(
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		std::complex<double> t
	);

	/** Create hexagonal lattice with two spins per site and nearest neighbor
	 *  hopping amplitude t. Unit cell conatins four atom such that the
	 *  final lattice is rectangualrly shaped.
	 *
	 *  @param size List of ranges. {10, 20}, creates a hexagonal lattice
	 *  with rectangular dimension 10x20.
	 *
	 *  @param periodic Specifies whether given dimension should have
	 *  periodic boundary conditions or not.
	 *
	 *  @param t Nearest neighbor hopping amplitude. */
	static Model* createHexagonalLattice(
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		std::complex<double> t
	);

	/** Create a model from a StateSet. */
	static Model* createModel(
		const StateSet &stateSet,
		const AbstractOperator &o = DefaultOperator()
	);

	/** Create a model from a UnitCell.
	 *
	 *  @param unitCell Unit cell to repeat.
	 *  @param size Number of unit cells in the final lattice. For example
	 *	{SIZE_X, SIZE_Y, SIZE_Z} results in a model consisting of
	 *	SIZE_XxSIZE_YxSIZE_Z unit cells.
	 *  @param periodic Specifies whether a given dimension should have
	 *	periodic boundary conditions or not.
	 *  @param includedCells a boolean array indicating which unit cells to
	 *	include in the final lattice. The array should be one
	 *	dimensional with the layout
	 *	includeedCell[x*SIZE_Y*SIZE_Z + y*SIZE_Z + z] (TBTK
	 *	conventional layout with trailing indices most closly packed).
	 *	If NULL is passed, every unit cell is included.
	 * @param o Operator to calculate matrix elements for. */
	static Model* createModel(
		const UnitCell &unitCell,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		const bool *includedCells = NULL,
		const AbstractOperator &o = DefaultOperator()
	);

	/** Add geometry information to square lattice. */
	static void addSquareGeometry(
		Model *model,
		std::initializer_list<int> size
	);

	/** Add geometry information to hexagonal lattice. */
	static void addHexagonalGeometry(
		Model *model,
		std::initializer_list<int> size
	);

	/** Merge models. */
	static Model* merge(
		std::initializer_list<Model*> models
	);
private:
	/** Helper function for createSquareLattice, for 1D. */
	static void createSquareLattice1D(
		Model *model,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		std::complex<double> t
	);

	/** Helper function for createSquareLattice, for 3D. */
	static void createSquareLattice2D(
		Model *model,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		std::complex<double> t
	);

	/** Helper function for createSquareLattice, for 3D. */
	static void createSquareLattice3D(
		Model *model,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		std::complex<double> t
	);

	/** Helper function for createModel, for 1D.*/
	static Model* createModel1D(
		const UnitCell &unitCell,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		const bool *includedCells,
		const AbstractOperator &o
	);

	/** Helper function for createModel, for 2D.*/
	static Model* createModel2D(
		const UnitCell &unitCell,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		const bool *includedCells,
		const AbstractOperator &o
	);

	/** Helper function for createModel, for 3D.*/
	static Model* createModel3D(
		const UnitCell &unitCell,
		std::initializer_list<int> size,
		std::initializer_list<bool> periodic,
		const bool *includedCells,
		const AbstractOperator &o
	);

	/** Helper function for addSquareGeometry, for 1D. */
	static void addSquareGeometry1D(
		Model *model,
		std::initializer_list<int> size
	);

	/** Helper function for addSquareGeometry, for 2D. */
	static void addSquareGeometry2D(
		Model *model,
		std::initializer_list<int> size
	);

	/** Helper function for addSquareGeometry, for 3D. */
	static void addSquareGeometry3D(
		Model *model,
		std::initializer_list<int> size
	);
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif
