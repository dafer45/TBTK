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

#include <initializer_list>
#include <complex>

namespace TBTK{
namespace Util{

class ModelFactory{
public:
	/** Create square lattice with two spins per sie and nearest neighbor
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

	/** Add geometry information to square lattice. */
	static void addSquareGeometry(
		Model *model,
		std::initializer_list<int> size
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
