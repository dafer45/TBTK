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
 *  @file DPropertyExtractor.h
 *  @brief Extracts physical properties from the DiagonalizationSolver
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_D_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_D_PROPERTY_EXTRACTOR

#include "PropertyExtractor.h"
#include "DiagonalizationSolver.h"
#include "EigenValues.h"
#include "DOS.h"
#include "Density.h"
#include "Magnetization.h"
#include "LDOS.h"
#include "SpinPolarizedLDOS.h"

#include <complex>

namespace TBTK{

/** The DPropertyExtractor extracts common physical properties such as DOS,
 *  Density, LDOS, etc. from a DiagonalizationSolver. These can then be written
 *  to file using the FileWriter.*/
class DPropertyExtractor : public PropertyExtractor{
public:
	/** Constructor. */
	DPropertyExtractor(DiagonalizationSolver *dSolver);

	/** Destructor. */
	~DPropertyExtractor();

	/** Legacy. */
	void saveEigenValues(
		std::string path = "./",
		std::string filename = "EV.dat"
	);

	/** Experimental. Extracts a tabulated version of the AmplitudeSet. */
	void getTabulatedAmplitudeSet(
		std::complex<double> **amplitudes,
		int **indices,
		int *numHoppingAmplitudes,
		int *maxIndexSize
	);

	/** Get eigenvalues. */
//	double* getEigenValues();
	Property::EigenValues* getEigenValues();

	/** Get eigenvalue. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$.
	 *  @param state Eigenstate number \f$n\f$
	 *  @param index Physical index \f$x\f$.
	 */
	const std::complex<double> getAmplitude(int state, const Index &index);

	/** Calculate density of states.
	 *  @param l_lim Lower limit for energy interval.
	 *  @param u_lim Upper limit for energy interval.
	 *  @param resolution Number of points used between l_lim and u_lim.
	 *  @return An array with size resolution. */
//	double* calculateDOS(double l_lim, double u_lim, int resolution);
	Property::DOS* calculateDOS(
		double l_lim,
		double u_lim,
		int resolution
	);

	/** Calculate expectation value. */
	std::complex<double> calculateExpectationValue(Index to, Index from);

	/** Calculate density.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the density. For example, assume that the index scheme is
	 *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SUM_ALL} will calculate the
	 *  density for each x along (y,z)=(5,10) by summing over spin.
	 *  Similarly {ID_X, 5, IDX_Y, IDX_SUM_ALL} will return a two
	 *  dimensional density for all x and z and y = 5. Note that IDX_X
	 *  IDX_Y, and IDX_Z refers to the first, second, and third index used
	 *  by the routine to create a one-, two-, or three-dimensional output,
	 *  rather than being tied to the x, y, and z used as physical
	 *  subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *   ignored for indices specified with positive integer in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A density array with size equal to the number of points
	 *  included by specified patter-range combination.
	 */
//	double* calculateDensity(Index pattern, Index ranges);
	Property::Density* calculateDensity(Index pattern, Index ranges);

	/** Calculate magnetization.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the magnetization. For example, assume that the index scheme is
	 *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SPIN} will calculate the
	 *  magnetization for each x along (y,z)=(5,10). Similarly
	 *  {ID_X, 5, IDX_Y, IDX_SPIN} will return a two dimensional
	 *  magnetiation for all x and z and y = 5. Note that IDX_X, IDX_Y, and
	 *  IDX_Z refers to the first, second, and third index used by the
	 *  routine to create a one-, two-, or three-dimensional output, rather
	 *  than being tied to the x, y, and z used as physical subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *   ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A magnetization array with size equal to four times the
	 *  number of points included by specified patter-range combination.
	 *  The four entries are
	 *  \f[
	 *      \left[\begin{array}{cc}
	 *          0	& 1\\
	 *          2	& 3
	 *      \end{array}\right] =
	 *      \left[\begin{array}{cc}
	 *          \langle c_{i\uparrow}^{\dagger}c_{i\uparrow}\rangle		& \langle c_{i\uparrow}^{\dagger}c_{i\downarrow}\rangle\\
	 *          \langle c_{i\downarrow}^{\dagger}c_{u\uparrow}\rangle	& \langle c_{i\downarrow}^{\dagger}c_{i\downarrow}\rangle
	 *      \end{array}\right].
	 *  \f]
	 */
//	double* calculateMAG(Index pattern, Index ranges);
//	std::complex<double>* calculateMAG(Index pattern, Index ranges);
	Property::Magnetization* calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Calculate local density of states.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
         *  the LDOS. For example, assume that the index scheme is
         *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SUM_ALL} will calculate the
         *  LDOS for each x along (y,z)=(5,10) by summing over spin. Similarly
         *  {ID_X, 5, IDX_Y, IDX_SUM_ALL} will return a two dimensional LDOS
         *  for all x and z and y = 5. Note that IDX_X, IDX_Y, and IDX_Z refers
         *  to the first, second, and third index used by the routine to create
         *  a one-, two-, or three-dimensional output, rather than being tied
         *  to the x, y, and z used as physical subindices.
         *
         *  @param ranges Speifies the number of elements for each subindex. Is
         *  ignored for indices specified with positive integers in the
         *  pattern, but is used to loop from 0 to the value in ranges for
         *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
         *  corresponding to the two pattern examples above are
         *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
         *  respectively.
         *
         *  @return A density array with size equal to the number of points
         *  included by specified patter-range combination.
	 */
	Property::LDOS* calculateLDOS(
		Index pattern,
		Index ranges,
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Calculate spin-polarized local density of states.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the spin-polarized LDOS. For example, assume that the index scheme
	 *  is {x, y, z, spin}. {ID_X, 5, 10, IDX_SPIN} will calculate the
	 *  spin-polarized LDOS for each x along (y,z)=(5,10). Similarly
	 *  {ID_X, 5, IDX_Y, IDX_SPIN} will return a two dimensional
	 *  spin-polarized LDOS for all x and z and y = 5. Note that IDX_X,
	 *  IDX_Y, and IDX_Z refers to the first, second, and third index used
	 *  by the routine to create a one-, two-, or three-dimensional output,
	 *  rather than being tied to the x, y, and z used as physical
	 *  subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *   ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A spin-polarized LDOS array with size equal to four times
	 *  the number of points included by specified patter-range
	 *  combination.
	 *  The four entries are
	 *  \f[
	 *      \left[\begin{array}{cc}
	 *          0	& 1\\
	 *          2	& 3
	 *      \end{array}\right] =
	 *      \left[\begin{array}{cc}
	 *          \rho_{i\uparrow i\uparrow}(E)	& \rho_{i\uparrow i\downarrow}(E)\\
	 *          \rho_{i\downarrow i\uparrow}(E)	& \rho_{i\downarrow i\downarrow}(E)\\
	 *      \end{array}\right],
	 *  \f]
	 *  where
	 *  \f[
	 *      \rho_{i\sigma i\sigma'}(E) = \sum_{E_n}\langle\Psi_n|c_{i\sigma}^{\dagger}c_{i\sigma'}|\Psi_n\rangle\delta(E - E_n) .
	 *  \f]
	 */
//	double* calculateSP_LDOS(Index pattern, Index ranges, double u_lim, double l_lim, int resolution);
//	std::complex<double>* calculateSP_LDOS(Index pattern, Index ranges, double l_lim, double u_lim, int resolution);
	Property::SpinPolarizedLDOS* calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges,
		double l_lim,
		double u_lim,
		int resolution
	);

/*	void save(int *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(double *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(std::complex<double> *memory, int size, int columns, std::string filename, std::string path = "./");
	void save2D(int *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(double *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(std::complex<double> *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");*/
private:
	/** Loops over range indices and calls the appropriate callback
	 *  function to calculate the correct quantity. */
/*	void calculate(
		void (*callback)(
			DPropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		void *memory,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier
	);*/

	/** Callback for calculating density. Used by calculateDensity. */
	static void calculateDensityCallback(
		PropertyExtractor *cb_this,
		void *density,
		const Index &index,
		int offset
	);

	/** Callback for calculating magnetization. Used by calculateMAG. */
	static void calculateMAGCallback(
		PropertyExtractor *cb_this,
		void *mag,
		const Index &index,
		int offset
	);

	/** Calback for callculating local density of states. Used by
	 *  calculateLDOS. */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		void *ldos,
		const Index &index,
		int offset
	);

	/** Callback for calculating spin-polarized local density of states.
	 *  Used by calculateSP_LDOS. */
	static void calculateSP_LDOSCallback(
		PropertyExtractor *cb_this,
		void *sp_ldos,
		const Index &index,
		int offset
	);

	/** DiagonalizationSolver to work on. */
	DiagonalizationSolver *dSolver;

	/** Hint used to pass information between calculate[Property] and
	 *  calculate[Property]Callback. */
//	void *hint;

	/** Ensure that range indices are on compliant format. (Set range to
	 *  one for indices with non-negative pattern value.) */
//	void ensureCompliantRanges(const Index &pattern, Index &ranges);

	/** Extract ranges for loop indices. */
/*	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *lDimensions,
		int **lRanges
	);*/
};

inline double DPropertyExtractor::getEigenValue(int state){
	return dSolver->getEigenValue(state);
}

inline const std::complex<double> DPropertyExtractor::getAmplitude(
	int state,
	const Index &index
){
	return dSolver->getAmplitude(state, index);
}

};	//End of namespace TBTK

#endif
