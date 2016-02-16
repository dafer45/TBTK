/** @package TBTKcalc
 *  @file PropertyExtracto.h
 *  @brief Extracts physical properties from the DiagonalizationSolver
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR

#include "DiagonalizationSolver.h"

/** The PropertyExtractor extracts common physical properties such as DOS,
 *  Density, LDOS, etc. from a DiagonalizationSolver. These can then be written
 *   to file using the FileWriter.*/
class PropertyExtractor{
public:
	/** Constructor. */
	PropertyExtractor(DiagonalizationSolver *dSolver);

	/** Destructor. */
	~PropertyExtractor();

	/** Legacy. */
	void saveEV(std::string path = "./", std::string filename = "EV.dat");

	/** Experimental. Extracts a tabulated version of the AmplitudeSet. */
	void getTabulatedAmplitudeSet(int **table, int *dims);

	/** Get eigenvalues. */
	double* getEV();

	/** Calculate DOS.
	 *  @param u_lim Upper limit for energy interval.
	 *  @param l_lim Lower limit for energy interval.
	 *  @param resolution Number of points used between l_lim and u_lim.
	 *  @return An array with size resolution. */
	double* calculateDOS(double u_lim, double l_lim, int resolution);

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
	 *  @parameter ranges Speifies the number of elements for each
	 *  subindex. Is ignored for indices specified with positive integers
	 *  in the pattern, but is used to loop from 0 to the value in ranges
	 *  for IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A density array with size equal to the number of points
	 *  included by specified patter-range combination.
	 */
	double* calculateDensity(Index pattern, Index ranges);

	/** Likely to change. Calculate magnetization.
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
	 *  @parameter ranges Speifies the number of elements for each
	 *  subindex. Is ignored for indices specified with positive integers
	 *  in the pattern, but is used to loop from 0 to the value in ranges
	 *  for IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A magnetization array with size equal to six time the
	 *  number of points included by specified patter-range combination.
	 *  The six entries per point corresponds to x-up, x-down, y-up,
	 *  y-down, z-up, and z-down, respectively.
	 */
	double* calculateMAG(Index pattern, Index ranges);

	/** Likely to change. Calculate spin-polarized local density of states.
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
	 *  @parameter ranges Speifies the number of elements for each
	 *  subindex. Is ignored for indices specified with positive integers
	 *  in the pattern, but is used to loop from 0 to the value in ranges
	 *  for IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A spin-polarized LDOS array with size equal to six time the
	 *  number of points included by specified patter-range combination.
	 *  The six entries per point corresponds to x-up, x-down, y-up,
	 *  y-down, z-up, and z-down, respectively.
	 */
	double* calculateSP_LDOS(Index pattern, Index ranges, double u_lim, double l_lim, int resolution);

	void save(int *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(double *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(std::complex<double> *memory, int size, int columns, std::string filename, std::string path = "./");
	void save2D(int *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(double *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(std::complex<double> *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
private:
	/** Loops over range indices and calls the appropriate callback
	 *  function to calculate the correct quantity. */
	void calculate(void (*callback)(PropertyExtractor *cb_this, void *memory, const Index &index, int offset),
			void *memory, Index pattern, const Index &ranges, int currentOffset, int offsetMultiplier);

	/** Callback for calculating density. Used by calculateDensity. */
	static void calculateDensityCallback(PropertyExtractor *cb_this, void *density, const Index &index, int offset);
	/** Callback for calculating magnetization. Used by calculateMAG. */
	static void calculateMAGCallback(PropertyExtractor *cb_this, void *mag, const Index &index, int offset);
	/** Callback for calculating spin-polarized local density of states.
	 *  Used by calculateSP_LDOS. */
	static void calculateSP_LDOSCallback(PropertyExtractor *cb_this, void *sp_ldos, const Index &index, int offset);

	/** DiagonalizationSolver to work on. */	
	DiagonalizationSolver *dSolver;

	/** Hint used to pass information between calculate[Property] and
	 *  calculate[Property]Callback. */
	void *hint;
};

#endif

