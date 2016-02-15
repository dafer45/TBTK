/** @package TBTKcalc
 *  @file PropertyExtracto.h
 *  @brief Extracts physical properties from the output of DiagonalizationSolver
 *
 * @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR

#include "DiagonalizationSolver.h"

class PropertyExtractor{
public:
	PropertyExtractor(DiagonalizationSolver *dSolver);
	~PropertyExtractor();

	void saveEV(std::string path = "./", std::string filename = "EV.dat");

	void getTabulatedAmplitudeSet(int **table, int *dims);

	double* getEV();
	double* calculateDOS(double u_lim, double l_lim, int resolution);
	double* calculateDensity(Index pattern, Index ranges);
	double* calculateMAG(Index pattern, Index ranges);
	//<Not tested>
	double* calculateSP_LDOS(Index pattern, Index ranges, double u_lim, double l_lim, int resolution);
	//</Not tested>

	void save(int *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(double *memory, int size, int columns, std::string filename, std::string path = "./");
	void save(std::complex<double> *memory, int size, int columns, std::string filename, std::string path = "./");
	void save2D(int *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(double *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
	void save2D(std::complex<double> *memory, int size_x, int size_y, int columns, std::string filename, std::string path = "./");
private:
	void calculate(void (*callback)(PropertyExtractor *cb_this, void *memory, const Index &index, int offset),
			void *memory, Index pattern, const Index &ranges, int currentOffset, int offsetMultiplier);

	static void calculateDensityCallback(PropertyExtractor *cb_this, void *density, const Index &index, int offset);
	static void calculateMAGCallback(PropertyExtractor *cb_this, void *mag, const Index &index, int offset);
	//<Not tested>
	static void calculateSP_LDOSCallback(PropertyExtractor *cb_this, void *sp_ldos, const Index &index, int offset);
	//</Not tested>

	DiagonalizationSolver *dSolver;
	void *hint;
};

#endif

