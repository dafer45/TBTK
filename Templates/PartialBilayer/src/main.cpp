/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Partial bilayer using diagonalization
 *
 *  Basic example of diagonalization of a 2D tight-binding model with t = 1 and
 *  mu = -1. Bilayer lattice with edges. First layers size is 20x20 sites, while
 *  the second layer is 20x10.
 *
 *  @author Kristofer Björnson
 */

#include <iostream>
#include <complex>
#include "Model.h"
#include "FileWriter.h"
#include "DPropertyExtractor.h"
#include "DiagonalizationSolver.h"

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 20;
	const int SIZE_Y_LAYER_BOTTOM = 20;
	const int SIZE_Y_LAYER_TOP = 10;

	//Parameters
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	//First layer
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y_LAYER_BOTTOM; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude(-mu,	{0, x, y, s},	{0, x, y, s}));

				//Add hopping parameters corresponding to t
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude(-t,	{0, (x+1)%SIZE_X, y, s},	{0, x, y, s}));
				if(y+1 < SIZE_Y_LAYER_BOTTOM)
					model.addHAAndHC(HoppingAmplitude(-t,	{0, x, (y+1)%SIZE_Y_LAYER_BOTTOM, s},	{0, x, y, s}));
			}
		}
	}
	//Second layer
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y_LAYER_TOP; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude(-mu,	{1, x, y, s},	{1, x, y, s}));

				//Add hopping amplitudes between layer 0 and 1
				model.addHAAndHC(HoppingAmplitude(-t,	{1, x, y, s},	{0, x, y, s}));

				//Add hopping amplitudes corresponding to t
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude(-t,	{1, (x+1)%SIZE_X, y, s},	{1, x, y, s}));
				if(y+1 < SIZE_Y_LAYER_TOP)
					model.addHAAndHC(HoppingAmplitude(-t,	{1, x, (y+1)%SIZE_Y_LAYER_TOP, s},	{1, x, y, s}));
			}
		}
	}

	//Construct model
	model.construct();

	//Setup and run DiagonalizationSolver
	DiagonalizationSolver dSolver;
	dSolver.setModel(&model);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor
	DPropertyExtractor pe(&dSolver);

	//Extract eigenvalues and write these to file
	double *ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev, model.getBasisSize());
	delete [] ev;

	//Extract DOS and write to file
	const double UPPER_LIMIT = 7.;
	const double LOWER_LIMIT = -5.;
	const int RESOLUTION = 1000;
	double *dos = pe.calculateDOS(LOWER_LIMIT, UPPER_LIMIT, RESOLUTION);
	FileWriter::writeDOS(dos, LOWER_LIMIT, UPPER_LIMIT, RESOLUTION);
	delete [] dos;

	return 0;
}
