/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Basic Chebyshev example
 *
 *  Basic example of using the Chebyshev method to solve a 2D tight-binding
 *  model with t = 1 and mu = -1. Lattice with edges and a size of 40x40 sites.
 *  Using 5000 Chebyshev coefficients and evaluating the Green's function with
 *  an energy resolution of 10000. Calculates LDOS at SIZE_X = 40 sites along
 *  the line y = SIZE_Y/2 = 20.
 *
 *  @author Kristofer Björnson
 */

#include <complex>
#include "Model.h"
#include "ChebyshevSolver.h"
#include "CPropertyExtractor.h"
#include "FileWriter.h"

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 40;
	const int SIZE_Y = 40;

	//Model parameters.
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude(-mu,	{x, y, s},	{x, y, s}));

				//Add hopping parameters corresponding to t
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, s},	{x, y, s}));
				if(y+1 < SIZE_Y)
					model.addHAAndHC(HoppingAmplitude(-t,	{x, (y+1)%SIZE_Y, s},	{x, y, s}));
			}
		}
	}

	//Construct model
	model.construct();

	//Chebyshev expansion parameters.
	const int NUM_COEFFICIENTS = 5000;
	const int ENERGY_RESOLUTION = 10000;
	const double SCALE_FACTOR = 5.;

	//Setup ChebyshevSolver
	ChebyshevSolver cSolver;
	cSolver.setModel(&model);
	cSolver.setScaleFactor(SCALE_FACTOR);

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor. The parameter are in order: The
	//ChebyshevSolver, number of expansion coefficients used in the
	//Cebyshev expansion, energy resolution with which the Green's function
	// is evaluated, whether calculate expansion functions using a GPU or
	//not, whether to evaluate the Green's function using a GPU or not,
	//whether to use a lookup table for the Green's function or not
	//(required if the Green's function is evaluated on a GPU), and the
	//lower and upper bound between which the Green's function is evaluated
	//(has to be inside the interval [-SCALE_FACTOR, SCALE_FACTOR]).
	CPropertyExtractor pe(&cSolver,
				NUM_COEFFICIENTS,
				ENERGY_RESOLUTION,
				false,
				false,
				true,
				-SCALE_FACTOR,
				SCALE_FACTOR);

	//Extract local density of states and write to file
	double *ldos = pe.calculateLDOS({IDX_X, SIZE_Y/2, IDX_SUM_ALL}, {SIZE_Y, 1, 2});
	const int RANK = 1;
	int dims[RANK] = {SIZE_X};
	FileWriter::writeLDOS(ldos,
				RANK,
				dims,
				-SCALE_FACTOR,
				SCALE_FACTOR,
				ENERGY_RESOLUTION);
	delete [] ldos;

	return 0;
}
