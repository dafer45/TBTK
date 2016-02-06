/* Basic example of using the Chebyshev method to solve a 2D tight-binding
 * model with t = 1 and mu = -1. Lattice with edges and a size of 20x20 sites.
 * Using 5000 Chebyshev coefficients and evaluating the Green's function with
 * an energy resolution of 10000. Calculates LDOS at SIZE_X = 20 sites along
 * the line y = SIZE_Y/2 = 10.
 */

#include <iostream>
#include <complex>
#include "Model.h"
#include "FileWriter.h"
#include "PropertyExtractor.h"
#include "ChebyshevSolver.h"

using namespace std;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 20;
	const int SIZE_Y = 20;
	const int NUM_COEFFICIENTS = 5000;
	const int ENERGY_RESOLUTION = 10000;

	//Parameters. The SCALE_FACOTR is required for restricting the energy
	//spectrum to the interval [-1,1].
	const double SCALE_FACTOR = 5.;
	complex<double> mu = -1.0/SCALE_FACTOR;
	complex<double> t = 1.0/SCALE_FACTOR;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude({x, y, s},		{x, y, s},	-mu));

				//Add hopping parameters corresponding to t
				if(x+1 < SIZE_X)
					model.addHAAndHC(HoppingAmplitude({x, y, s},	{(x+1)%SIZE_X, y, s},	-t));
				if(y+1 < SIZE_Y)
					model.addHAAndHC(HoppingAmplitude({x, y, s},	{x, (y+1)%SIZE_Y, s},	-t));
			}
		}
	}
	//Construct model
	model.amplitudeSet.construct();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Setup ChebyshevSolver
	ChebyshevSolver cSolver;
	cSolver.setModel(&model);

	//Generate lookup table for quicker evaluation of Green's functions.
	cSolver.generateLookupTable(NUM_COEFFICIENTS, ENERGY_RESOLUTION);
	//Load lookup table to GPU. Remove this if evaluation on cpu is preffered.
	cSolver.loadLookupTableGPU();

	//Calculate and save LDOS on site (x, SIZE_Y/2) for x \in [0, SIZE_Y-1]
	for(int x = 0; x < SIZE_X; x++){
		//Calculate chebyshev coefficients for
		//G_{\uparrow\uparrow}(x,SIZE_Y/2) and
		//G_{\downarrow\downarrow}(x, SIZE_Y/2). Remove GPU from
		//function name to run on cpu instead.
		complex<double> *cCoefficientsU = new complex<double>[NUM_COEFFICIENTS];
		complex<double> *cCoefficientsD = new complex<double>[NUM_COEFFICIENTS];
		cSolver.calculateCoefficientsGPU({x, SIZE_Y/2, 0},
							{x, SIZE_Y/2, 0},
							cCoefficientsU,
							NUM_COEFFICIENTS);
		cSolver.calculateCoefficientsGPU({x, SIZE_Y/2, 1},
							{x, SIZE_Y/2, 1},
							cCoefficientsD,
							NUM_COEFFICIENTS);

		//Generate Green's function. Remove GPU from function name to
		//run on cpu instead.
		complex<double> *greensFunctionU = new complex<double>[ENERGY_RESOLUTION];
		complex<double> *greensFunctionD = new complex<double>[ENERGY_RESOLUTION];
		cSolver.generateGreensFunctionGPU(greensFunctionU, cCoefficientsU);
		cSolver.generateGreensFunctionGPU(greensFunctionD, cCoefficientsD);

		//Calculate LDOS
		double *ldos = new double[ENERGY_RESOLUTION];
		for(int n = 0; n < ENERGY_RESOLUTION; n++)
			ldos[n] = -imag(greensFunctionU[n] + greensFunctionD[n])/M_PI;

		//Save LDOS at x to LDOS_x
		const int LDOS_RANK = 1;
		int ldosDims[LDOS_RANK];
		ldosDims[0] = ENERGY_RESOLUTION;
		stringstream ss;
		ss.str("");
		ss << "LDOS_" << x;
		FileWriter::write(ldos, LDOS_RANK, ldosDims, ss.str().c_str());

		//Free allocated memory. Note that allocation and deallocation
		//of these arrays can be moved outside of the loop for
		//optimization. They are continusly reallocated inside the loop
		//in this example to make the example clearer by keeping
		//variable definitions close to their usage point.
		delete [] cCoefficientsU;
		delete [] cCoefficientsD;
		delete [] greensFunctionU;
		delete [] greensFunctionD;
		delete [] ldos;
	}

	//Free lookup table from GPU. Remove this if evaluation on cpu is
	//preffered.
	cSolver.destroyLookupTableGPU();

	return 0;
}
