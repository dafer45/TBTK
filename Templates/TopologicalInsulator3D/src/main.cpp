/* Calculating the band structure at the surface of a 3D topological insulator
 * with periodic boundary conditions along x- and y-direction, and with edges
 * in the z-direciton. The calculation is made in real space for a 100x100x100
 * block by calculating G_{\sigma\sigma}(x', x) for one x and all x' in the
 * surface layer. The Green's function is then Fourer transformed using fftw3
 * to give G_{\sigma\sigma}(k).
 */

#include <iostream>
#include <complex>
#include "AmplitudeSet.h"
#include "Model.h"
#include "FileWriter.h"
#include "ChebyshevSolver.h"
#include <vector>
#include <fftw3.h>

using namespace std;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Parameters
	const int SIZE_X = 100;
	const int SIZE_Y = 100;
	const int SIZE_Z = 100;
	const double SCALE_FACTOR = 15.;

	complex<double> mu = -1.0/SCALE_FACTOR;
	complex<double> t = 1.0/SCALE_FACTOR;
	complex<double> alpha = 1.0/SCALE_FACTOR;

	//Number of coefficeints and energy resolution used in the expansion
	//and evaluation of the Green's function.
	const int NUM_COEFFICIENTS = 4000;
	const int ENERGY_RESOLUTION = 5000;

	//Create model and setup hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int z = 0; z < SIZE_Z; z++){
				//Diagonal terms
				model.addHA(HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 0},	{x, y, z, 0}));
				model.addHA(HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 1},	{x, y, z, 1}));
				model.addHA(HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 2},	{x, y, z, 2}));
				model.addHA(HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 3},	{x, y, z, 3}));

				//Hopping elements along x-direction
				model.addHAAndHC(HoppingAmplitude(t,	{(x+1)%SIZE_X, y, z, 0},	{x, y, z, 0}));
				model.addHAAndHC(HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, z, 1},	{x, y, z, 1}));
				model.addHAAndHC(HoppingAmplitude(t,	{(x+1)%SIZE_X, y, z, 2},	{x, y, z, 2}));
				model.addHAAndHC(HoppingAmplitude(-t,	{(x+1)%SIZE_X, y, z, 3},	{x, y, z, 3}));
				model.addHAAndHC(HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 3},	{x, y, z, 0}));
				model.addHAAndHC(HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 0},	{x, y, z, 3}));
				model.addHAAndHC(HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 2},	{x, y, z, 1}));
				model.addHAAndHC(HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 1},	{x, y, z, 2}));

				//Hopping elements along y-direction
				model.addHAAndHC(HoppingAmplitude(t,	{x, y, z, 0},	{x, (y+1)%SIZE_Y, z, 0}));
				model.addHAAndHC(HoppingAmplitude(-t,	{x, y, z, 1},	{x, (y+1)%SIZE_Y, z, 1}));
				model.addHAAndHC(HoppingAmplitude(t,	{x, y, z, 2},	{x, (y+1)%SIZE_Y, z, 2}));
				model.addHAAndHC(HoppingAmplitude(-t,	{x, y, z, 3},	{x, (y+1)%SIZE_Y, z, 3}));
				model.addHAAndHC(HoppingAmplitude(alpha,	{x, (y+1)%SIZE_Y, z, 3},	{x, y, z, 0}));
				model.addHAAndHC(HoppingAmplitude(-alpha,	{x, (y+1)%SIZE_Y, z, 0},	{x, y, z, 3}));
				model.addHAAndHC(HoppingAmplitude(alpha,	{x, (y+1)%SIZE_Y, z, 2},	{x, y, z, 1}));
				model.addHAAndHC(HoppingAmplitude(-alpha,	{x, (y+1)%SIZE_Y, z, 1},	{x, y, z, 2}));

				//Hopping elements along y-direction
				if(z+1 < SIZE_Z){
					model.addHAAndHC(HoppingAmplitude(t,	{x, y, (z+1)%SIZE_Z, 0},	{x, y, z, 0}));
					model.addHAAndHC(HoppingAmplitude(-t,	{x, y, (z+1)%SIZE_Z, 1},	{x, y, z, 1}));
					model.addHAAndHC(HoppingAmplitude(t,	{x, y, (z+1)%SIZE_Z, 2},	{x, y, z, 2}));
					model.addHAAndHC(HoppingAmplitude(-t,	{x, y, (z+1)%SIZE_Z, 3},	{x, y, z, 3}));
					model.addHAAndHC(HoppingAmplitude(i*alpha,	{x, y, (z+1)%SIZE_Z, 1},	{x, y, z, 0}));
					model.addHAAndHC(HoppingAmplitude(i*alpha,	{x, y, (z+1)%SIZE_Z, 0},	{x, y, z, 1}));
					model.addHAAndHC(HoppingAmplitude(-i*alpha,	{x, y, (z+1)%SIZE_Z, 3},	{x, y, z, 2}));
					model.addHAAndHC(HoppingAmplitude(-i*alpha,	{x, y, (z+1)%SIZE_Z, 2},	{x, y, z, 3}));
				}
			}
		}
	}

	//Construct model
	model.construct();

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

	//Create storage for Chebyshev coefficients. SIZE_X*SIZE_Y Green's
	//functions G_{\sigma\sigma}(x', x) is to be calculated, where x' runs
	//over each site on the surface. Each Green's function is exapnded using
	//NUM_COEFFICEINTS coefficients.
	complex<double> *cCoefficientsU = new complex<double>[NUM_COEFFICIENTS*SIZE_X*SIZE_Y];
	complex<double> *cCoefficientsD = new complex<double>[NUM_COEFFICIENTS*SIZE_X*SIZE_Y];

	//Create list of all x' indices (to-indices according to the index name
	//convention <c_{to}^{\dagger}c_{from}>)
	vector<Index> toIndicesU;
	vector<Index> toIndicesD;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			toIndicesU.push_back({x, y, 0, 0});
			toIndicesD.push_back({x, y, 0, 1});
		}
	}

	//Calculate Chebyshev coefficients for G_{\uparrow\uparrow}(x', x) and
	//G_{\downarrow\downarrow}(x', x) for x = (0,0,0), and all x' in the
	//surface layer. Remove GPU from function name to run on cpu instead.
	cSolver.calculateCoefficientsGPU(toIndicesU,
						{0, 0, 0, 0},
						cCoefficientsU,
						NUM_COEFFICIENTS);
	cSolver.calculateCoefficientsGPU(toIndicesD,
						{0, 0, 0, 1},
						cCoefficientsD,
						NUM_COEFFICIENTS);

	//Setup and run Fourier transform using fftw3
	fftw_complex *in[2][NUM_COEFFICIENTS], *out[2][NUM_COEFFICIENTS];
	fftw_plan plan[2][NUM_COEFFICIENTS];
	for(int n = 0; n < NUM_COEFFICIENTS; n++){
		//Allocate input, output, and plans
		in[0][n] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SIZE_X*SIZE_Y);
		in[1][n] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SIZE_X*SIZE_Y);
		out[0][n] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SIZE_X*SIZE_Y);
		out[1][n] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SIZE_X*SIZE_Y);
		plan[0][n] = fftw_plan_dft_2d(SIZE_X, SIZE_Y, in[0][n], out[0][n], -1, FFTW_ESTIMATE);
		plan[1][n] = fftw_plan_dft_2d(SIZE_X, SIZE_Y, in[1][n], out[1][n], -1, FFTW_ESTIMATE);

		//Setup input
		for(int x = 0; x < SIZE_X; x++){
			for(int y = 0; y < SIZE_Y; y++){
				in[0][n][x + y*SIZE_X][0] = real(cCoefficientsU[(x + y*SIZE_X)*NUM_COEFFICIENTS + n]);
				in[0][n][x + y*SIZE_Y][1] = imag(cCoefficientsU[(x + y*SIZE_X)*NUM_COEFFICIENTS + n]);
				in[1][n][x + y*SIZE_Y][0] = real(cCoefficientsD[(x + y*SIZE_X)*NUM_COEFFICIENTS + n]);
				in[1][n][x + y*SIZE_Y][1] = imag(cCoefficientsD[(x + y*SIZE_X)*NUM_COEFFICIENTS + n]);
			}
		}
		//Execute Fourier transforms
		fftw_execute(plan[0][n]);
		fftw_execute(plan[1][n]);

		//Overwrite storage used for real space coefficeints to store
		//k-space coefficients
		for(int x = 0; x < SIZE_X; x++){
			for(int y = 0; y < SIZE_Y; y++){
				cCoefficientsU[(x + y*SIZE_X)*NUM_COEFFICIENTS + n] = out[0][n][x + y*SIZE_X][0] + i*out[0][n][x + y*SIZE_X][1];
				cCoefficientsD[(x + y*SIZE_X)*NUM_COEFFICIENTS + n] = out[1][n][x + y*SIZE_X][0] + i*out[1][n][x + y*SIZE_X][1];
			}
		}

		//Free memory
		fftw_destroy_plan(plan[0][n]);
		fftw_destroy_plan(plan[1][n]);
		fftw_free(in[0][n]);
		fftw_free(in[1][n]);
		fftw_free(out[0][n]);
		fftw_free(out[1][n]);
	}

	//Generate Green's functions. Remove GPU from function name to run on
	//cpu instead.
	complex<double> *greensFunctionU[SIZE_X*SIZE_Y];
	complex<double> *greensFunctionD[SIZE_X*SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			greensFunctionU[x + y*SIZE_X] = new complex<double>[ENERGY_RESOLUTION];
			greensFunctionD[x + y*SIZE_X] = new complex<double>[ENERGY_RESOLUTION];
			cSolver.generateGreensFunctionGPU(greensFunctionU[x + y*SIZE_X], &(cCoefficientsU[(x + y*SIZE_X)*NUM_COEFFICIENTS]));
			cSolver.generateGreensFunctionGPU(greensFunctionD[x + y*SIZE_X], &(cCoefficientsD[(x + y*SIZE_X)*NUM_COEFFICIENTS]));
		}
	}

	//Evaluate spectral function
	double *spectralFunction[SIZE_X][SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			spectralFunction[x][y] = new double[ENERGY_RESOLUTION];
			for(int n = 0; n < ENERGY_RESOLUTION; n++)
				spectralFunction[x][y][n] = -imag(greensFunctionU[x + y*SIZE_X][n] + greensFunctionD[x + y*SIZE_X][n])/M_PI;
		}
	}

	//Save spectral function at (k_x, k_y) to Spectral_function_x_y
	int dims[1];
	dims[0] = ENERGY_RESOLUTION;
	stringstream ss;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			ss.str("");
			ss << "Spectral_function_" << x << "_" << y;
			FileWriter::write(spectralFunction[x][y], 1, dims, ss.str().c_str());
		}
	}

	//Free lookup table from GPU. Remove this if evaluation on cpu is preffered.
	cSolver.destroyLookupTableGPU();

	//Free memory
	delete [] cCoefficientsU;
	delete [] cCoefficientsD;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			delete [] greensFunctionU[x + y*SIZE_X];
			delete [] greensFunctionD[x + y*SIZE_X];
			delete [] spectralFunction[x][y];
		}
	}

	return 0;
}
