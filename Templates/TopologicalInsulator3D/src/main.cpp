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

/** @package TBTKtemp
 *  @file main.cpp
 *  @breif Calculating edge spectrum of a TI using Chebyshev expansion
 *
 *  Calculating the band structure at the surface of a 3D topological insulator
 *  with periodic boundary conditions along x- and y-direction, and with edges
 *  in the z-direciton. The calculation is made in real space for a 100x100x100
 *  block by calculating G_{\sigma\sigma}(x', x) for one x and all x' in the
 *  surface layer. The Green's function is then Fourer transformed using fftw3
 *  to give G_{\sigma\sigma}(k).
 *
 *  @author Kristofer Björnson
 */

#include "ChebyshevSolver.h"
#include "FileWriter.h"
#include "GreensFunction.h"
#include "HoppingAmplitudeSet.h"
#include "Model.h"

#include <fftw3.h>

#include <complex>
#include <iostream>
#include <vector>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Parameters
	const int SIZE_X = 25;
	const int SIZE_Y = 25;
	const int SIZE_Z = 25;
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
				model << HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 0},	{x, y, z, 0});
				model << HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 1},	{x, y, z, 1});
				model << HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 2},	{x, y, z, 2});
				model << HoppingAmplitude(mu + 6./SCALE_FACTOR,	{x, y, z, 3},	{x, y, z, 3});

				//Hopping elements along x-direction
				model << HoppingAmplitude(t,		{(x+1)%SIZE_X, y, z, 0},	{x, y, z, 0}) + HC;
				model << HoppingAmplitude(-t,		{(x+1)%SIZE_X, y, z, 1},	{x, y, z, 1}) + HC;
				model << HoppingAmplitude(t,		{(x+1)%SIZE_X, y, z, 2},	{x, y, z, 2}) + HC;
				model << HoppingAmplitude(-t,		{(x+1)%SIZE_X, y, z, 3},	{x, y, z, 3}) + HC;
				model << HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 3},	{x, y, z, 0}) + HC;
				model << HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 0},	{x, y, z, 3}) + HC;
				model << HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 2},	{x, y, z, 1}) + HC;
				model << HoppingAmplitude(i*alpha,	{(x+1)%SIZE_X, y, z, 1},	{x, y, z, 2}) + HC;

				//Hopping elements along y-direction
				model << HoppingAmplitude(t,		{x, y, z, 0},			{x, (y+1)%SIZE_Y, z, 0}) + HC;
				model << HoppingAmplitude(-t,		{x, y, z, 1},			{x, (y+1)%SIZE_Y, z, 1}) + HC;
				model << HoppingAmplitude(t,		{x, y, z, 2},			{x, (y+1)%SIZE_Y, z, 2}) + HC;
				model << HoppingAmplitude(-t,		{x, y, z, 3},			{x, (y+1)%SIZE_Y, z, 3}) + HC;
				model << HoppingAmplitude(alpha,	{x, (y+1)%SIZE_Y, z, 3},	{x, y, z, 0}) + HC;
				model << HoppingAmplitude(-alpha,	{x, (y+1)%SIZE_Y, z, 0},	{x, y, z, 3}) + HC;
				model << HoppingAmplitude(alpha,	{x, (y+1)%SIZE_Y, z, 2},	{x, y, z, 1}) + HC;
				model << HoppingAmplitude(-alpha,	{x, (y+1)%SIZE_Y, z, 1},	{x, y, z, 2}) + HC;

				//Hopping elements along y-direction
				if(z+1 < SIZE_Z){
					model << HoppingAmplitude(t,		{x, y, (z+1)%SIZE_Z, 0},	{x, y, z, 0}) + HC;
					model << HoppingAmplitude(-t,		{x, y, (z+1)%SIZE_Z, 1},	{x, y, z, 1}) + HC;
					model << HoppingAmplitude(t,		{x, y, (z+1)%SIZE_Z, 2},	{x, y, z, 2}) + HC;
					model << HoppingAmplitude(-t,		{x, y, (z+1)%SIZE_Z, 3},	{x, y, z, 3}) + HC;
					model << HoppingAmplitude(i*alpha,	{x, y, (z+1)%SIZE_Z, 1},	{x, y, z, 0}) + HC;
					model << HoppingAmplitude(i*alpha,	{x, y, (z+1)%SIZE_Z, 0},	{x, y, z, 1}) + HC;
					model << HoppingAmplitude(-i*alpha,	{x, y, (z+1)%SIZE_Z, 3},	{x, y, z, 2}) + HC;
					model << HoppingAmplitude(-i*alpha,	{x, y, (z+1)%SIZE_Z, 2},	{x, y, z, 3}) + HC;
				}
			}
		}
	}

	//Construct model
	model.construct();
	model.constructCOO();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Setup ChebyshevSolver
	ChebyshevSolver cSolver;
	cSolver.setModel(model);

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
	Property::GreensFunction *greensFunctionU[SIZE_X*SIZE_Y];
	Property::GreensFunction *greensFunctionD[SIZE_X*SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			greensFunctionU[x + y*SIZE_X] = cSolver.generateGreensFunctionGPU(
				&(cCoefficientsU[(x + y*SIZE_X)*NUM_COEFFICIENTS])
			);
			greensFunctionD[x + y*SIZE_X] = cSolver.generateGreensFunctionGPU(
				&(cCoefficientsD[(x + y*SIZE_X)*NUM_COEFFICIENTS])
			);
		}
	}

	//Evaluate spectral function
	double *spectralFunction[SIZE_X][SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			spectralFunction[x][y] = new double[ENERGY_RESOLUTION];
			const complex<double> *greensFunctionUData = greensFunctionU[x + y*SIZE_X]->getArrayData();
			const complex<double> *greensFunctionDData = greensFunctionD[x + y*SIZE_X]->getArrayData();
			for(int n = 0; n < ENERGY_RESOLUTION; n++)
				spectralFunction[x][y][n] = -imag(greensFunctionUData[n] + greensFunctionDData[n])/M_PI;
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
			delete greensFunctionU[x + y*SIZE_X];
			delete greensFunctionD[x + y*SIZE_X];
			delete [] spectralFunction[x][y];
		}
	}

	return 0;
}
