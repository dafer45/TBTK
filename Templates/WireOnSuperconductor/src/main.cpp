/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Wire on superconductor using self-consistent diagonalization
 *  
 *  Self-consistent solution for superconducting order-parameter for a 2D
 *  tight-binding model with t_ss = 1, mu = -1, and V_sc = 2. Lattice with
 *  edges and a size of 36x19 sites. A ferromagnetic wire of length 18 is
 *  attached on top of the superconductor with t_mm = 0.75 inside the wire, and
 *  t_sm = 0.9 between wire and superconductor.
 *
 *  @author Kristofer Björnson
 */

#include <iostream>
#include <complex>
#include "Model.h"
#include "FileWriter.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

//Lattice size
const int SIZE_X = 4*7;
const int SIZE_Y = 2*7+1;

//Order parameter. The two buffers are alternatively swaped by setting dCounter
// = 0 or 1. One buffer contains the order parameter used in the previous
//calculation, while the other is used to store the newly obtained order
//parameter that will be used in the next calculation.
complex<double> D[2][SIZE_X][SIZE_Y];
int dCounter = 0;

//Superconducting pair potential, convergence limit, max iterations, and initial guess
const double V_sc = 2.;
const double CONVERGENCE_LIMIT = 0.0001;
const int MAX_ITERATIONS = 50;
const complex<double> D_INITIAL_GUESS = 0.3;

//Self-consistent callback that is to be called each time a diagonalization has
//finished. Calculates the order parameter from the current solution.
bool scCallback(DiagonalizationSolver *dSolver){
	//Clear the order parameter
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			D[(dCounter+1)%2][x][y] = 0.;
		}
	}

	//Calculate D(x, y) = <c_{x, y, \downarrow}c_{x, y, \uparrow}> = \sum_{E_n<E_F} conj(v_d^{(n)})*u_u^{(n)}
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int n = 0; n < dSolver->getModel()->getBasisSize()/2; n++){
				//Obtain amplitudes at site (x,y) for electron_up and hole_down components
				complex<double> u_u = dSolver->getAmplitude(n, {0, x, y, 0});
				complex<double> v_d = dSolver->getAmplitude(n, {0, x, y, 3});

				D[(dCounter+1)%2][x][y] -= V_sc*conj(v_d)*u_u;
			}
		}
	}

	//Swap order parameter buffers
	dCounter = (dCounter+1)%2;

	//Calculate convergence parameter
	double maxError = 0.;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			double error = 0.;
			if(abs(D[dCounter][x][y]) == 0 && abs(D[(dCounter+1)%2][x][y]) == 0)
				error = 0.;
			else
				error = abs(D[dCounter][x][y] - D[(dCounter+1)%2][x][y])/max(abs(D[dCounter][x][y]), abs(D[(dCounter+1)%2][x][y]));

			if(error > maxError)
				maxError = error;
		}
	}

	return true;

	//Return true or false depending on whether the result has converged or not
	if(maxError < CONVERGENCE_LIMIT)
		return true;
	else
		return false;
}

//Callback function responsible for determining the value of the order
//parameter D_{to,from}c_{to}c_{from} where to and from are indices of the form
//(x, y, spin).
complex<double> fD(Index from, Index to){
	//Obtain indices
	int x = from.indices.at(1);
	int y = from.indices.at(2);
	int s = from.indices.at(3);

	//Return appropriate amplitude
	switch(s){
		case 0:
			return conj(D[dCounter][x][y]);
		case 1:
			return -conj(D[dCounter][x][y]);
		case 2:
			return -D[dCounter][x][y];
		case 3:
			return D[dCounter][x][y];
		default://Never happens
			return 0;
	}
}

//Function responsible for initializing the order parameter
void initD(){
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			D[0][x][y] = D_INITIAL_GUESS;
		}
	}
}

int main(int argc, char **argv){
	//Parameters
	complex<double> mu = -1.0;
	complex<double> t_ss = 1.0;
	complex<double> t_mm = 0.75;
	complex<double> t_sm = 0.9;	
	complex<double> V_z = 0.5;

	//Create model
	Model model;
	//Setup hopping parameters for superconducting layer
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude(-mu,	{0, x, y, s},	{0, x, y, s}));
				model.addHA(HoppingAmplitude(mu,	{0, x, y, s+2},	{0, x, y, s+2}));

				//Add hopping parameters corresponding to t_ss
				if(x+1 < SIZE_X){
					model.addHAAndHC(HoppingAmplitude(-t_ss,	{0, (x+1)%SIZE_X, y, s},		{0, x, y, s}));
					model.addHAAndHC(HoppingAmplitude(t_ss,		{0, (x+1)%SIZE_X, y, s+2},	{0, x, y, s+2}));
				}
				if(y+1 < SIZE_Y){
					model.addHAAndHC(HoppingAmplitude(-t_ss,	{0, x, (y+1)%SIZE_Y, s},		{0, x, y, s}));
					model.addHAAndHC(HoppingAmplitude(t_ss,		{0, x, (y+1)%SIZE_Y, s+2},	{0, x, y, s+2}));
				}
				model.addHAAndHC(HoppingAmplitude(fD,	{0, x, y, 3-s},	{0, x, y, s}));
			}
		}
	}
	//Setup wire and attach it to superconductor
	for(int x = 0; x < SIZE_X/2; x++){
		for(int s = 0; s < 2; s++){
			//Add hopping amplitudes corresponding to Zeeman term
			model.addHA(HoppingAmplitude(-2.*V_z*(s-1/2.),	{1, x, s},		{1, x, s}));
			model.addHA(HoppingAmplitude(2.*V_z*(s-1/2.),	{1, x, s+2},	{1, x, s+2}));

			//Add hopping amplitudes corresponding to t_mm
			if(x+1 < SIZE_X/2){
				model.addHAAndHC(HoppingAmplitude(-t_mm,	{1, x+1, s},	{1, x, s}));
				model.addHAAndHC(HoppingAmplitude(t_mm,		{1, x+1, s},	{1, x, s+2}));
			}

			//Add hopping amplitudes corresponding to t_ms
			model.addHAAndHC(HoppingAmplitude(-t_sm,	{0, x+SIZE_X/4, SIZE_Y/2, s},	{1, x, s}));
			model.addHAAndHC(HoppingAmplitude(t_sm,		{0, x+SIZE_X/4, SIZE_Y/2, s+2},	{1, x, s+2}));
		}
	}

	//Construct model
	model.construct();

	//Initialize D
	initD();

	//Setup and run DiagonalizationSolver
	DiagonalizationSolver dSolver;
	dSolver.setModel(&model);
	dSolver.setMaxIterations(MAX_ITERATIONS);
	dSolver.setSCCallback(scCallback);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Calculate abs(D) and arg(D)
	double D_abs[SIZE_X*SIZE_Y];
	double D_arg[SIZE_X*SIZE_Y];
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			D_abs[x*SIZE_Y + y] = abs(D[dCounter][x][y]);
			D_arg[x*SIZE_Y + y] = arg(D[dCounter][x][y]);
		}
	}

	//Save abs(D) and arg(D) in "D_abs" and "D_arg"
	const int D_RANK = 2;
	int dDims[D_RANK] = {SIZE_X, SIZE_Y};
	FileWriter::write(D_abs, D_RANK, dDims, "D_abs");
	FileWriter::write(D_arg, D_RANK, dDims, "D_arg");

	//Calculate and save magnetization
	DPropertyExtractor pe(&dSolver);
	complex<double> *mag = pe.calculateMAG({0, IDX_X, IDX_Y, IDX_SPIN},
						{1, SIZE_X, SIZE_Y, 2});
	const int MAG_RANK = 2;
	int mag_dims[MAG_RANK];
	mag_dims[0] = SIZE_X;
	mag_dims[1] = SIZE_Y;
	FileWriter::writeMAG(mag, MAG_RANK, mag_dims);
	delete [] mag;

	//Calculate and save spin-polarized local density of states
	const int SP_LDOS_UPPER_LIMIT = 2.;
	const int SP_LDOS_LOWER_LIMIT = -2.;
	const int SP_LDOS_RESOLUTION = 1000;
	complex<double> *sp_ldos = pe.calculateSP_LDOS({0, IDX_X, SIZE_Y/2, IDX_SPIN},
							{1, SIZE_X, 1, 2},
							SP_LDOS_UPPER_LIMIT,
							SP_LDOS_LOWER_LIMIT,
							SP_LDOS_RESOLUTION);
	const int SP_LDOS_RANK = 1;
	int sp_ldos_dims[SP_LDOS_RANK];
	sp_ldos_dims[0] = SIZE_X;
	FileWriter::writeSP_LDOS(sp_ldos,
					SP_LDOS_RANK,
					sp_ldos_dims,
					SP_LDOS_UPPER_LIMIT,
					SP_LDOS_LOWER_LIMIT,
					SP_LDOS_RESOLUTION);
	delete [] sp_ldos;

	return 0;
}
