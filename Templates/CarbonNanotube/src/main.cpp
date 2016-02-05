/* Basic example of diagonalization of a 2D tight-binding model with t = 1 and
 * mu = -1. Carbon nanotube with 4*SIZE_X=8 atoms circumference, and SIZE_Y=20
 * atoms long.
 */

#include <iostream>
#include <complex>
#include "Model.h"
#include "FileWriter.h"
#include "PropertyExtractor.h"
#include "DiagonalizationSolver.h"

using namespace std;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Lattice size
	const int SIZE_X = 2;
	const int SIZE_Y = 20;

	//Parameters
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping amplitudes corresponding to chemical potential
				model.addHA(HoppingAmplitude({x, y, 0, s},	{x, y, 0, s},	-mu));
				model.addHA(HoppingAmplitude({x, y, 1, s},	{x, y, 1, s},	-mu));
				model.addHA(HoppingAmplitude({x, y, 2, s},	{x, y, 2, s},	-mu));
				model.addHA(HoppingAmplitude({x, y, 3, s},	{x, y, 3, s},	-mu));

				//Add hopping parameters corresponding to t
				model.addHAAndHC(HoppingAmplitude({x, y, 0, s},	{x, y, 1, s},	-t));
				model.addHAAndHC(HoppingAmplitude({x, y, 1, s},	{x, y, 2, s},	-t));
				model.addHAAndHC(HoppingAmplitude({x, y, 2, s},	{x, y, 3, s},	-t));
				model.addHAAndHC(HoppingAmplitude({x, y, 3, s},	{(x+1)%SIZE_X, y, 0, s},	-t));
				if(y+1 < SIZE_Y){
					model.addHAAndHC(HoppingAmplitude({x, y, 1, s},	{x, (y+1)%SIZE_Y, 0, s},	-t));
					model.addHAAndHC(HoppingAmplitude({x, y, 2, s},	{x, (y+1)%SIZE_Y, 3, s},	-t));
				}
			}
		}
	}

	//Setup and run DiagonalizationSolver
	DiagonalizationSolver dSolver;
	dSolver.setModel(&model);
	dSolver.run();

	//Set filename and remove any file already in the folder
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

	//Create PropertyExtractor
	PropertyExtractor pe(&dSolver);

	//Extract eigenvalues and write these to file
	double *ev = pe.getEV();
	FileWriter::writeEV(ev, model.getBasisSize());
	delete [] ev;

	return 0;
}
