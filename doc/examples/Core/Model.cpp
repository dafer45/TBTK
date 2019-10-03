#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Model");

//! [Model]
#include "TBTK/Model.h"
#include "TBTK/Streams.h"

#include <complex>

using namespace std;
using namespace TBTK;

//Create a 2D-lattice with on-site potential U, nearest neighbor hopping
//amplitude t, chemical potential mu, temperature T, and Fermi-Dirac
//statistics.
int main(){
	//Parameters.
	complex<double> U = 1;
	complex<double> t = 1;
	double mu = -1;
	double T = 300;
	unsigned int SIZE_X = 10;
	unsigned int SIZE_Y = 10;

	//Model specification.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int spin = 0; spin < 2; spin++){
				model << HoppingAmplitude(
					U,
					{x, y, spin},
					{x, y, spin}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{x+1, y, spin},
						{x, y, spin}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{x, y+1, spin},
						{x, y, spin}
					) + HC;
				}
			}
		}
	}
	model.setChemicalPotential(mu);
	model.setTemperature(T);
	model.setStatistics(Statistics::FermiDirac);

	//Construct a mapping from the Physical Indices to a linear Hilbert
	//space basis.
	model.construct();

	Streams::out << model << "\n";
}
//! [Model]
