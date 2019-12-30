#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("CubicLattice3D");

//! [CubicLattice3D]
#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

complex<double> i(0, 1);

int main(){
	//Initialize TBTK.
	Initialize();

	//Parameters.
#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 12;
	const unsigned int SIZE_Y = 12;
	const unsigned int SIZE_Z = 12;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 7;
	const unsigned int SIZE_Y = 7;
	const unsigned int SIZE_Z = 7;
#endif //TBTK_DOCUMENTATION_NICE
	const double t = -1;
	const double mu = -3;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int z = 0; z < SIZE_Z; z++){
				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						t,
						{x+1, y, z},
						{x, y, z}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						t,
						{x, y+1, z},
						{x, y, z}
					) + HC;
				}
				if(z+1 < SIZE_Z){
					model << HoppingAmplitude(
						t,
						{x, y, z+1},
						{x, y, z}
					) + HC;
				}
			}
		}
	}
	model.construct();
	model.setChemicalPotential(mu);

	//Set up the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the density of states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 101;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	Plotter plotter;
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Calculate the wave functions.
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_, _a_}},
			{_a_}
		);

	//Plot the wave function for state 37.
	plotter.clear();
	plotter.plot({_a_, _a_, SIZE_Z/2}, 10, waveFunctions);
	plotter.save("figures/WaveFunction.png");
}
//! [CubicLattice3D]
