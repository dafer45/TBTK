#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("SquareLattice2D");

//! [SquareLattice2D]
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
	const unsigned int SIZE_X = 30;
	const unsigned int SIZE_Y = 30;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 20;
	const unsigned int SIZE_Y = 20;
#endif //TBTK_DOCUMENTATION_NICE
	const double t = -1;
	const double mu = 0;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			if(x+1 < SIZE_X){
				model << HoppingAmplitude(
					t,
					{x+1, y},
					{x, y}
				) + HC;
			}
			if(y+1 < SIZE_Y){
				model << HoppingAmplitude(
					t,
					{x, y+1},
					{x, y}
				) + HC;
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
	const double LOWER_BOUND = -5;
	const double UPPER_BOUND = 5;
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
	const double SMOOTHING_SIGMA = 0.1;
	const unsigned int SMOOTHING_WINDOW = 101;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	Plotter plotter;
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Calculate the wave functions.
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_}},
			{_a_}
		);

	//Plot the wave function for state 37.
	plotter.clear();
	plotter.plot({_a_, _a_}, 37, waveFunctions);
	plotter.save("figures/WaveFunction.png");
}
//! [SquareLattice2D]
