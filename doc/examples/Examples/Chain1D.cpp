#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Chain1D");

//! [Chain1D]
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
	const unsigned int SIZE = 500;
	const double t = -1;
	const double mu = -1;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE; x++)
		model << HoppingAmplitude(t, {x+1}, {x}) + HC;
	model.construct();
	model.setChemicalPotential(mu);

	//Set up the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -3;
	const double UPPER_BOUND = 3;
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
	const double SMOOTHING_SIGMA = 0.03;
	const unsigned int SMOOTHING_WINDOW = 101;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	Plotter plotter;
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Calculate the wave functions.
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_}},
			{_a_}
		);

	//Plot wave function for state 0, 1, and 2.
	plotter.clear();
	plotter.setTitle("Wave function for state 0, 1, and 2.");
	for(unsigned int state = 0; state < 3; state++)
		plotter.plot({_a_}, state, waveFunctions);
	plotter.save("figures/WaveFunctions.png");
}
//! [Chain1D]
