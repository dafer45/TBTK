#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ArnoldiIterator");

//! [ArnoldiIterator]
#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/ArnoldiIterator.h"
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
	const unsigned int SIZE_X = 80;
	const unsigned int SIZE_Y = 80;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 15;
	const unsigned int SIZE_Y = 15;
#endif //TBTK_DOCUMENTATION_NICE
	const double t = -1;
	const double mu = -4;

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

	//Set up the Solver. The central value is perturbed slightly from -4
	//to avoid division by zero because the model has an eigenstate exactly
	//at E=-4.
	const unsigned int NUM_EIGEN_VALUES = 100;
	const unsigned int NUM_LANCZOS_VECTORS = 200;
	const unsigned int MAX_ITERATIONS = 400;
	Solver::ArnoldiIterator solver;
	solver.setModel(model);
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(-4 - 1e-6);
	solver.setNumEigenValues(NUM_EIGEN_VALUES);
	solver.setCalculateEigenVectors(true);
	solver.setNumLanczosVectors(NUM_LANCZOS_VECTORS);
	solver.setMaxIterations(MAX_ITERATIONS);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -4.02;
	const double UPPER_BOUND = -3.8;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::ArnoldiIterator propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate eigenvalues.
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	//Plot eigenvalues.
	Plotter plotter;
	plotter.plot(eigenValues);
	plotter.save("figures/EigenValues.png");

	//Calculate the density of states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.001;
	const unsigned int SMOOTHING_WINDOW = 201;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	plotter.clear();
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Calculate the local density of states (LDOS).
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, _a_}
	});
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the LDOS.
	plotter.clear();
	plotter.setNumContours(100);
	plotter.plot({_a_, SIZE_Y/2}, ldos);
	plotter.save("figures/LDOS.png");
}
//! [ArnoldiIterator]
