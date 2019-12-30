#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("SuperconductingVortex");

//! [SuperconductingVortex]
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
	const unsigned int SIZE_X = 31;
	const unsigned int SIZE_Y = 31;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 21;
	const unsigned int SIZE_Y = 21;
#endif //TBTK_DOCUMENTATION_NICE
	const double t = -1;
	const double mu = -2;
	const double Delta = 0.5;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int ph = 0; ph < 2; ph++){
				model << HoppingAmplitude(
					-mu*(1. - 2*ph),
					{x, y, ph},
					{x, y, ph}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						t*(1. - 2*ph),
						{x+1, y, ph},
						{x, y, ph}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						t*(1. - 2*ph),
						{x, y+1, ph},
						{x, y, ph}
					) + HC;
				}
			}

			double X = x - SIZE_X/2.;
			double Y = y - SIZE_Y/2.;
			double R = sqrt(X*X + Y*Y);
			model << HoppingAmplitude(
				Delta*exp(i*atan2(Y, X))*tanh(R),
				{x, y, 1},
				{x, y, 0}
			) + HC;
		}
	}
	model.construct();

	//Set up the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -1.5;
	const double UPPER_BOUND = 1.5;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the density of states (DOS).
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, SIZE_Y/2, IDX_SUM_ALL}
	});

	//Smooth the LDOS.
	const double SMOOTHING_SIGMA = 0.01;
	const unsigned int SMOOTHING_WINDOW = 201;
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	Plotter plotter;
	plotter.setNumContours(100);
	plotter.plot(
		{_a_, SIZE_Y/2, IDX_SUM_ALL},
		ldos
	);
	plotter.save("figures/LDOS.png");
}
//! [SuperconductingVortex]
