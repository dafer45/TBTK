#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Superconductivity");

//! [Superconductivity]
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
	const double mu = -2;

	Plotter plotter;
	for(unsigned int n = 0; n < 2; n++){
		double Delta;
		if(n == 0)
			Delta = 0.5;
		else
			Delta = 0;

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

				model << HoppingAmplitude(
					Delta,
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
		Property::DOS dos = propertyExtractor.calculateDOS();

		//Smooth the DOS.
		const double SMOOTHING_SIGMA = 0.1;
		const unsigned int SMOOTHING_WINDOW = 201;
		dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

		//Plot the DOS.
		if(n == 0){
			plotter.plot(
				dos,
				{{"linestyle", "-"}, {"color", "black"}}
			);
		}
		else{
			plotter.plot(
				dos,
				{{"linestyle", "--"}, {"color", "black"}}
			);
		}
	}
	plotter.save("figures/DOS.png");
}
//! [Superconductivity]
