#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("DOS");

//! [DOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	Initialize();

#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 40;
	const unsigned int SIZE_Y = 40;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
#endif //TBTK_DOCUMENTATION_NICE
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -5;
	const double UPPER_BOUND = 5;
	const unsigned int RESOLUTION = 200;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::DOS dos = propertyExtractor.calculateDOS();

	Streams::out << dos << "\n";
	Streams::out << "dos(50) = " << dos(50) << "\n";

	double integratedDOS = 0;
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < dos.getResolution(); n++)
		integratedDOS += dos(n)*dE;
	Streams::out << "Integrated DOS: " << integratedDOS << "\n";

	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 51;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	Plotter plotter;
	plotter.plot(dos);
	plotter.save("figures/DOS.png");
}
//! [DOS]
