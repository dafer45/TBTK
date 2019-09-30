#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Diagonalizer");

//! [Diagonalizer]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	const int SIZE_X = 10;
	const int SIZE_Y = 10;
	double t = 1;

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

	Streams::out << "----------------\n";

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const int RESOLUTION = 10;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Streams::out << "----- DOS ------\n";

	Property::DOS dos = propertyExtractor.calculateDOS();
	Streams::out << dos.getLowerBound() << "\n";
	Streams::out << dos.getUpperBound() << "\n";
	for(int n = 0; n < dos.getResolution(); n++)
		Streams::out << dos(n) << "\t";
	Streams::out << "\n";

	Streams::out << "--- Density ----\n";

	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, 0},	//All points of the form {x, 0}
		{5, _a_}	//All points of the form {5, y}
	});
	for(int x = 0; x < SIZE_X; x++)
		Streams::out << density({x, 0}) << "\t";
	Streams::out << "\n";
	for(int y = 0; y < SIZE_Y; y++)
		Streams::out << density({5, y}) << "\t";
	Streams::out << "\n";
}
//! [Diagonalizer]
