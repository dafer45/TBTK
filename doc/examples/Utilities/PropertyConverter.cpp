#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("PropertyConverter");

//! [PropertyConverter]
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/PropertyConverter.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

#include <vector>

using namespace std;
using namespace TBTK;

int main(){
	Initialize();

	//Create Model.
	const unsigned int SIZE_X = 2;
	const unsigned int SIZE_Y = 3;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	//Set up and run the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	PropertyExtractor::Diagonalizer propertyExtractor;
	propertyExtractor.setSolver(solver);
	propertyExtractor.setEnergyWindow(-10, 10, 5);

	//Calculate Properties.
	Property::DOS dos = propertyExtractor.calculateDOS();
	Property::Density density
		= propertyExtractor.calculateDensity({{_a_, _a_}});
	Property::LDOS ldos = propertyExtractor.calculateLDOS({{_a_, _a_}});

	//Convert the Properties to AnnotatedArrays.
	AnnotatedArray<double, Subindex> dosArray
		= PropertyConverter::convert(dos);
	AnnotatedArray<double, Subindex> densityArray1D
		= PropertyConverter::convert(density, {_a_, 1});
	AnnotatedArray<double, Subindex> densityArray2D
		= PropertyConverter::convert(density, {_a_, _a_});
	AnnotatedArray<double, Subindex> ldosArray
		= PropertyConverter::convert(ldos, {_a_, 1});

	//Print results.
	Streams::out << "DOS:\n";
	Streams::out << dosArray << "\n\n";
	Streams::out << "Density 1D:\n";
	Streams::out << densityArray1D << "\n\n";
	Streams::out << "Density 2D:\n";
	Streams::out << densityArray2D << "\n\n";
	Streams::out << "LDOS:\n";
	Streams::out << ldosArray << "\n\n";
}
//! [PropertyConverter]
