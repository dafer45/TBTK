#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Plotter");

//! [Plotter]
#include "TBTK/Array.h"
#include "TBTK/Model.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Density.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

void plotArray1D(){
	Array<double> x({10});
	Array<double> y({10});
	for(unsigned int n = 0; n < 10; n++){
		x[{n}] = 2*n;
		y[{n}] = n*n;
	}

	Plotter plotter;
	plotter.setTitle("Array 1D");
	plotter.setLabelX("x-axis");
	plotter.setLabelY("y-axis");
	plotter.plot(x, y, {{"linewidth", "2"}, {"color", "blue"}});
	plotter.plot(y, {{"linewidth", "2"}, {"color", "red"}});
	plotter.save("figures/Array1D.png");
}

void plotArray2D(){
	Array<double> x({10, 10});
	Array<double> y({10, 10});
	Array<double> z({10, 10});
	for(unsigned int X = 0; X < 10; X++){
		for(unsigned int Y = 0; Y < 10; Y++){
			x[{X, Y}] = X;
			y[{X, Y}] = Y;
			z[{X, Y}] = pow(X - 4.5, 2) + pow(Y - 4.5, 2);
		}
	}

	Plotter plotter;
	plotter.setNumContours(10);
	plotter.setTitle("Contourf");
	plotter.setLabelX("x-axis");
	plotter.setLabelY("y-axis");
	plotter.plot(x, y, z);
	plotter.save("figures/Contourf.png");

	plotter.clear();
	plotter.setPlotMethod3D("plot_surface");
	plotter.setTitle("Plot surface");
	plotter.setLabelX("x-axis");
	plotter.setLabelY("y-axis");
	plotter.setLabelZ("z-axis");
	plotter.setRotation(30, 60);
	plotter.plot(x, y, z);
	plotter.save("figures/PlotSurface.png");
}

void plotDensity(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.setChemicalPotential(-1);
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	Property::Density density
		= propertyExtractor.calculateDensity({{_a_, _a_}});

	Plotter plotter;
	plotter.setTitle("Full density");
	plotter.plot({_a_, _a_}, density, {{"cmap", "magma"}});
	plotter.save("figures/FullDensity.png");

	plotter.clear();
	plotter.setTitle("Density cut");
	plotter.plot({_a_, SIZE_Y/2}, density, {{"linewidth", "2"}});
	plotter.save("figures/DensityCut.png");
}

int main(){
	plotArray1D();
	plotArray2D();
	plotDensity();
}
//! [Plotter]
