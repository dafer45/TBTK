#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Plotter");

//! [Plotter]
#include "TBTK/Array.h"
#include "TBTK/Model.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Density.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

void plotArray1D(){
	const unsigned int SIZE = 10;
	Array<double> x({SIZE});
	Array<double> y({SIZE});
	for(unsigned int n = 0; n < SIZE; n++){
		x[{n}] = 2*n;
		y[{n}] = n*n;
	}

	Plotter plotter;
	plotter.setTitle("Array 1D");
	plotter.setLabelX("x-axis");
	plotter.setLabelY("y-axis");
	plotter.plot(
		x,
		y,
		{
			{"linewidth", "2"},
			{"linestyle", "--"},
			{"label", "Custom x-values"}
		}
	);
	plotter.plot(
		y,
		{
			{"linewidth", "2"},
			{"linestyle", "-."},
			{"label", "Default x-values"}
		}
	);
	plotter.save("figures/Array1D.png");
}

void plotDefaultLineStyles(){
	const unsigned int SIZE = 100;
	Plotter plotter;
	plotter.setTitle("Default line styles");
	for(unsigned int n = 0; n < 18; n++){
		Array<double> y({SIZE});
		Range range(0, 2*M_PI, SIZE);
		for(unsigned int c = 0; c < range.getResolution(); c++)
			y[{c}] = n + sin(range[c]);
		plotter.plot(y);
	}
	plotter.save("figures/DefaultLineStyles.png");
}

void plotArray2D(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	Array<double> x({SIZE_X, SIZE_Y});
	Array<double> y({SIZE_X, SIZE_Y});
	Array<double> z({SIZE_X, SIZE_Y});
	for(unsigned int X = 0; X < SIZE_X; X++){
		for(unsigned int Y = 0; Y < SIZE_Y; Y++){
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

void plotCustomAxes(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	Array<double> x({SIZE_X, SIZE_Y});
	Array<double> y({SIZE_X, SIZE_Y});
	Array<double> z({SIZE_X, SIZE_Y});
	Range kx(0, 2*M_PI, SIZE_X);
	Range ky(0, 4*M_PI, SIZE_Y);
	for(unsigned int X = 0; X < SIZE_X; X++){
		for(unsigned int Y = 0; Y < SIZE_Y; Y++){
			x[{X, Y}] = X;
			y[{X, Y}] = Y;
			z[{X, Y}] = sin(kx[X])*sin(ky[Y]);
		}
	}

	Plotter plotter;
	plotter.setTitle("Custom axes");
	plotter.setLabelX("x-axis");
	plotter.setLabelY("y-axis");
	plotter.setAxes({
		{0, {0, 1}},
		{1, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}}
	});
	plotter.setBoundsX(0, 1);
	plotter.setBoundsY(0, 13.5);
	plotter.plot(x, y, z);
	plotter.save("figures/CustomAxes.png");
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

	PropertyExtractor::Diagonalizer propertyExtractor;
	propertyExtractor.setSolver(solver);
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
	Initialize();

	plotArray1D();
	plotDefaultLineStyles();
	plotArray2D();
	plotCustomAxes();
	plotDensity();
}
//! [Plotter]
