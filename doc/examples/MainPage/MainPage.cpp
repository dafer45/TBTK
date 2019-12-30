#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("MainPage");

//! [MainPage]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Parameters.
	const int SIZE_X = 21;
	const int SIZE_Y = 21;
	const int IMPURITY_X = 10;
	const int IMPURITY_Y = 10;

	double U_S   = 1;
	double U_Imp = 1;
	double t     = 1;
	double J     = 1;
	double delta = 1;
	double mu = -2;

	//Create model.
	Model model;

	//Setup substrate.
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int spin = 0; spin < 2; spin++){
				model << HoppingAmplitude(
					U_S,
					{0, x, y, spin},
					{0, x, y, spin}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{0, x+1, y, spin},
						{0, x,   y, spin}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{0, x, y+1, spin},
						{0, x, y,   spin}
					) + HC;
				}
			}
		}
	}
	model.setChemicalPotential(mu);

	for(int spin = 0; spin < 2; spin++){
		//Setup impurity.
		model << HoppingAmplitude(        U_Imp, {1, spin}, {1, spin});
		model << HoppingAmplitude(-J*(1-2*spin), {1, spin}, {1, spin});

		//Add coupling between the substrate and impurity.
		model << HoppingAmplitude(
			delta,
			{0, IMPURITY_X, IMPURITY_Y, spin},
			{1, spin}
		) + HC;
	}

	//Construct model.
	model.construct();

	//Setup Solver
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Extract the spin-polarized LDOS in the substrate.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(-5, 5, 1000);
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{{0, _a_, _a_, IDX_SPIN}}
		);

	//Extract the magnetization in the substrate.
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{0, _a_, _a_, IDX_SPIN}
		});


	//Define the polarization axis of interest.
	Vector3d polarizationAxis({0, 0, 1});

	//Plot the spin-polarized LDOS along the line y=IMPURITY_Y in the
	//substrate.
	Plotter plotter;
	plotter.setNumContours(20);
	spinPolarizedLDOS = Smooth::gaussian(spinPolarizedLDOS, 0.1, 101);
	plotter.plot(
		{0, _a_, IMPURITY_Y, IDX_SPIN},

		polarizationAxis,
		spinPolarizedLDOS
	);
	plotter.save("figures/SpinPolarizedLDOS.png");

	//Plot the magnetization in the whole substrate.
	plotter.clear();
	plotter.plot(
		{0, _a_, _a_, IDX_SPIN},
		polarizationAxis,
		magnetization
	);
	plotter.save("figures/Magnetization.png");
}
//! [MainPage]
