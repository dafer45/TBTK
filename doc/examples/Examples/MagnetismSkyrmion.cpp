#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("MagnetismSkyrmion");

//! [MagnetismSkyrmion]
#include "TBTK/Property/Magnetization.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

complex<double> i(0, 1);

//Callback that allows for the Zeeman term (J) to be updated after the Model
//has been set up.
class JCallback : public HoppingAmplitude::AmplitudeCallback{
public:
	//Function that returns the HoppingAmplitude value for the given
	//Indices. The to- and from-Indices are identical in this example.
	complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		Subindex x = from[0];
		Subindex y = from[1];
		Subindex toSpin = to[2];
		Subindex fromSpin = from[2];

		double X = x - sizeX/2.;
		double Y = y - sizeY/2.;
		double r = sqrt(X*X + Y*Y);
		double theta = atan2(Y, X);

		double S_X = sin(2*M_PI*r/radius)*cos(theta);
		double S_Y = sin(2*M_PI*r/radius)*sin(theta);
		double S_Z = cos(2*M_PI*r/radius);
		if(r > radius){
			S_X = 0;
			S_Y = 0;
			S_Z = 1;
		}

		if(toSpin == 0 && fromSpin == 0)
			return J*S_Z;
		else if(toSpin == 0 && fromSpin == 1)
			return J*(S_X - i*S_Y);
		else if(toSpin == 1 && fromSpin == 0)
			return J*(S_X + i*S_Y);
		else
			return -J*S_Z;
	}

	//Set the value for J.
	void setJ(double J){
		this->J = J;
	}

	void setSize(double sizeX, double sizeY){
		this->sizeX = sizeX;
		this->sizeY = sizeY;
	}

	//Set Skyrmion radius.
	void setSkyrmionRadius(double radius){
		this->radius = radius;
	}
private:
	double J;
	double sizeX;
	double sizeY;
	double radius;
};

int main(){
	//Initialize TBTK.
	Initialize();

	//Parameters.
#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 21;
	const unsigned int SIZE_Y = 21;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 11;
	const unsigned int SIZE_Y = 11;
#endif //TBTK_DOCUMENTATION_NICE
	const double t = -1;
	const double mu = 0;
	const double J = 1;
	const double SKYRMION_RADIUS = 10;

	//Create a callback that returns the Zeeman term and that will be used
	//as input to the Model.
	JCallback jCallback;
	jCallback.setJ(J);
	jCallback.setSize(SIZE_X, SIZE_Y);
	jCallback.setSkyrmionRadius(SKYRMION_RADIUS);

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int spin = 0; spin < 2; spin++){
				for(unsigned int spin2 = 0; spin2 < 2; spin2++){
					model << HoppingAmplitude(
						jCallback,
						{x, y, spin},
						{x, y, spin2}
					);
				}

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						t,
						{x+1, y, spin},
						{x, y, spin}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						t,
						{x, y+1, spin},
						{x, y, spin}
					) + HC;
				}
			}
		}
	}
	model.construct();
	model.setChemicalPotential(mu);

	//Set up the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -8;
	const double UPPER_BOUND = 8;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the Magnetization.
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization(
			{{_a_, _a_, IDX_SPIN}}
		);

	Plotter plotter;
	plotter.setNumContours(20);
	plotter.setTitle("Magnetization x-axis");
	plotter.plot({_a_, _a_, IDX_SPIN}, {1, 0, 0}, magnetization);
	plotter.save("figures/MagnetizationX.png");
	plotter.clear();
	plotter.setNumContours(20);
	plotter.setTitle("Magnetization y-axis");
	plotter.plot({_a_, _a_, IDX_SPIN}, {0, 1, 0}, magnetization);
	plotter.save("figures/MagnetizationY.png");
	plotter.clear();
	plotter.setNumContours(20);
	plotter.setTitle("Magnetization z-axis");
	plotter.plot({_a_, _a_, IDX_SPIN}, {0, 0, 1}, magnetization);
	plotter.save("figures/MagnetizationZ.png");
}
//! [MagnetismSkyrmion]
