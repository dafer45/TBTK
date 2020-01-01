#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Magnetism");

//! [Magnetism]
#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Smooth.h"
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
		Subindex spin = from[2];
		return J*(1 - 2*spin);
	}

	//Set the value for J.
	void setJ(double J){
		this->J = J;
	}
private:
	double J;
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

	//Create a callback that returns the Zeeman term and that will be used
	//as input to the Model.
	JCallback jCallback;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned int spin = 0; spin < 2; spin++){
				model << HoppingAmplitude(
					jCallback,
					{x, y, spin},
					{x, y, spin}
				);

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

	//Number of iterations.
	const unsigned int NUM_ITERATIONS = 10;

	//Array to store the z-component of the magnetization in.
	Array<double> magnetizationZ({NUM_ITERATIONS});

	//Iterate over 100 values for J.
	Range j(0, 5, NUM_ITERATIONS);
	for(unsigned int n = 0; n < NUM_ITERATIONS; n++){
		//Update the callback with the current value of J.
		jCallback.setJ(j[n]);

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
				{{SIZE_X/2, SIZE_Y/2, IDX_SPIN}}
			);

		//Project the magnetization on the z-axis and store the value
		//in magnetizationZ.
		magnetizationZ[{n}] = Vector3d::dotProduct(
			magnetization({
				SIZE_X/2,
				SIZE_Y/2,
				IDX_SPIN
			}).getSpinVector(),
			{0, 0, 1}
		);

		//If J has its middle value, Calculate and plot the density of
		//states (DOS).
		if(n == NUM_ITERATIONS/2){
			//Calculate the DOS.
			Property::DOS dos = propertyExtractor.calculateDOS();

			//Calculate the DOS for spin up and down separately by
			//summing the LDOS over all lattice sites.
			Property::LDOS ldos = propertyExtractor.calculateLDOS(
				{{IDX_SUM_ALL, IDX_SUM_ALL, _a_}}
			);

			//Smooth the DOS.
			const double SMOOTHING_SIGMA = 0.2;
			const unsigned int SMOOTHING_WINDOW = 101;
			dos = Smooth::gaussian(
				dos,
				SMOOTHING_SIGMA,
				SMOOTHING_WINDOW
			);

			//Smooth the LDOS.
			ldos = Smooth::gaussian(
				ldos,
				SMOOTHING_SIGMA,
				SMOOTHING_WINDOW
			);

			//Plot the DOS.
			Plotter plotter;
			plotter.setTitle("DOS");
			plotter.setLabelX("Energy");
			plotter.setLabelY("DOS");
			plotter.plot(dos, {{"label", "Total"}});

			//Plot the LDOS.
			plotter.plot(
				{IDX_SUM_ALL, IDX_SUM_ALL, 0},
				ldos,
				{
					{"color", "red"},
					{"linestyle", "--"},
					{"label", "Spin up"}
				}
			);
			plotter.plot(
				{IDX_SUM_ALL, IDX_SUM_ALL, 1},
				ldos,
				{
					{"color", "blue"},
					{"linestyle", "--"},
					{"label", "Spin down"}
				}
			);

			//Save the plot.
			plotter.save("figures/DOS.png");
		}
	}

	//Plot the Magnetization along the z-axis.
	Plotter plotter;
	plotter.setTitle("Magnetization along the  z-axis");
	plotter.setLabelX("J");
	plotter.setLabelY("Magnetization");
	plotter.setAxes({
		{0, {0, 5}}
	});
	plotter.plot(magnetizationZ);
	plotter.save("figures/Magnetization.png");
}
//! [Magnetism]
