#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("FourierTransform");

//! [FourierTransform]
#include "TBTK/FourierTransform.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>
#include <string>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

void example1D(){
	const unsigned int SIZE = 100;
	CArray<complex<double>> input(SIZE);
	for(unsigned int n = 0; n < SIZE; n++){
		if(n < SIZE/10)
			input[n] = 1;
		else
			input[n] = 0;
	}

	CArray<complex<double>> output(100);
	FourierTransform::forward(input, output, {SIZE});

	Array<double> outputReal({SIZE});
	Array<double> outputImaginary({SIZE});
	for(unsigned int n = 0; n < SIZE; n++){
		outputReal[{n}] = real(output[n]);
		outputImaginary[{n}] = imag(output[n]);
	}

	Plotter plotter;
	plotter.setTitle("1D");
	plotter.plot(outputReal, {{"linewidth", "2"}});
	plotter.plot(outputImaginary, {{"linewidth", "2"}});
	plotter.save("figures/1D.png");
}

void example2D(){
	const unsigned int SIZE_X = 100;
	const unsigned int SIZE_Y = 100;
	Array<complex<double>> input({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			if(x < SIZE_X/20 && y < SIZE_Y/20)
				input[{x, y}] = 1;
			else
				input[{x, y}] = 0;
		}
	}

	Array<complex<double>> output({SIZE_X, SIZE_Y});
	FourierTransform::forward(
		input.getData(),
		output.getData(),
		{SIZE_X, SIZE_Y}
	);

	Array<double> outputReal({SIZE_X, SIZE_Y});
	Array<double> outputImaginary({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			outputReal[{x, y}] = real(output[{x, y}]);
			outputImaginary[{x, y}] = imag(output[{x, y}]);
		}
	}

	Plotter plotter;
	plotter.setTitle("2D real");
	plotter.plot(outputReal);
	plotter.save("figures/2DReal.png");

	plotter.clear();
	plotter.setTitle("2D imaginary");
	plotter.plot(outputImaginary);
	plotter.save("figures/2DImaginary.png");
}

void examplePlan(){
	const unsigned int SIZE = 100;
	CArray<complex<double>> input(SIZE);
	CArray<complex<double>> output(SIZE);
	FourierTransform::ForwardPlan<complex<double>> plan(
		input,
		output,
		{SIZE}
	);
	plan.setNormalizationFactor(1);
	Plotter plotter;
	plotter.setTitle("With plan");
	for(unsigned int n = 0; n < 10; n++){
		for(unsigned int x = 0; x < SIZE; x++){
			if(x < SIZE/(n+10))
				input[x] = 1;
			else
				input[x] = 0;
		}

		FourierTransform::transform(plan);

		Array<double> outputReal({SIZE});
		Array<double> outputImaginary({SIZE});
		for(unsigned int n = 0; n < SIZE; n++){
			outputReal[{n}] = real(output[n]);
			outputImaginary[{n}] = imag(output[n]);
		}

		plotter.plot(
			outputReal,
			{
				{"linewidth", "2"},
				{"color", "black"},
				{"linestyle", "-"}
			}
		);
		plotter.plot(
			outputImaginary,
			{
				{"linewidth", "2"},
				{"color", "orangered"},
				{"linestyle", "-"}
			}
		);
	}
	plotter.save("figures/WithPlan.png");
}

int main(){
	example1D();
	example2D();
	examplePlan();
}
//! [FourierTransform]
