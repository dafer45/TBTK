/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Basic Fourier transform example
 *
 *  Basic example of a one- and two-dimensional Fourier transform.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Array.h"
#include "TBTK/FourierTransform.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <iostream>
#include <math.h>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(int agrc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Parameters.
	const int SIZE_X = 100;
	const int SIZE_Y = 200;

	//Setup one-dimensional function f.
	Array<complex<double>> f({SIZE_X});
	for(unsigned int n = 0; n < SIZE_X; n++){
		f[{n}] = cos(2.*M_PI*3.*n/(double)SIZE_X)
			+ cos(2.*M_PI*5.*n/(double)SIZE_X);
	}

	//Calculate the transforms f -> F, and inverse transform F -> f.
	Array<complex<double>> F({SIZE_X});
	FourierTransform::forward(f.getData(), F.getData(), {SIZE_X});
	FourierTransform::inverse(F.getData(), f.getData(), {SIZE_X});

	//Split f and F into real and imaginary parts.
	Array<double> fReal({SIZE_X});
	Array<double> fImaginary({SIZE_X});
	Array<double> FReal({SIZE_X});
	Array<double> FImaginary({SIZE_X});
	for(unsigned int n = 0; n < SIZE_X; n++){
		fReal[{n}] = real(f[{n}]);
		fImaginary[{n}] = imag(f[{n}]);
		FReal[{n}] = real(F[{n}]);
		FImaginary[{n}] = imag(F[{n}]);
	}

	//Plot the result.
	Plotter plotter;
	plotter.plot(fReal, {{"label", "Real"}});
	plotter.plot(fImaginary, {{"label", "Imaginary"}});
	plotter.save("figures/f.png");
	plotter.clear();
	plotter.plot(FReal, {{"label", "Real"}});
	plotter.plot(FImaginary, {{"label", "Imaginary"}});
	plotter.save("figures/F.png");

	//Setup two-dimensional function g.
	Array<complex<double>> g({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			g[{x, y}] = sin(2.*M_PI*3.*x/(double)SIZE_X)*sin(
				2.*M_PI*7.*y/(double)SIZE_Y
			) - sin(2.*M_PI*5.*x/(double)SIZE_X)*sin(
				2.*M_PI*9.*y/(double)SIZE_Y
			);
		}
	}

	//Calculate the transforms g -> G, and inverse transform G -> g.
	Array<complex<double>> G({SIZE_X, SIZE_Y});
	FourierTransform::forward(g.getData(), G.getData(), {SIZE_X, SIZE_Y});
	FourierTransform::inverse(G.getData(), g.getData(), {SIZE_X, SIZE_Y});

	//Split g and G into real and imaginary parts.
	Array<double> gReal({SIZE_X, SIZE_Y});
	Array<double> gImaginary({SIZE_X, SIZE_Y});
	Array<double> GReal({SIZE_X, SIZE_Y});
	Array<double> GImaginary({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			gReal[{x, y}] = real(g[{x, y}]);
			gImaginary[{x, y}] = imag(g[{x, y}]);
			GReal[{x, y}] = real(G[{x, y}]);
			GImaginary[{x, y}] = imag(G[{x, y}]);
		}
	}

	//Plot the result.
	plotter.clear();
	plotter.plot(gReal);
	plotter.save("figures/gReal.png");
	plotter.clear();
	plotter.plot(gImaginary);
	plotter.save("figures/gImaginary.png");
	plotter.clear();
	plotter.plot(GReal);
	plotter.save("figures/GReal.png");
	plotter.clear();
	plotter.plot(GImaginary);
	plotter.save("figures/GImaginary.png");

	return 0;
}
