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

#include "FileWriter.h"
#include "FourierTransform/FourierTransform.h"

#include <iostream>
#include <math.h>

using namespace std;
using namespace TBTK;

int main(int agrc, char **argv){
	const int SIZE_X = 100;
	const int SIZE_Y = 200;

	FileWriter::clear();

	//Setup one-dimensional function f.
	complex<double> *f = new complex<double>[SIZE_X];
	for(int n = 0; n < SIZE_X; n++)
		f[n] = cos(2.*M_PI*3.*n/(double)SIZE_X) + cos(2.*M_PI*5.*n/(double)SIZE_X);

	//Calculate the transforms f -> F, and inverse transform F -> f.
	complex<double> *F = new complex<double>[SIZE_X];
	FourierTransform::forward(f, F, SIZE_X);
	FourierTransform::inverse(F, f, SIZE_X);

	//Split f and F into real and imaginary parts.
	double *fR = new double[SIZE_X];
	double *fI = new double[SIZE_X];
	double *FR = new double[SIZE_X];
	double *FI = new double[SIZE_X];
	for(int n = 0; n < SIZE_X; n++){
		fR[n] = real(f[n]);
		fI[n] = imag(f[n]);
		FR[n] = real(F[n]);
		FI[n] = imag(F[n]);
	}

	//Write real and imaginary parts of f and F to file.
	const int RANK = 1;
	int dims[RANK] = {SIZE_X};
	FileWriter::write(fR, RANK, dims, "fR");
	FileWriter::write(fI, RANK, dims, "fI");
	FileWriter::write(FR, RANK, dims, "FR");
	FileWriter::write(FI, RANK, dims, "FI");

	//Setup two-dimensional function g
	complex<double> *g = new complex<double>[SIZE_X*SIZE_Y];
	for(int x = 0; x < SIZE_X; x++)
		for(int y = 0; y < SIZE_Y; y++)
			g[x*SIZE_Y + y] = sin(2.*M_PI*3.*x/(double)SIZE_X)*sin(2.*M_PI*7.*y/(double)SIZE_Y) - sin(2.*M_PI*5.*x/(double)SIZE_X)*sin(2.*M_PI*9.*y/(double)SIZE_Y);

	//Calculate the transforms g -> G, and inverse transform G -> g.
	complex<double> *G = new complex<double>[SIZE_X*SIZE_Y];
	FourierTransform::forward(g, G, SIZE_X, SIZE_Y);
	FourierTransform::inverse(G, g, SIZE_X, SIZE_Y);

	//Split g and G into real and imaginary parts.
	double *gR = new double[SIZE_X*SIZE_Y];
	double *gI = new double[SIZE_X*SIZE_Y];
	double *GR = new double[SIZE_X*SIZE_Y];
	double *GI = new double[SIZE_X*SIZE_Y];
	for(int n = 0; n < SIZE_X*SIZE_Y; n++){
		gR[n] = real(g[n]);
		gI[n] = imag(g[n]);
		GR[n] = real(G[n]);
		GI[n] = imag(G[n]);
	}

	//Write real and imaginary parts of g and G to file.
	const int RANK2 = 2;
	int dims2[RANK2] = {SIZE_X, SIZE_Y};
	FileWriter::write(gR, RANK2, dims2, "gR");
	FileWriter::write(gI, RANK2, dims2, "gI");
	FileWriter::write(GR, RANK2, dims2, "GR");
	FileWriter::write(GI, RANK2, dims2, "GI");

	return 0;
}
