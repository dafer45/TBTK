/* Copyright 2017 Kristofer Björnson
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
 *  @brief Interactive plotting of LDOS using RayTracer.
 *
 *  Demonstrates the use of RayTracer::plotInteractive() to interactively plot
 *  the local density of states.
 *
 *  @author Kristofer Björnson
 */

#include "Solver/Diagonalizer.h"
#include "DOS.h"
#include "PropertyExtractor/Diagonalizer.h"
#include "FileWriter.h"
#include "Model.h"
#include "Plotter/Plotter.h"
#include "RayTracer/RayTracer.h"
#include "Smooth.h"
#include "Streams.h"
#include "Timer.h"

#include <cmath>
#include <complex>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	//Parameters.
	const int SIZE_X = 20;
	const int SIZE_Y = 20;
	complex<double> mu = -1;
	complex<double> t = 1;

	//Create model.
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				model << HoppingAmplitude(
					-mu,
					{x,	y,	s},
					{x,	y,	s}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{(x+1)%SIZE_X,	y,	s},
						{x,		y,	s}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{x,	(y+1)%SIZE_Y,	s},
						{x,		y,	s}
					) + HC;
				}
			}
		}
	}
	model.construct();

	//Create geometry.
	model.createGeometry(3, 0);
	Geometry *geometry = model.getGeometry();;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				geometry->setCoordinates(
					{x, y, s},
					{(double)x, (double)y, 0}
				);
			}
		}
	}

	//Choose Diagonalizer.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Setup PropertyExtractor and calculate LDOS.
	PropertyExtractor::Diagonalizer pe(solver);
	pe.setEnergyWindow(-10, 10, 1000);
	Property::LDOS ldos = pe.calculateLDOS({
		{IDX_ALL, IDX_ALL, IDX_SUM_ALL}
	});

	//Parameters used by the ray tracer to perform Gaussian smoothing on
	//the LDOS data.
	const double SIGMA = 0.15;
	unsigned int WINDOW_SIZE = 101;

	//Numbers of times a ray is deflected before the ray tracing is
	//terminated.
	unsigned int NUM_DEFLECTIONS = 1;

	//Setup and run RayTracer.
	RayTracer rayTracer;
	rayTracer.setCameraPosition({-SIZE_X/2., -SIZE_Y/2., 7.5});
	rayTracer.setUp({0, 0, 1});
	rayTracer.setFocus({SIZE_X/2, SIZE_Y/2, 0});
	rayTracer.setWidth(1200);
	rayTracer.setHeight(800);
	rayTracer.setNumDeflections(NUM_DEFLECTIONS);
	rayTracer.interactivePlot(model, ldos, SIGMA, WINDOW_SIZE);

	return 0;
}
