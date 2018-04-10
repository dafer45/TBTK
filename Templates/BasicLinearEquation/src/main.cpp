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
 *  @brief Basic linear equation example
 *
 *  Basic example of solving a linear equation Hx = y for x.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FileWriter.h"
#include "TBTK/Model.h"
#include "TBTK/MatrixElement.h"
#include "TBTK/Plotter.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/LinearEquationSolver.h"
#include "TBTK/Timer.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Plot;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	int NUM_SITES = 1000;

	//Setup Model.
	Model model;
	for(int n = 0; n < NUM_SITES; n++){
		model << MatrixElement(2., {n}, {n});
		if(n > 0)
			model << MatrixElement(-1., {n}, {n-1}) + HC;
	}
	model.construct();

	for(int n = 0; n < NUM_SITES; n++){
		if((n > 100 && n < 150) || (n > 300 && n < 350))
			model << SourceAmplitude(1, {n});
		else
			model << SourceAmplitude(0, {n});
	}

	//Solve
	Solver::LinearEquationSolver solver;
	solver.setModel(model);
	solver.run();

	//Extract the result.
	const Matrix<complex<double>> result0 = solver.getResult();
	vector<complex<double>> result;
	for(unsigned int n = 0; n < result0.getNumRows(); n++)
		result.push_back(result0.at(n, 0));

	vector<double> data;
	for(unsigned int n = 0; n < result.size(); n++)
		data.push_back(real(result[n]));

	//Plot.
	Plotter plotter;
	plotter.plot(data);
	plotter.save("figures/Solution.png");

	return 0;
}
