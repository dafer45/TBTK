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

#include "PropertyExtractor/Diagonalizer.h"
#include "FileWriter.h"
#include "Solver/LinearEquationSolver/LinearEquationSolver.h"
#include "Model.h"
#include "MatrixElement.h"
#include "Plotter/Plotter.h"
#include "Timer.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Plot;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	TBTKNotYetImplemented("BasicLinearEquation");

	Timer::tick("Full calculation");
	int NUM_SITES = 1000;

	Model model;
	for(int n = 0; n < NUM_SITES; n++){
		model << MatrixElement(2., {n}, {n});
		if(n > 0)
			model << MatrixElement(-1., {n}, {n-1}) + HC;
	}
	model.construct();
	model.constructCOO();

	Timer::tick("Solve");
	Solver::LinearEquationSolver solver;
	solver.setModel(model);
	vector<complex<double>> b;
	for(int n = 0; n < NUM_SITES; n++){
		if((n > 100 && n < 150) || (n > 300 && n < 350))
			b.push_back(1);
		else
			b.push_back(0);
//		b.push_back(1);
	}
	vector<complex<double>> result = solver.solve(b);
	Timer::tock();

	vector<double> data;
	for(unsigned int n = 0; n < result.size(); n++)
		data.push_back(real(result[n]));

	Plotter plotter;
	plotter.plot(data);
	plotter.save("figures/Solution.png");

//	for(int n = 0; n < NUM_SITES; n++)
//		Streams::out << result[n] << "\n";

	Timer::tock();

	return 0;
}
