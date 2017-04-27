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
 *  @brief New project
 *
 *  Empty template project.
 *
 *  @author Kristofer Björnson
 */

#include "BandDiagramGenerator.h"
#include "BlockDiagonalizationSolver.h"
#include "DiagonalizationSolver.h"
#include "BPropertyExtractor.h"
#include "DOS.h"
#include "FileWriter.h"
#include "ParametrizedLine3d.h"
#include "Plotter/Plotter.h"
#include "RayTracer/RayTracer.h"
#include "Streams.h"
#include "TBTKMacros.h"
#include "Timer.h"
#include "WannierParser.h"

#include <complex>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	TBTKAssert(
		argc == 2,
		"main()",
		"Need one input parameter (filename).",
		""
	);
	Timer::tick("Full calculation");
	FileWriter::clear();

	string filename = argv[1];

/*	Timer::tick("Parse");
	WannierParser wannierParser;
	UnitCell *unitCell = wannierParser.parseMatrixElements(filename);
	ReciprocalLattice *reciprocalLattice = new ReciprocalLattice(unitCell);
	Timer::tock();

	BandDiagramGenerator bandDiagramGenerator;
	bandDiagramGenerator.setReciprocalLattice(*reciprocalLattice);
	vector<vector<double>> data1 = bandDiagramGenerator.generateBandDiagram(
		{
			{0,		0,		0},
			{M_PI/2.,	M_PI/2.,	0},
			{0,		M_PI,		0},
			{0,		0,		0},
			{0,		0,		M_PI}
		},
		10,
		{{M_PI,	M_PI,	0}}
	);
	Streams::out << "Finished.\n";
	Plotter plotter;
	plotter.setWidth(1200);
	plotter.setHeight(800);
	plotter.setHold(true);
//	for(unsigned int n = 0; n < 2*wannierParser.getMatrixDimension(); n++){
	for(unsigned int n = 0; n < 2*unitCell->getNumStates(); n++){
		Streams::out << n << "\n";
		plotter.plot(data1[n]);
	}
	plotter.save("figures/BandStructureA.png");*/

	WannierParser wannierParser;
	vector<ParallelepipedArrayState*> ppaStates = wannierParser.parseWannierFunctions(filename);

	unsigned int NUM_STATES = 10;
	Timer::tick("Ray tracing individual Wannier functions.");
	#pragma omp parallel for
	for(unsigned int n = 0; n < NUM_STATES; n++){
		RayTracer rayTracer;
		rayTracer.setCameraPosition({25, -25, 25});
		rayTracer.setUp({0, 0, 1});
		rayTracer.setFocus({0, 0, 0});
		rayTracer.setNumRaySegments(300);
		rayTracer.setRayLength(80);
		rayTracer.setWidth(120);
		rayTracer.setHeight(80);

		vector<const FieldWrapper*> fields;
		for(int x = -10; x <= 10; x++){
			for(int y = -10; y <= 10; y++){
				ParallelepipedArrayState *clone = ppaStates[n]->clone();
				clone->setCoordinates({x*7.46328, y*7.46328, 0});
				fields.push_back(new FieldWrapper(*clone));
			}
		}

		rayTracer.plot(fields);

		stringstream ss;
		ss << "figures/WannierFunction" << n << ".png";
		rayTracer.save(ss.str());
	}
	Timer::tock();

	Timer::tick("Ray tracing Wannier function.");
	RayTracer rayTracer;
	rayTracer.setCameraPosition({25, -25, 25});
	rayTracer.setUp({0, 0, 1});
	rayTracer.setFocus({0, 0, 0});
	rayTracer.setNumRaySegments(300);
	rayTracer.setRayLength(80);
	rayTracer.setWidth(120);
	rayTracer.setHeight(80);
	vector<const FieldWrapper*> fields;
	for(unsigned int n = 0; n < NUM_STATES; n++){
		for(int x = -10; x <= 10; x++){
			for(int y = -10; y <= 10; y++){
				ParallelepipedArrayState *clone = ppaStates[n]->clone();
				clone->setCoordinates({x*7.46328, y*7.46328, 0});
				fields.push_back(new FieldWrapper(*clone));
			}
		}
	}
	rayTracer.plot(fields);

	stringstream ss;
	ss << "figures/WannierFunctions.png";
	rayTracer.save(ss.str());
	Timer::tock();

	Timer::tock();

	return 0;
}
