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
 *  @brief Wannier
 *
 *  Demonstration of the WannierParser and ReciprocalLattice.
 *
 *  @author Kristofer Björnson
 */

#include "BandDiagramGenerator.h"
#include "FileWriter.h"
#include "ParametrizedLine3d.h"
#include "Plotter/Plotter.h"
#include "PropertyExtractor/BlockDiagonalizer.h"
#include "Property/DOS.h"
#include "RayTracer/RayTracer.h"
#include "Solver/BlockDiagonalizer.h"
#include "Solver/Diagonalizer.h"
#include "Streams.h"
#include "TBTKMacros.h"
#include "Timer.h"
#include "WannierParser.h"

#include <complex>
#include <iostream>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

int main(int argc, char **argv){
	TBTKNotYetImplemented("Templates/Wannier main()");

	TBTKAssert(
		argc == 3,
		"main()",
		"Need two input parameters. Filenames for matrix elements and Wannier functions.",
		""
	);

	bool done = false;
	while(!done){
		Streams::out << "This template will download data over the "
			<< "internet. Are you sure you want to proceed? (y/n): ";
		char answer;
		cin >> answer;
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
		switch(answer){
		case 'y':
		case 'Y':
			done = true;
			break;
		case 'n':
		case 'N':
			exit(0);
		default:
			break;
		}
	}
	Streams::out << "OK\n";

	Timer::tick("Full calculation");
	FileWriter::clear();

	string filenameMatrixElements = argv[1];
	string filenameWannierFunctions = argv[2];
	Resource resource1;
	resource1.read(argv[1]);

//	Timer::tick("Parse");
	WannierParser wannierParser;
/*	UnitCell *unitCell = wannierParser.parseMatrixElements(resource1);
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
	for(unsigned int n = 0; n < 2*unitCell->getNumStates(); n++){
		Streams::out << n << "\n";
		plotter.plot(data1[n]);
	}
	plotter.save("figures/BandStructureA.png");*/

	Resource resource2;
	resource2.read(argv[2]);
/*	vector<ParallelepipedArrayState*> ppaStates = wannierParser.parseWannierFunctions(
		resource2,
		141,
		141,
		81,
		10,
		{
			{7.0*7.46328,	0.0,		0.0},
			{0.0,		7.0*7.46328,	0.0},
			{0.0,		0.0,		0.8*33.302916}
		}
	);*/
	vector<ParallelepipedArrayState*> ppaStates = wannierParser.parseWannierFunctions(
		resource2
	);

	unsigned int numWannierFunctions = ppaStates.size();
	Timer::tick("Ray tracing individual Wannier functions.");
	#pragma omp parallel for
	for(unsigned int n = 0; n < numWannierFunctions; n++){
		RayTracer rayTracer;
		rayTracer.setCameraPosition({25, -25, 25});
		rayTracer.setUp({0, 0, 1});
		rayTracer.setFocus({0, 0, 0});
		rayTracer.setNumRaySegments(300);
		rayTracer.setRayLength(80);
		rayTracer.setWidth(600);
		rayTracer.setHeight(400);

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
	for(unsigned int n = 0; n < numWannierFunctions; n++){
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
