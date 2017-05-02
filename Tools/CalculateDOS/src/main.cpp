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

/** @package TBTKTools
 *  @file main.cpp
 *  @brief Calculates the DOS using diagonalization.
 *
 *  Reads model from text file and calculates the DOS using the
 *  DiagonalizationSolver.
 *
 *  Takes filename of file to parse as first argument.
 *
 *  @author Kristofer Björsnon
 */

#include "Model.h"
#include "FileParser.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"
#include "DOS.h"
#include "FileWriter.h"
#include "Timer.h"
#include "Streams.h"

#include <complex>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <getopt.h>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);
int main(int argc, char **argv){
	Streams::openLog();
	int isVerbose			= false;
	int energyResolution		= 10000;
	bool useDefaultLowerBound	= true;
	bool useDefaultUpperBound	= true;
	double lowerBound		= -1.;
	double upperBound		= 1.;

	while(true){
		static struct option long_options[] = {
			//Sets flags.
			{"verbose",		no_argument,		&isVerbose,	1},
			//Does not set flags.
			{"lower-bound",		required_argument,	0,		'l'},
			{"upper-bound",		required_argument,	0,		'u'},
			{"energy-resolution",	required_argument,	0,		'r'},
			{0,			0,			0,		0}
		};

		int option_index = 0;
		int c = getopt_long(argc, argv, "l:u:r:", long_options, &option_index);
		if(c == -1)
			break;

		switch(c){
		case 0:
			//If the option sets a flag, do nothing.
			if(long_options[option_index].flag != 0)
				break;
			Streams::err << "option " << long_options[option_index].name;
			if(optarg)
				Streams::err << " with argument " << optarg;
			Streams::err << "\n";
			break;
		case 'r':
			energyResolution = atoi(optarg);
			break;
		case 'l':
			lowerBound = atof(optarg);
			useDefaultLowerBound = false;
			break;
		case 'u':
			upperBound = atof(optarg);
			useDefaultUpperBound = false;
			break;
		default:
			TBTKExit(
				"CalculateDOS",
				"Unknown argument.",
				""
			);
		}
	}

	//Supress output if not verbose
	if(!isVerbose)
		Streams::setStdMuteOut();

	//Get input file name
	TBTKAssert(
		argc == optind+1,
		"CalculateDOS",
		"Input file missing.",
		""
	);
	string fileName = argv[optind];

	//Parse model from file and setup output target
	Model *model = FileParser::readModel(fileName);
	model->constructCOO();

	FileWriter::setFileName("TBTKResults.h5");

	//Setup and run DiagonalizationSolver and corresponding PropertyExtractor
	DiagonalizationSolver dSolver;
	dSolver.setModel(*model);
	dSolver.run();
	DPropertyExtractor pe(dSolver);
	pe.setEnergyWindow(
		lowerBound,
		upperBound,
		energyResolution
	);

	//Setup default bounds
	if(useDefaultLowerBound)
		lowerBound = pe.getEigenValue(0);
	if(useDefaultUpperBound)
		upperBound = pe.getEigenValue(model->getBasisSize()-1);
	double energyRange = upperBound - lowerBound;
	if(useDefaultLowerBound)
		lowerBound -= energyRange/20.;
	if(useDefaultUpperBound)
		upperBound += energyRange/20.;

	//Write DOS to file
	Property::DOS dos = pe.calculateDOS();
	FileWriter::writeDOS(dos);

	delete model;

	Streams::closeLog();
	return 0;
}
