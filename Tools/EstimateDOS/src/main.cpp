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
 *  @brief Estimates the DOS using random sampling of LDOS.
 *
 *  Reads model from text file and estimates the DOS using the
 *  ChebyshevSolver.
 *
 *  Takes filename of file to parse as first argument.
 *
 *  @author Kristofer Björsnon
 */

#include "Model.h"
#include "FileParser.h"
#include "ChebyshevSolver.h"
#include "CPropertyExtractor.h"
#include "DOS.h"
#include "FileWriter.h"
#include "Timer.h"
#include "GPUResourceManager.h"
#include "Streams.h"

#include <complex>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <getopt.h>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

Index* getIndexPattern(string patternString);

int main(int argc, char **argv){
	Streams::openLog();
	int isVerbose		= false;
	int forceGPU		= false;
	int forceCPU		= false;
	int numSamples		= 100;
	int scaleFactor		= 20;
	int numCoefficients	= 5000;
	int energyResolution	= 10000;
	Index *pattern		= NULL;

	while(true){
		static struct option long_options[] = {
			//Sets flags.
			{"verbose",		no_argument,		&isVerbose,	1},
			{"use-gpu",		no_argument,		&forceGPU,	1},
			{"use-cpu",		no_argument,		&forceCPU,	1},
			//Does not set flags.
			{"scale-factor",	required_argument,	0,		's'},
			{"coefficients",	required_argument,	0,		'c'},
			{"energy-resolution",	required_argument,	0,		'r'},
			{"samples",		required_argument,	0,		'S'},
			{"index-pattern",	required_argument,	0,		'i'},
			{0,			0,			0,		0}
		};

		int option_index = 0;
		int c = getopt_long(argc, argv, "s:c:r:S:i:", long_options, &option_index);
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
		case 's':
			scaleFactor = atof(optarg);
			break;
		case 'c':
			numCoefficients = atoi(optarg);
			break;
		case 'r':
			energyResolution = atoi(optarg);
			break;
		case 'S':
			numSamples = atoi(optarg);
			break;
		case 'i':
			pattern = getIndexPattern(optarg);
			break;
		default:
			TBTKExit(
				"EstimateDOS",
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
		"EstimateDOS",
		"Input file missing.",
		""
	);
	string fileName = argv[optind];

	//Use GPU if devices
	bool useGPU;
	if(GPUResourceManager::getInstance().getNumDevices() > 0)
		useGPU = true;
	else
		useGPU = false;

	TBTKAssert(
		!(forceGPU && forceCPU),
		"EstimateDOS",
		"--use-cpu and --use-gpu cannot be simultaneously specified.",
		""
	);
	if(forceGPU)
		useGPU = true;
	if(forceCPU)
		useGPU = false;

	//Parse model from file and setup output target
	Model *model = FileParser::readModel(fileName);
	model->constructCOO();

	FileWriter::setFileName("TBTKResults.h5");

	//Setup ChebyshevSolver and corresponding PropertyExtractor
	ChebyshevSolver cSolver;
	cSolver.setModel(model);
	cSolver.setScaleFactor(scaleFactor);

	CPropertyExtractor pe(
		&cSolver,
		numCoefficients,
		energyResolution,
		useGPU,
		false,
		true,
		-scaleFactor,
		scaleFactor
	);

	//Initialize randomization and dos
	srand(time(NULL));
	double *dosData = new double[energyResolution];
	for(int n = 0; n < energyResolution; n++)
		dosData[n] = 0.;

	//Main loop: Repeatedly calculate LDOS for random sites
	for(int n = 0; n < numSamples; n++){
		//Get new random index
		int b = rand()%model->getBasisSize();
		Index index = model->getAmplitudeSet()->tree.getPhysicalIndex(b);

		//Ensure index conforms to index pattern
		if(pattern != NULL){
			if(!pattern->equals(index, true)){
				n--;
				continue;
			}
		}

		//Calculate LDOS
		Property::LDOS *ldos = pe.calculateLDOS(
			index,
			index.getUnitRange()
		);

		//Add calculated LDOS to total DOS
		const double *data = ldos->getData();
		for(int e = 0; e < energyResolution; e++)
			dosData[e] += data[e]/(double)numSamples;

		//Free memory
		delete ldos;

		//Print progress
		cout << "." << flush;
		if(n%10 == 9)
			cout << " ";
		if(n%50 == 49)
			cout << "\n";
	}
	cout << "\n";

	//Write DOS to file
	Property::DOS dos(-scaleFactor, scaleFactor, energyResolution, dosData);
	FileWriter::writeDOS(&dos);

	delete [] dosData;
	delete model;

	Streams::closeLog();
	return 0;
}

Index* getIndexPattern(string patternString){
	TBTKAssert(
		patternString[0] == '{',
		"EstimateDOS",
		"Expected '{' while reading index pattern, found '" << patternString[0] << "'.",
		"Specify index pattern using the format \"{X, X, X}\"."
	);

	vector<int> patternVector;
	unsigned int numSeparators = 0;
	bool parsingNumeric = false;
	int numeric = 0;
	for(unsigned int n = 1; n < patternString.size(); n++){
		switch(patternString[n]){
		case '*':
			TBTKAssert(
				patternVector.size() == numSeparators,
				"EstimateDOS",
				"Expected ',' while reading index, found '*'.",
				"Specify index pattern using the format \"{X, X, X}\"."
			);
			TBTKAssert(
				!parsingNumeric,
				"EstimateDOS",
				"Found '*' while parsing numeric value.",
				"Specify index pattern using the format \"{X, X, X}\"."
			);

			patternVector.push_back(IDX_ALL);
			break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
		{
			const int ASCII_OFFSET = 48;
			numeric = 10*numeric + (int)patternString[n] - ASCII_OFFSET;
			parsingNumeric = true;
			break;
		}
		case ' ':
			if(parsingNumeric){
				TBTKAssert(
					patternVector.size() == numSeparators,
					"EstimateDOS",
					"Expected ',' while reading index, found '*'.",
					"Specify index pattern using the format \"{X, X, X}\"."
				);

				patternVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			break;
		case ',':
			if(parsingNumeric){
				TBTKAssert(
					patternVector.size() == numSeparators,
					"EstimateDOS",
					"Expected ',' while reading index, found '*'.",
					"Specify index pattern using the format \"{X, X, X}\"."
				);

				patternVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			numSeparators++;
			break;
		case '}':
			if(parsingNumeric){
				TBTKAssert(
					patternVector.size() == numSeparators,
					"EstimateDOS",
					"Expected ',' while reading index, found '*'.",
					"Specify index pattern using the format \"{X, X, X}\"."
				);

				patternVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			n = patternString.size();
			break;
		default:
			TBTKExit(
					"EstimateDOS",
					"Found '" << patternString[n] << "' while parsing the interior of the index pattern.",
					"Specify index pattern using the format \"{X, X, X}\"."
			);
			break;
		}
	}

	TBTKAssert(
		patternString[patternString.size()-1] == '}',
		"EstimateDOS",
		"Expected '}' while reading index pattern, found '" << patternString[patternString.size()-1] << "'.",
		"Specify index pattern using the format \"{X, X, X}\"."
	);

	Index *pattern = new Index(patternVector);
	return pattern;
}
