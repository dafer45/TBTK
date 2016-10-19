/** @package TBTKTools
 *  @file main.cpp
 *  @brief Estimates the DOS using random smapling of LDOS.
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
#include "Util.h"
#include "GPUResourceManager.h"

#include <complex>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <getopt.h>

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);
int NUM_SAMPLES = 100;
int SCALE_FACTOR = 20;
int NUM_COEFFICIENTS = 5000;
int ENERGY_RESOLUTION = 10000;

int main(int argc, char **argv){
	int isTalkative	= false;
	int forceGPU	= false;
	int forceCPU = false;

	while(true){
		static struct option long_options[] = {
			//Sets flags.
			{"verbose",		no_argument,		&isTalkative,	1},
			{"use-gpu",		no_argument,		&forceGPU,	1},
			{"use-cpu",		no_argument,		&forceCPU,	1},
			//Does not set flags.
			{"scale-factor",	required_argument,	0,		's'},
			{"coefficients",	required_argument,	0,		'c'},
			{"energy-resolution",	required_argument,	0,		'r'},
			{"samples",		required_argument,	0,		'S'},
			{0,			0,			0,		0}
		};

		int option_index = 0;
		int c = getopt_long(argc, argv, "s:c:r:S:", long_options, &option_index);
		if(c == -1)
			break;

		cout << (char)c << "\n";
		switch(c){
		case 0:
			//If the option sets a flag, do nothing.
			if(long_options[option_index].flag != 0)
				break;
			cout << "option " << long_options[option_index].name;
			if(optarg)
				cout << " with argument " << optarg;
			cout << "\n";
			break;
		case 's':
			SCALE_FACTOR = atof(optarg);
			break;
		case 'c':
			NUM_COEFFICIENTS = atoi(optarg);
			break;
		case 'r':
			ENERGY_RESOLUTION = atoi(optarg);
			break;
		case 'S':
			NUM_SAMPLES = atoi(optarg);
			break;
		default:
			cout << "Error: Unknown argument.\n";
			exit(1);
		}
	}

	//Get input file name
	if(argc != optind+1){
		cout << "Input file missing.\n";
		exit(1);
	}
	string fileName = argv[optind];

	//Use GPU if devices
	bool useGPU;
	if(GPUResourceManager::getNumDevices() > 0)
		useGPU = true;
	else
		useGPU = false;

	if(forceGPU && forceCPU){
		cout << "Error: useCPU and useGPU cannot be simultaneously specified.\n";
		exit(1);
	}
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
	cSolver.setScaleFactor(SCALE_FACTOR);

	CPropertyExtractor pe(
		&cSolver,
		NUM_COEFFICIENTS,
		ENERGY_RESOLUTION,
		useGPU,
		false,
		true,
		-SCALE_FACTOR,
		SCALE_FACTOR
	);

	//Initialize randomization and dos
	srand(time(NULL));
	double dosData[ENERGY_RESOLUTION];
	for(int n = 0; n < ENERGY_RESOLUTION; n++)
		dosData[n] = 0.;

	//Main loop: Repeatedly calculate LDOS for random sites
	for(int n = 0; n < NUM_SAMPLES; n++){
		//Print progress
		cout << "." << flush;
		if(n%10 == 9)
			cout << " ";
		if(n%50 == 49)
			cout << "\n";

		//Get new random index
		int b = rand()%model->getBasisSize();
		Index index = model->getAmplitudeSet()->tree.getPhysicalIndex(b);

		//Calculate LDOS
		Property::LDOS *ldos = pe.calculateLDOS(
			index,
			index.getUnitRange()
		);

		//Add calculated LDOS to total DOS
		const double *data = ldos->getData();
		for(int e = 0; e < ENERGY_RESOLUTION; e++)
			dosData[e] += data[e]/(double)NUM_SAMPLES;

		//Free memory
		delete ldos;

	}
	cout << "\n";

	//Write DOS to file
	Property::DOS dos(-SCALE_FACTOR, SCALE_FACTOR, ENERGY_RESOLUTION, dosData);
	FileWriter::writeDOS(&dos);

	return 0;
}
