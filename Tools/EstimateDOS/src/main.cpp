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
	Util::Streams::muteOut();
	Util::Streams::muteLog();

	int isTalkative		= false;
	int forceGPU		= false;
	int forceCPU		= false;
	int numSamples		= 1;
	int scaleFactor		= 20;
	int numCoefficients	= 5000;
	int energyResolution	= 10000;

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

//	if(!isTalkative)
//		

	//Use GPU if devices
	bool useGPU;
	if(GPUResourceManager::getInstance().getNumDevices() > 0)
		useGPU = true;
	else
		useGPU = false;

	if(forceGPU && forceCPU){
		Util::Streams::err << "Error: --use-cpu and --use-gpu cannot be simultaneously specified.\n";
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
		for(int e = 0; e < energyResolution; e++)
			dosData[e] += data[e]/(double)numSamples;

		//Free memory
		delete ldos;

	}
	cout << "\n";

	//Write DOS to file
	Property::DOS dos(-scaleFactor, scaleFactor, energyResolution, dosData);
	FileWriter::writeDOS(&dos);

	delete [] dosData;

	return 0;
}
