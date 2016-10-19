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

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);
const int NUM_SAMPLES = 100;
const double CHEBYSHEV_SCALE_FACTOR = 20;
const int CHEBYSHEV_NUM_COEFFICIENTS = 5000;
const int CHEBYSHEV_ENERGY_RESOLUTION = 10000;

int main(int argc, char **argv){
	//Read input parameters
	if(argc != 2){
		cout << "Need one argument: Filename.\n";
		exit(1);
	}
	string fileName = argv[1];

	//Use GPU if devices
	bool useGPU;
	if(GPUResourceManager::getNumDevices() > 0)
		useGPU = true;
	else
		useGPU = false;

	//Parse model from file and setup output target
	Model *model = FileParser::readModel(fileName);
	model->constructCOO();

	FileWriter::setFileName("TBTKResults.h5");

	//Setup ChebyshevSolver and corresponding PropertyExtractor
	ChebyshevSolver cSolver;
	cSolver.setModel(model);
	cSolver.setScaleFactor(CHEBYSHEV_SCALE_FACTOR);

	CPropertyExtractor pe(
		&cSolver,
		CHEBYSHEV_NUM_COEFFICIENTS,
		CHEBYSHEV_ENERGY_RESOLUTION,
		useGPU,
		false,
		true,
		-CHEBYSHEV_SCALE_FACTOR,
		CHEBYSHEV_SCALE_FACTOR
	);

	//Initialize randomization and dos
	srand(time(NULL));
	double dosData[CHEBYSHEV_ENERGY_RESOLUTION];
	for(int n = 0; n < CHEBYSHEV_ENERGY_RESOLUTION; n++)
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
		for(int e = 0; e < CHEBYSHEV_ENERGY_RESOLUTION; e++)
			dosData[e] += data[e]/(double)NUM_SAMPLES;

		//Free memory
		delete ldos;

	}
	cout << "\n";

	//Write DOS to file
	Property::DOS dos(-CHEBYSHEV_SCALE_FACTOR, CHEBYSHEV_SCALE_FACTOR, CHEBYSHEV_ENERGY_RESOLUTION, dosData);
	FileWriter::writeDOS(&dos);

	return 0;
}
