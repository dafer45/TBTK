/** Reads model from text file and estimates the DOS using the
 *  ChebyshevSolver.
 *
 *  Takes filename of file to parse as first argument.
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
	if(argc != 2){
		cout << "Need one argument: Filename.\n";
		exit(1);
	}
	string fileName = argv[1];

	bool useGPU;
	if(GPUResourceManager::getNumDevices() > 0)
		useGPU = true;
	else
		useGPU = false;

	Model *model = FileParser::readModel(fileName);
	model->constructCOO();

	//For testing: calculating the DOS for the model.
	FileWriter::setFileName("TBTKResults.h5");
	FileWriter::clear();

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

	srand(time(NULL));
	double dos_data[CHEBYSHEV_ENERGY_RESOLUTION];
	for(int n = 0; n < CHEBYSHEV_ENERGY_RESOLUTION; n++)
		dos_data[n] = 0.;
	for(int n = 0; n < NUM_SAMPLES; n++){
		cout << "." << flush;
		if(n%10 == 9)
			cout << " ";
		if(n%50 == 49)
			cout << "\n";

		int b = rand()%model->getBasisSize();
		Index index = model->getAmplitudeSet()->tree.getPhysicalIndex(b);
		Property::LDOS *ldos = pe.calculateLDOS(
			index,
			index.getUnitRange()
		);
		const double *data = ldos->getData();
		for(int e = 0; e < CHEBYSHEV_ENERGY_RESOLUTION; e++)
			dos_data[e] += data[e]/(double)NUM_SAMPLES;

		delete ldos;

	}
	cout << "\n";

	Property::DOS dos(-CHEBYSHEV_SCALE_FACTOR, CHEBYSHEV_SCALE_FACTOR, CHEBYSHEV_ENERGY_RESOLUTION, dos_data);
	FileWriter::writeDOS(&dos);

	return 0;
}
