/** @file Model.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Model.h"
#include "../include/Geometry.h"
#include "../include/Streams.h"

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

namespace TBTK{

Model::Model(){
	temperature = 0.;
	chemicalPotential = 0.;
	statistics = Statistics::FermiDirac;
	amplitudeSet = new AmplitudeSet();
	geometry = NULL;
	isTalkative = true;
}

Model::~Model(){
	delete amplitudeSet;
	if(geometry != NULL)
		delete geometry;
}

/*void Model::addHA(HoppingAmplitude ha){
	amplitudeSet.addHA(ha);
}

void Model::addHAAndHC(HoppingAmplitude ha){
	amplitudeSet.addHAAndHC(ha);
}*/

void Model::construct(){
	if(isTalkative)
		Util::Streams::out << "Constructing system (!)\n";

	amplitudeSet->construct();

	int basisSize = getBasisSize();

	if(isTalkative)
		Util::Streams::out << "\tBasis size: " << basisSize << "\n";
}

void Model::createGeometry(int dimensions, int numSpecifiers){
	if(!getIsConstructed()){
		Util::Streams::err << "Error in Model::createGeometry: Hilbert space basis has not been constructed yet.\n";
		exit(1);
	}
	geometry = new Geometry(dimensions, numSpecifiers, this);
}

};	//End of namespace TBTK
