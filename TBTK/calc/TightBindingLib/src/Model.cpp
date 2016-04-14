/** @file Model.cpp
 *
 *  @author Kristofer Björnson
 */

#include <iostream>
#include "../include/Model.h"
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

namespace TBTK{

Model::Model(){
	temperature = 0.;
	chemicalPotential = 0.;
	isTalkative = true;
}

Model::~Model(){
}

/*void Model::addHA(HoppingAmplitude ha){
	amplitudeSet.addHA(ha);
}

void Model::addHAAndHC(HoppingAmplitude ha){
	amplitudeSet.addHAAndHC(ha);
}*/

void Model::construct(){
	if(isTalkative)
		cout << "Constructing system\n";

	amplitudeSet.construct();

	int basisSize = getBasisSize();
	
	if(isTalkative)
		cout << "\tBasis size: " << basisSize << "\n";
}

};	//End of namespace TBTK
