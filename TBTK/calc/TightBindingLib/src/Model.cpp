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

Model::Model(int mode, int numEigenstates){
	this->mode = mode;
	this->numEigenstates = numEigenstates;
}

Model::~Model(){
}

void Model::addHA(HoppingAmplitude ha){
	amplitudeSet.addHA(ha);
}

void Model::addHAAndHC(HoppingAmplitude ha){
	amplitudeSet.addHAAndHC(ha);
}

void Model::construct(){
	cout << "Constructing system\n";

	amplitudeSet.construct();

	int basisSize = getBasisSize();
	cout << "\tBasis size: " << basisSize << "\n";
}
