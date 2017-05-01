/* Copyright 2017 Kristofer Björnson and Andreas Theiler
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

/** @file WannierParser.cpp
 *
 *  @author Kristofer Björnson
 */

#include "BasicState.h"
#include "FieldWrapper.h"
#include "ParallelepipedCell.h"
#include "ParallelepipedArrayState.h"
#include "Timer.h"
#include "UnitCell.h"
#include "Vector3d.h"
#include "WannierParser.h"

#include <fstream>
#include <sstream>

using namespace std;

namespace TBTK{

/*UnitCell* WannierParser::parseMatrixElements(string filename){
	ifstream fin(filename);

	//Throw awy first line
	string dummy;
	getline(fin, dummy);

	//Read numPoints and matrixDimension, and throw away the rest of the
	//line.
	unsigned int numPoints;
	unsigned int matrixDimension;
	fin >> numPoints;
	fin >> matrixDimension;
	getline(fin, dummy);

	//Setup UnitCell and create BasicStates
	UnitCell *unitCell = new UnitCell({
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	});
	vector<BasicState*> states;
	for(unsigned int n = 0; n < matrixDimension; n++){
		states.push_back(new BasicState({(int)n},	{0,	0,	0}));
		states[n]->setExtent(15);
		states[n]->setCoordinates({0, 0, 0});
	}

	//Read and add the matrix elements to the corresponding BasicState.
	double threshold = 1e-10;
	while(!(fin >> std::ws).eof()){
		//Read x, y, and z coordinate and throw away the rest of the
		//line.
		int x, y, z;
		fin >> x;
		fin >> y;
		fin >> z;
		getline(fin, dummy);

		//Read matrix elements.
		complex<double> amplitude;
		for(unsigned int row = 0; row < matrixDimension; row++){
			for(unsigned int col = 0; col < matrixDimension; col++){
				fin >> amplitude;
				if(abs(amplitude) > threshold){
					states[col]->addMatrixElement(
						amplitude,
						{(int)row},
						{x, y, z}
					);
				}
			}
		}
	}

	//Add BasicStates to UnitCell.
	for(unsigned int n = 0; n < matrixDimension; n++)
		unitCell->addState(states[n]);

	return unitCell;
}

vector<ParallelepipedArrayState*> WannierParser::parseWannierFunctions(
	string filename,
	unsigned int resolutionX,
	unsigned int resolutionY,
	unsigned int resolutionZ,
	unsigned int numStates,
	initializer_list<initializer_list<double>> basisVectors
){
	ifstream fin(filename);

	vector<ParallelepipedArrayState*> ppaStates;
	for(unsigned int n = 0; n < numStates; n++){
		ppaStates.push_back(
			new ParallelepipedArrayState(
				basisVectors,
				{resolutionX, resolutionY, resolutionZ}
			)
		);
		ppaStates[n]->setCoordinates({0, 0, 0});
		ppaStates[n]->setExtent(8);
	}

	unsigned int counter = 0;
	while(!(fin >> std::ws).eof()){
		double x, y, z;
		fin >> x;
		fin >> y;
		fin >> z;
		for(unsigned int n = 0; n < numStates; n++){
			complex<double> amplitude;
			fin >> amplitude;
			ppaStates[n]->setAmplitude(amplitude,	{x, y, z});
		}
		counter++;
	}

	return ppaStates;
}*/

UnitCell* WannierParser::parseMatrixElements(Resource &resource){
	stringstream ss;
	ss.str(resource.getData());

	//Throw awy first line
	string dummy;
	getline(ss, dummy);

	//Read numPoints and matrixDimension, and throw away the rest of the
	//line.
	unsigned int numPoints;
	unsigned int matrixDimension;
	ss >> numPoints;
	ss >> matrixDimension;
	getline(ss, dummy);

	//Setup UnitCell and create BasicStates
	UnitCell *unitCell = new UnitCell({
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	});
	vector<BasicState*> states;
	for(unsigned int n = 0; n < matrixDimension; n++){
		states.push_back(new BasicState({(int)n},	{0,	0,	0}));
		states[n]->setExtent(15);
		states[n]->setCoordinates({0, 0, 0});
	}

	//Read and add the matrix elements to the corresponding BasicState.
	double threshold = 1e-10;
	while(!(ss >> std::ws).eof()){
		//Read x, y, and z coordinate and throw away the rest of the
		//line.
		int x, y, z;
		ss >> x;
		ss >> y;
		ss >> z;
		getline(ss, dummy);

		//Read matrix elements.
		complex<double> amplitude;
		for(unsigned int row = 0; row < matrixDimension; row++){
			for(unsigned int col = 0; col < matrixDimension; col++){
				ss >> amplitude;
				if(abs(amplitude) > threshold){
					states[col]->addMatrixElement(
						amplitude,
						{(int)row},
						{x, y, z}
					);
				}
			}
		}
	}

	//Add BasicStates to UnitCell.
	for(unsigned int n = 0; n < matrixDimension; n++)
		unitCell->addState(states[n]);

	return unitCell;
}

vector<ParallelepipedArrayState*> WannierParser::parseWannierFunctions(
	Resource &resource,
	unsigned int resolutionX,
	unsigned int resolutionY,
	unsigned int resolutionZ,
	unsigned int numStates,
	initializer_list<initializer_list<double>> basisVectors
){
	stringstream ss;
	ss.str(resource.getData());

	vector<ParallelepipedArrayState*> ppaStates;
	for(unsigned int n = 0; n < numStates; n++){
		ppaStates.push_back(
			new ParallelepipedArrayState(
				basisVectors,
				{resolutionX, resolutionY, resolutionZ}
			)
		);
		ppaStates[n]->setCoordinates({0, 0, 0});
		ppaStates[n]->setExtent(8);
	}

	unsigned int counter = 0;
	while(!(ss >> std::ws).eof()){
		double x, y, z;
		ss >> x;
		ss >> y;
		ss >> z;
		for(unsigned int n = 0; n < numStates; n++){
			complex<double> amplitude;
			ss >> amplitude;
			ppaStates[n]->setAmplitude(amplitude,	{x, y, z});
		}
		counter++;
	}

	return ppaStates;
}

};	//End of namespace TBTK
