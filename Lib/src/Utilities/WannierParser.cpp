/* Copyright 2017 Kristofer Björnson
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

#include "TBTK/Atom.h"
#include "TBTK/BasicState.h"
#include "TBTK/FieldWrapper.h"
#include "TBTK/ParallelepipedCell.h"
#include "TBTK/ParallelepipedArrayState.h"
#include "TBTK/Timer.h"
#include "TBTK/UnitCell.h"
#include "TBTK/Vector3d.h"
#include "TBTK/WannierParser.h"

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
	const vector<vector<double>> &basisVectors
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

vector<ParallelepipedArrayState*> WannierParser::parseWannierFunctions(
	Resource &resource
){
	stringstream ssWithComments;
	ssWithComments.str(resource.getData());
	ssWithComments >> noskipws;

	stringstream ss;
	char c;
	while(!(ssWithComments >> c).eof()){
		if(c != '#'){
			ss << c;
		}
		else{
			string dummy;
			getline(ssWithComments, dummy);
			ss << "\n";
		}
	}

	int dimGroup = -1;
	double *primVec = nullptr;
	double *convVec = nullptr;
	double *primCoord = nullptr;

	ParallelepipedArrayState *parallelepipedArrayState;
	while(!(ss >> std::ws).eof()){
		string command;
		ss >> command;
		if(command.compare("MOLECULE") == 0){
		}
		else if(command.compare("POLYMER") == 0){
		}
		else if(command.compare("SLAB") == 0){
		}
		else if(command.compare("CRYSTAL") == 0){
		}
		else if(command.compare("DIM-GROUP") == 0){
			TBTKAssert(
				dimGroup == -1,
				"WannierParser::parseWannierFunction()",
				"Multiple definitions of DIM-GROUP encountered.",
				""
			);
			ss >> dimGroup;
		}
		else if(command.compare("PRIMVEC") == 0){
			TBTKAssert(
				primVec == nullptr,
				"WannierParser::parseWannierFunction()",
				"Multiple definitions of PRIMVEC encountered.",
				""
			);

			primVec = new double[9];
			for(unsigned int n = 0; n < 9; n++)
				ss >> primVec[n];
		}
		else if(command.compare("CONVVEC") == 0){
			TBTKAssert(
				convVec == nullptr,
				"WannierParser::parseWannierFunction()",
				"Multiple definitions of CONVVEC encountered.",
				""
			);

			convVec = new double[9];
			for(unsigned int n = 0; n < 9; n++)
				ss >> convVec[n];
		}
		else if(command.compare("PRIMCOORD") == 0){
			TBTKAssert(
				primCoord == nullptr,
				"WannierParser::parseWannierFunction()",
				"Multiple definitions of PRIMCOORD encountered.",
				""
			);

			unsigned int numCoordinates;
			ss >> numCoordinates;

			//Hrow away the rest of the line.
			string dummy;
			getline(ss, dummy);
//			ss >> dummy;

			primCoord = new double[3*numCoordinates];
			for(unsigned int n = 0; n < numCoordinates; n++){
				string word;
				ss >> word;
				Streams::out << word << "\n";

				//Check if word is a number.
				char *p;
				int i = strtol(word.c_str(), &p, 10);
				Streams::out << "*p = " << *p << "\n";
				if(*p == '\0'){
					unsigned int atomicNumber = i;
//					ss >> atomNumber;
					Streams::out << "Atomic number:\t" << atomicNumber << "\n";
				}
				else{
					string atomType = word;
//					ss >> atomType;
					Atom atom = Atom::getAtomBySymbol(word);
					Streams::out << "Atom type:\t" << atomType << ", and number:\t" << atom.getAtomicNumber() << "\n";
				}

				for(unsigned int c = 0; c < 3; c++)
					ss >> primCoord[3*n + c];

				//Throw away the rest of the line.
//				ss >> dummy;
				getline(ss, dummy);
				Streams::out << dummy << "\n";
			}
		}
		else if(
			command.compare("BEGIN_BLOCK_DATAGRID3D") == 0
			|| command.compare("BEGIN_BLOCK_DATAGRID_3D") == 0
		){
			Streams::out << "Ey!\n";
			string gridName;
			ss >> gridName;

			string beginGrid;
			ss >> beginGrid;

			unsigned int resolutionX, resolutionY, resolutionZ;
			ss >> resolutionX;
			ss >> resolutionY;
			ss >> resolutionZ;

			Vector3d origin;
			ss >> origin.x;
			ss >> origin.y;
			ss >> origin.z;

			Vector3d spanningVectors[3];
			for(unsigned int n = 0; n < 3; n++){
				ss >> spanningVectors[n].x;
				ss >> spanningVectors[n].y;
				ss >> spanningVectors[n].z;
			}

			Vector3d midPoint
				= (spanningVectors[0]
				+ spanningVectors[1]
				+ spanningVectors[2])/2.;

			Streams::out << "Resolution:\t" << resolutionX << "\t" << resolutionY << "\t" << resolutionZ << "\n";
			Streams::out << "Origin:\t" << origin << "\n";
			Streams::out << "Spanning vector 0:\t" << spanningVectors[0] << "\n";
			Streams::out << "Spanning vector 1:\t" << spanningVectors[1] << "\n";
			Streams::out << "Spanning vector 2:\t" << spanningVectors[2] << "\n";

			parallelepipedArrayState = new ParallelepipedArrayState(
				{
					{
						spanningVectors[0].x,
						spanningVectors[0].y,
						spanningVectors[0].z
					},
					{
						spanningVectors[1].x,
						spanningVectors[1].y,
						spanningVectors[1].z
					},
					{
						spanningVectors[2].x,
						spanningVectors[2].y,
						spanningVectors[2].z
					}
				},
				{resolutionX, resolutionY, resolutionZ}
			);
			parallelepipedArrayState->setCoordinates({0, 0, 0});
			parallelepipedArrayState->setExtent(8.);

			for(unsigned int n = 0; n < resolutionX*resolutionY*resolutionZ; n++){
				unsigned int x = n%resolutionX;
				unsigned int y = (n/resolutionX)%resolutionY;
				unsigned int z = n/(resolutionX*resolutionY);

				double X
					= -midPoint.x
					+ (x/*+1/2.*/)*spanningVectors[0].x/resolutionX
					+ (y/*+1/2.*/)*spanningVectors[1].x/resolutionX
					+ (z/*+1/2.*/)*spanningVectors[2].x/resolutionX;
				double Y
					= -midPoint.y
					+ (x/*+1/2.*/)*spanningVectors[0].y/resolutionY
					+ (y/*+1/2.*/)*spanningVectors[1].y/resolutionY
					+ (z/*+1/2.*/)*spanningVectors[2].y/resolutionY;
				double Z
					= -midPoint.z
					+ (x/*+1/2.*/)*spanningVectors[0].z/resolutionZ
					+ (y/*+1/2.*/)*spanningVectors[1].z/resolutionZ
					+ (z/*+1/2.*/)*spanningVectors[2].z/resolutionZ;

				complex<double> amplitude;
				ss >> amplitude;

				Streams::out << x << "\t" << y << "\t" << z << "\n";
				parallelepipedArrayState->setAmplitude(
					amplitude,
					{X, Y, Z}
				);
			}

			string endGrid;
			ss >> endGrid;
			TBTKAssert(
				endGrid.substr(0, 4).compare("END_") == 0,
				"WannierParser::parseWannierFunction()",
				"Expected END_... but found " << endGrid << ".",
				""
			);

			string endBlock;
			ss >> endBlock;
			TBTKAssert(
				endBlock.compare("END_BLOCK_DATAGRID3D") == 0
				|| endBlock.compare("END_BLOCK_DATAGRID_3D") == 0,
				"WannierParser::parseWannierFunction()",
				"Expected END_BLOCK_DATAGRID3D but found " << endBlock << ".",
				""
			);
		}
		else{
			TBTKExit(
				"WannierParser::parseWannierFunction()",
				"Encountered unknown command '" << command
				<< "' while parsing input.",
				""
			);
		}
	}

	vector<ParallelepipedArrayState*> ppaStates;
	ppaStates.push_back(parallelepipedArrayState);

	return ppaStates;
}

};	//End of namespace TBTK
