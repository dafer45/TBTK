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
 *  @brief Generates a model from a gray scale image.
 *
 *  @author Kristofer Björsnon
 */

#include "BasicState.h"
#include "UnitCell.h"
#include "ModelFactory.h"
#include "FileParser.h"
#include "FileWriter.h"
#include "Util.h"
#include "ArrayManager.h"
#include "Lattice.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <complex>
#include <getopt.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace TBTK;

complex<double> i(0, 1.);

enum UnitCellType{Qubic, BCC, FCC};

UnitCell* setupUnitCell(UnitCellType unitCellType);

int main(int argc, char **argv){
	int isVerbose = false;
	int depth = 1;
	UnitCellType unitCellType = UnitCellType::Qubic;

	while(true){
		static struct option long_options[] = {
			//Sets flags.
			{"verbose",	no_argument,		&isVerbose,	1},
			//Does not set flags.
			{"depth",	required_argument,	0,		'd'},
			{"qubic",	no_argument,		0,		'q'},
			{"bcc",		no_argument,		0,		'b'},
			{"fcc",		no_argument,		0,		'f'},
			{0,		0,			0,		0}
		};

		int option_index = 0;
		int c = getopt_long(argc, argv, "s:b:f:", long_options, &option_index);
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
		case 'd':
			depth = atoi(optarg);
			break;
		case 'q':
			unitCellType = UnitCellType::Qubic;
			break;
		case 'b':
			unitCellType = UnitCellType::BCC;
			break;
		case 'f':
			unitCellType = UnitCellType::FCC;
			break;
		default:
			TBTKExit(
				"ImageToModel",
				"Unknown argument.",
				""
			);
		}
	}

	//Supress output if not verbose
	if(!isVerbose)
		Util::Streams::setStdMuteOut();

	//Get input file name
	TBTKAssert(
		argc == optind+1,
		"ImageToModel",
		"Input file missing.",
		""
	);
	string fileName = argv[optind];

	UnitCell *unitCell = setupUnitCell(unitCellType);

	Mat image;
	image = imread(fileName, CV_LOAD_IMAGE_COLOR);
	if(!image.data){
		cout << "Unable to open file '" << fileName << "'.";
		exit(1);
	}

	const int SIZE_X = image.rows;
	const int SIZE_Y = image.cols;
	const int SIZE_Z = depth;
	Lattice lattice(unitCell);
	int counter = 0;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int z = 0; z < SIZE_Z; z++){
				if(image.at<Vec3b>(x, y).val[0] > (z+ 1/2.)*255/(double)depth){
					lattice.addLatticePoint({x, y, z});
					counter++;
				}
			}
		}
	}
	cout << counter << "\n";

	StateSet *stateSet = lattice.generateStateSet();
	StateTreeNode stateTreeNode(*stateSet);
	Model *model = Util::ModelFactory::createModel(*stateSet, stateTreeNode);

	FileParser::writeModel(
		model,
		"ModelFile",
		FileParser::AmplitudeMode::ALL,
		""
	);
	FileWriter::writeModel(model);

	return 0;
}

UnitCell* setupUnitCell(UnitCellType unitCellType){
	UnitCell *unitCell;
	switch(unitCellType){
	case UnitCellType::Qubic:
	{
		unitCell = new UnitCell({
			{1.,	0.,	0.},
			{0.,	1.,	0.},
			{0.,	0.,	1.}
		});
		BasicState *state = new BasicState({0},	{0, 0, 0});
		state->addMatrixElement(1e-10,	{0},	{0,	0,	0});	//Addition of an infinitesimal amplitude ensures that disconnected particles are added to the Model.
		state->addMatrixElement(-1.0,	{0},	{-1,	0,	0});
		state->addMatrixElement(-1.0,	{0},	{1,	0,	0});
		state->addMatrixElement(-1.0,	{0},	{0,	-1,	0});
		state->addMatrixElement(-1.0,	{0},	{0,	1,	0});
		state->addMatrixElement(-1.0,	{0},	{0,	0,	-1});
		state->addMatrixElement(-1.0,	{0},	{0,	0,	1});
		state->setCoordinates({0., 0., 0.});
		state->setSpecifiers({0, 0});
		state->setExtent(2.);

		unitCell->addState(state);
		break;
	}
	case UnitCellType::BCC:
	{
		unitCell = new UnitCell({
			{2/sqrt(3.),	0,		0},
			{0,		2/sqrt(3.),	0},
			{0,		0,		2/sqrt(3.)}
		});

		BasicState *state = new BasicState({0}, {0, 0, 0});
		state->addMatrixElement(-1.0,	{1},	{0,	0,	0});
		state->addMatrixElement(-1.0,	{1},	{-1,	0,	0});
		state->addMatrixElement(-1.0,	{1},	{0,	-1,	0});
		state->addMatrixElement(-1.0,	{1},	{-1,	-1,	0});
		state->addMatrixElement(-1.0,	{1},	{0,	0,	-1});
		state->addMatrixElement(-1.0,	{1},	{-1,	0,	-1});
		state->addMatrixElement(-1.0,	{1},	{0,	-1,	-1});
		state->addMatrixElement(-1.0,	{1},	{-1,	-1,	-1});
		state->setCoordinates({0, 0, 0});
		state->setSpecifiers({0, 0});
		state->setExtent(2.);

		unitCell->addState(state);

		state = new BasicState({1});
		state->addMatrixElement(-1.0,	{0},	{0,	0,	0});
		state->addMatrixElement(-1.0,	{0},	{1,	0,	0});
		state->addMatrixElement(-1.0,	{0},	{0,	1,	0});
		state->addMatrixElement(-1.0,	{0},	{1,	1,	0});
		state->addMatrixElement(-1.0,	{0},	{0,	0,	1});
		state->addMatrixElement(-1.0,	{0},	{1,	0,	1});
		state->addMatrixElement(-1.0,	{0},	{0,	1,	1});
		state->addMatrixElement(-1.0,	{0},	{1,	1,	1});
		state->setCoordinates({1/sqrt(3.), 1/sqrt(3.), 1/sqrt(3.)});
		state->setSpecifiers({1, 0});
		state->setExtent(2.);

		unitCell->addState(state);
		break;
	}
	case UnitCellType::FCC:
	{
		unitCell = new UnitCell({
			{2/sqrt(2.),    0,              0},
			{0,             2/sqrt(2.),     0},
			{0,             0,              2/sqrt(2.)}
		});

		BasicState *state = new BasicState({0}, {0, 0, 0});
		state->addMatrixElement(-1.0,   {1},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {2},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {3},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {1},    {-1,    0,      0});
		state->addMatrixElement(-1.0,   {2},    {-1,    0,      0});
		state->addMatrixElement(-1.0,   {1},    {0,     -1,     0});
		state->addMatrixElement(-1.0,   {3},    {0,     -1,     0});
		state->addMatrixElement(-1.0,   {1},    {-1,    -1,     0});
		state->addMatrixElement(-1.0,   {2},    {0,     0,      -1});
		state->addMatrixElement(-1.0,   {3},    {0,     0,      -1});
		state->addMatrixElement(-1.0,   {2},    {-1,    0,      -1});
		state->addMatrixElement(-1.0,   {3},    {0,     -1,     -1});
		state->setCoordinates({0., 0., 0.});
		state->setSpecifiers({0, 0});
		state->setExtent(2.);

		unitCell->addState(state);

		state = new BasicState({1});
		state->addMatrixElement(-1.0,   {0},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {0},    {1,     0,      0});
		state->addMatrixElement(-1.0,   {0},    {0,     1,      0});
		state->addMatrixElement(-1.0,   {0},    {1,     1,      0});
		state->setCoordinates({1/sqrt(2.), 1/sqrt(2.), 0.});
		state->setSpecifiers({1, 0});
		state->setExtent(2.);

		unitCell->addState(state);
		state = new BasicState({2});
		state->addMatrixElement(-1.0,   {0},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {0},    {1,     0,      0});
		state->addMatrixElement(-1.0,   {0},    {0,     0,      1});
		state->addMatrixElement(-1.0,   {0},    {1,     0,      1});
		state->setCoordinates({1/sqrt(2.), 0., 1/sqrt(2.)});
		state->setSpecifiers({1, 0});
		state->setExtent(2.);

		unitCell->addState(state);

		state = new BasicState({3});
		state->addMatrixElement(-1.0,   {0},    {0,     0,      0});
		state->addMatrixElement(-1.0,   {0},    {0,     1,      0});
		state->addMatrixElement(-1.0,   {0},    {0,     0,      1});
		state->addMatrixElement(-1.0,   {0},    {0,     1,      1});
		state->setCoordinates({0., 1/sqrt(2.), 1/sqrt(2.)});
		state->setSpecifiers({1, 0});
		state->setExtent(2.);

		unitCell->addState(state);

		break;
	}
	default:
		TBTKExit(
			"ImageToModel",
			"Choose unit cell not yet implemented.",
			""
		);
		break;
	}

	return unitCell;
}
