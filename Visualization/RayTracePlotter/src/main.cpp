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

/** @package TBTKVisualization
 *  @file main.cpp
 *  @brief Ray tracing plotter
 *
 *  Plots Properties using ray tracing.
 *
 *  @author Kristofer Björnson
 */

#include "Density.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "IndexTree.h"
#include "Model.h"
#include "Vector3d.h"

#include <algorithm>
#include <complex>
#include <vector>

#include <getopt.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace TBTK;

const complex<double> i(0, 1);

Vector3d getCoordinate(string coordinateString);

int main(int argc, char **argv){
	int isVerbose = false;
	string propertyName = "";
	Vector3d cameraPosition({0, 0, 10});
	Vector3d focus({0, 0, 0});
	Vector3d up({0, 1, 0});
	unsigned int width = 600;
	unsigned int height = 400;
	double scale = 0.1/3.;
	double radius = 0.2;
	string inputName = "";
	string outputName = "";

	//Specific for WaveFunction
	unsigned int waveFunctionState = 0;

	while(true){
		static struct option long_options[] = {
			//Set flags
			{"verbose",		no_argument,		&isVerbose,	1},
			//Does not set flags
			{"property",		required_argument,	0,		'P'},
			{"position",		required_argument,	0,		'p'},
			{"focus",		required_argument,	0,		'f'},
			{"up",			required_argument,	0,		'u'},
			{"width",		required_argument,	0,		'w'},
			{"height",		required_argument,	0,		'h'},
			{"radius",		required_argument,	0,		'r'},
			{"input",		required_argument,	0,		'i'},
			{"output",		required_argument,	0,		'o'},
			{"wave-function-state",	required_argument,	0,		's'},
			{0,			0,			0,		0}
		};

		int option_index = 0;
		int c = getopt_long(argc, argv, "P:p:f:u:w:h:r:i:o:s:", long_options, &option_index);
		if(c == -1)
			break;

		switch(c){
		case 0:
			//If the option sets a flag, do nothing.
			if(long_options[option_index].flag != 0)
				break;
			Streams::err << "option " << long_options[option_index].name;
			if(optarg)
				Streams::err << " with argument " << optarg;
			Streams::err << "\n";
			break;
		case 'P':
			propertyName = optarg;
			break;
		case 'p':
			cameraPosition = getCoordinate(optarg);
			break;
		case 'f':
			focus = getCoordinate(optarg);
			break;
		case 'u':
			up = getCoordinate(optarg);
			break;
		case 'w':
			width = atoi(optarg);
			break;
		case 'h':
			height = atoi(optarg);
			break;
		case 'r':
			radius = atof(optarg);
			break;
		case 'i':
			inputName = optarg;
			break;
		case 'o':
			outputName = optarg;
			break;
		case 's':
			waveFunctionState = atoi(optarg);
			break;
		default:
			TBTKExit(
				"TBTKRayTracePlotter",
				"Unknown argument",
				""
			);
		}
	}

	if(!isVerbose)
		Streams::setStdMuteOut();

	Vector3d basisY = up.unit();
	Vector3d basisX = ((focus-cameraPosition)*up).unit();
	up = (basisX*(focus-cameraPosition)).unit();
	scale = (focus-cameraPosition).norm()/width;

	Model *model = FileReader::readModel();
	Property::Density *density = nullptr;
	Property::Magnetization *magnetization = nullptr;
	Property::WaveFunction *waveFunction = nullptr;
	if(propertyName.compare("Density") == 0){
		if(inputName.compare("") == 0)
			density = FileReader::readDensity();
		else
			density = FileReader::readDensity(inputName);
	}
	else if(propertyName.compare("Magnetization") == 0){
		if(inputName.compare("") == 0){
			magnetization = FileReader::readMagnetization();
		}
		else{
			magnetization = FileReader::readMagnetization(
				inputName
			);
		}
	}
	else if(propertyName.compare("WaveFunction") == 0){
		if(inputName.compare("") == 0)
			waveFunction = FileReader::readWaveFunction();
		else
			waveFunction = FileReader::readWaveFunction(inputName);
	}
	else{
		TBTKExit(
			"TBTKRayTracePlotter",
			"Property '" << propertyName << "' not supported.",
			"Use '--property PROPERTY_NAME' to set which property to plot."
		);
	}
	Geometry *geometry = model->getGeometry();

	const IndexDescriptor* indexDescriptor;
	if(propertyName.compare("Density") == 0){
		indexDescriptor = &density->getIndexDescriptor();
	}
	else if(propertyName.compare("Magnetization") == 0){
		indexDescriptor = &magnetization->getIndexDescriptor();
	}
	else if(propertyName.compare("WaveFunction") == 0){
		indexDescriptor = &waveFunction->getIndexDescriptor();
	}
	else{
		TBTKExit(
			"TBTKRayTracePlotter",
			"Property '" << propertyName << "' not supported.",
			"Use '--property PROPERTY_NAME' to set which property to plot."
		);
	}
	TBTKAssert(
		indexDescriptor->getFormat() == IndexDescriptor::Format::Custom,
		"",
		"",
		""
	);

	const IndexTree &indexTree = indexDescriptor->getIndexTree();
	IndexTree::Iterator it = indexTree.begin();
	const Index* index;
	vector<Vector3d> coordinates;
	while((index = it.getIndex())){
		Index i = *index;
		for(unsigned int n = 0; n < i.size(); n++)
			if(i.at(n) < 0)
				i.at(n) = IDX_ALL;

		vector<Index> indices = model->getHoppingAmplitudeSet()->getIndexList(i);

		coordinates.push_back(Vector3d({0., 0., 0.}));
		for(unsigned int n = 0; n < indices.size(); n++){
			const double* c = geometry->getCoordinates(indices.at(n));
			coordinates.back().x += c[0]/indices.size();
			coordinates.back().y += c[1]/indices.size();
			coordinates.back().z += c[2]/indices.size();
		}

		it.searchNext();
	}

	Mat canvas = Mat::zeros(height, width, CV_32FC3);
	for(unsigned int y = 0; y < height; y++){
		for(unsigned int x = 0; x < width; x++){
			Vector3d target = focus
				+ (scale*((double)x-width/2))*basisX
				+ (scale*(height/2 - (double)y))*basisY;
			Vector3d direction = target - cameraPosition;
			direction = direction.unit();

			vector<unsigned int> hits;
			for(unsigned int n = 0; n < coordinates.size(); n++)
				if(((coordinates.at(n) - cameraPosition)*direction).norm() < radius)
					hits.push_back(n);

			double valueR = 0.;
			double valueG = 0.;
			double valueB = 0.;
			if(hits.size() > 0){
				double minDistance = (coordinates.at(hits.at(0)) - cameraPosition).norm();
				unsigned int minDistanceIndex = hits.at(0);
				for(unsigned int n = 1; n < hits.size(); n++){
					double distance = (coordinates.at(hits.at(n)) - cameraPosition).norm();
					if(distance < minDistance){
						minDistance = distance;
						minDistanceIndex = hits.at(n);
					}
				}

				if(propertyName.compare("Density") == 0){
					valueR = (*density)(indexTree.getPhysicalIndex(minDistanceIndex));
					valueG = (*density)(indexTree.getPhysicalIndex(minDistanceIndex));
					valueB = (*density)(indexTree.getPhysicalIndex(minDistanceIndex));
				}
				else if(propertyName.compare("Magnetization") == 0){
					//Calculate intersection point between
					//the direction line and the sphere.
					Vector3d objectCoordinate = coordinates.at(minDistanceIndex);
					Vector3d v = objectCoordinate - cameraPosition;
					double a = Vector3d::dotProduct(v, direction);
					double b = Vector3d::dotProduct(v, v);
					double lambda = a - sqrt(radius*radius + a*a - b);
					Vector3d intersection = cameraPosition + lambda*direction;

					Vector3d directionFromObject = (intersection - objectCoordinate).unit();

					if(directionFromObject.z > 0){
						valueR = 255*(5 + directionFromObject.z);
						valueG = 0;
						valueB = 0;
					}
					else{
						valueR = 255*(5 + directionFromObject.z);
						valueG = 255*(5 + directionFromObject.z);
						valueB = 255*(5 + directionFromObject.z);
					}
				}
				else if(propertyName.compare("WaveFunction") == 0){
					complex<double> amplitude = (*waveFunction)(indexTree.getPhysicalIndex(minDistanceIndex), waveFunctionState);
					double absolute = abs(amplitude);
					double argument = arg(amplitude);
					if(argument < 0)
						argument += 2*M_PI;
					valueR = absolute*argument;
					valueG = 0;
					valueB = absolute*(2*M_PI - argument);
				}
				else{
					TBTKExit(
						"TBTKRayTracePlotter",
						"Property '" << propertyName << "' not supported.",
						"Use '--property PROPERTY_NAME' to set which property to plot."
					);
				}
			}

			canvas.at<Vec3f>(y, x)[0] = valueB;
			canvas.at<Vec3f>(y, x)[1] = valueG;
			canvas.at<Vec3f>(y, x)[2] = valueR;
		}
	}

	double minValue = canvas.at<Vec3f>(0, 0)[0];
	double maxValue = canvas.at<Vec3f>(0, 0)[0];
	for(unsigned int x = 0; x < width; x++){
		for(unsigned int y = 0; y < height; y++){
			for(int n = 0; n < 3; n++){
				if(canvas.at<Vec3f>(y, x)[n] < minValue)
					minValue = canvas.at<Vec3f>(y, x)[n];
				if(canvas.at<Vec3f>(y, x)[n] > maxValue)
					maxValue = canvas.at<Vec3f>(y, x)[n];
			}
		}
	}

	Streams::out << "Min:\t" << minValue << "\n";
	Streams::out << "Max:\t" << maxValue << "\n";

	Mat image = Mat::zeros(height, width, CV_8UC3);
	for(unsigned int x = 0; x < width; x++){
		for(unsigned int y = 0; y < height; y++){
			image.at<Vec3b>(y, x)[0] = 255*((canvas.at<Vec3f>(y, x)[0] - minValue)/(maxValue - minValue));
			image.at<Vec3b>(y, x)[1] = 255*((canvas.at<Vec3f>(y, x)[1] - minValue)/(maxValue - minValue));
			image.at<Vec3b>(y, x)[2] = 255*((canvas.at<Vec3f>(y, x)[2] - minValue)/(maxValue - minValue));
		}
	}

	if(outputName.compare("") != 0){
		stringstream ss;
		ss << "figures/" << outputName;
		imwrite(ss.str(), image);
	}
	else if(propertyName.compare("Density") == 0){
		imwrite("figures/Density.png", image);
	}
	else if(propertyName.compare("Magnetization") == 0){
		imwrite("figures/Magnetization.png", image);
	}
	else if(propertyName.compare("WaveFunction") == 0){
		imwrite("figures/WaveFunction.png", image);
	}
	else{
		TBTKExit(
			"TBTKRayTracePlotter",
			"Property '" << propertyName << "' not supported.",
			"Use '--property PROPERTY_NAME' to set which property to plot."
		);
	}

	return 0;
}

Vector3d getCoordinate(string coordinateString){
	coordinateString.erase(
		remove_if(
			coordinateString.begin(),
			coordinateString.end(),
			[](char c){
				switch(c){
				case ' ':
				case '\t':
					return true;
				default:
					return false;
				}
			}
		),
		coordinateString.end()
	);
	stringstream ss(coordinateString);
	double x, y, z;
	char c;
	ss >> c;
	TBTKAssert(
		c == '(',
		"TBTKRayTracePlotter",
		"Expected '(' but found '" << c << "'.",
		"Input coordinate on format (x, y, z)"
	);
	ss >> x;
	ss >> c;
	TBTKAssert(
		c == ',',
		"TBTKRayTracePlotter",
		"Expected ',' but found '" << c << "'.",
		"Input coordinate on format (x, y, z)"
	);
	ss >> y;
	ss >> c;
	TBTKAssert(
		c == ',',
		"TBTKRayTracePlotter",
		"Expected ',' but found '" << c << "'.",
		"Input coordinate on format (x, y, z)"
	);
	ss >> z;
	ss >> c;
	TBTKAssert(
		c == ')',
		"TBTKRayTracePlotter",
		"Expected ')' but found '" << c << "'.",
		"Input coordinate on format (x, y, z)"
	);

	return Vector3d({x, y, z});
}
