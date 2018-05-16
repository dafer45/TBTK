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

/** @Geometry.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Geometry.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

Geometry::Geometry(
	int dimensions,
	int numSpecifiers,
	const HoppingAmplitudeSet *hoppingAmplitudeSet
){
	this->dimensions = dimensions;
	this->numSpecifiers = numSpecifiers;
	this->hoppingAmplitudeSet = hoppingAmplitudeSet;

	coordinates = new double[dimensions*hoppingAmplitudeSet->getBasisSize()];
	if(numSpecifiers != 0)
		specifiers = new int[numSpecifiers*hoppingAmplitudeSet->getBasisSize()];
	else
		specifiers = nullptr;
}

Geometry::Geometry(const Geometry &geometry){
	dimensions = geometry.dimensions;
	numSpecifiers = geometry.numSpecifiers;
	hoppingAmplitudeSet = geometry.hoppingAmplitudeSet;

	coordinates = new double[dimensions*hoppingAmplitudeSet->getBasisSize()];
	for(
		unsigned int n = 0;
		n < dimensions*hoppingAmplitudeSet->getBasisSize();
		n++
	){
		coordinates[n] = geometry.coordinates[n];
	}

	if(numSpecifiers > 0){
		specifiers = new int[dimensions*hoppingAmplitudeSet->getBasisSize()];
		for(
			unsigned int n = 0;
			n < dimensions*hoppingAmplitudeSet->getBasisSize();
			n++
		){
			specifiers[n] = geometry.specifiers[n];
		}
	}
	else{
		specifiers = nullptr;
	}
}

Geometry::Geometry(Geometry &&geometry){
	dimensions = geometry.dimensions;
	numSpecifiers = geometry.numSpecifiers;
	hoppingAmplitudeSet = geometry.hoppingAmplitudeSet;

	coordinates = geometry.coordinates;
	geometry.coordinates = nullptr;

	specifiers = geometry.specifiers;
	geometry.specifiers = nullptr;
}

Geometry::Geometry(
	const string &serialization,
	Mode mode,
	const HoppingAmplitudeSet &hoppingAmplitudeSet
){
	this->hoppingAmplitudeSet = &hoppingAmplitudeSet;

	TBTKAssert(
		validate(serialization, "Geometry", mode),
		"Geometry::Geometry()",
		"Unable to parse string as Geometry '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::Debug:
	{
		string content = getContent(serialization, mode);

		vector<string> elements = split(content, mode);

		stringstream ss;
		ss.str(elements.at(0));
		ss >> dimensions;
		ss.clear();
		ss.str(elements.at(1));
		ss >> numSpecifiers;
		unsigned int counter = 2;
		coordinates = new double[dimensions*hoppingAmplitudeSet.getBasisSize()];
		for(
			unsigned int n = 0;
			n < dimensions*hoppingAmplitudeSet.getBasisSize();
			n++
		){
			ss.clear();
			ss.str(elements.at(counter));
			ss >> coordinates[n];
			counter++;
		}
		if(numSpecifiers > 0){
			specifiers = new int[numSpecifiers*hoppingAmplitudeSet.getBasisSize()];
			for(
				unsigned int n = 0;
				n < numSpecifiers*hoppingAmplitudeSet.getBasisSize();
				n++
			){
				ss.clear();
				ss.str(elements.at(counter));
				ss >> specifiers[n];
				counter++;
			}
		}
		else{
			specifiers = nullptr;
		}

		break;
	}
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			dimensions = j.at("dimensions").get<int>();
			numSpecifiers = j.at("numSpecifiers").get<int>();

			nlohmann::json c = j.at("coordinates");
			TBTKAssert(
				distance(
					c.begin(),
					c.end()
				) == dimensions*hoppingAmplitudeSet.getBasisSize(),
				"Geometry::Geometry()",
				"Incompatible array sizes. "
				<< "'dimensions*hoppingAmplitudeSet.getBasisSize()'"
				<< " is "
				<< dimensions*hoppingAmplitudeSet.getBasisSize()
				<< " but coordinates has "
				<< distance(c.begin(), c.end())
				<< " elements.",
				""
			);
			coordinates = new double[dimensions*hoppingAmplitudeSet.getBasisSize()];
			unsigned int counter = 0;
			for(nlohmann::json::iterator it = c.begin(); it < c.end(); ++it){
				coordinates[counter] = *it;
				counter++;
			}

			if(numSpecifiers > 0){
				nlohmann::json s = j.at("specifiers");
				TBTKAssert(
					distance(
						s.begin(),
						s.end()
					) == numSpecifiers*hoppingAmplitudeSet.getBasisSize(),
					"Geometry::Geometry()",
					"Incompatible array sizes. "
					<< "'numSpecifiers*hoppingAmplitudeSet.getBasisSize()'"
					<< " is "
					<< numSpecifiers*hoppingAmplitudeSet.getBasisSize()
					<< " but specifiers has "
					<< distance(s.begin(), s.end())
					<< " elements.",
					""
				);
				specifiers = new int[numSpecifiers*hoppingAmplitudeSet.getBasisSize()];
				unsigned int counter = 0;
				for(nlohmann::json::iterator it = s.begin(); it < s.end(); ++it){
					specifiers[counter] = *it;
					counter++;
				}
			}
			else{
				specifiers = nullptr;
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"Geometry::Geometry()",
				"Unable to parse string as Geometry '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Geometry::Geometry()",
			"Only Serializable::Mode:Debug is supported yet.",
			""
		);
	}
}

Geometry::~Geometry(){
	if(coordinates != nullptr)
		delete [] coordinates;
	if(specifiers != nullptr)
		delete [] specifiers;
}

Geometry& Geometry::operator=(const Geometry &rhs){
	if(this != &rhs){
		dimensions = rhs.dimensions;
		numSpecifiers = rhs.numSpecifiers;
		hoppingAmplitudeSet = rhs.hoppingAmplitudeSet;

		coordinates = new double[dimensions*hoppingAmplitudeSet->getBasisSize()];
		for(
			unsigned int n = 0;
			n < dimensions*hoppingAmplitudeSet->getBasisSize();
			n++
		){
			coordinates[n] = rhs.coordinates[n];
		}

		if(numSpecifiers > 0){
			specifiers = new int[dimensions*hoppingAmplitudeSet->getBasisSize()];
			for(
				unsigned int n = 0;
				n < dimensions*hoppingAmplitudeSet->getBasisSize();
				n++
			){
				specifiers[n] = rhs.specifiers[n];
			}
		}
		else{
			specifiers = nullptr;
		}
	}

	return *this;
}

Geometry& Geometry::operator=(Geometry &&rhs){
	if(this != &rhs){
		dimensions = rhs.dimensions;
		numSpecifiers = rhs.numSpecifiers;
		hoppingAmplitudeSet = rhs.hoppingAmplitudeSet;

		coordinates = rhs.coordinates;
		rhs.coordinates = nullptr;

		specifiers = rhs.specifiers;
		rhs.specifiers = nullptr;
	}

	return *this;
}

void Geometry::setCoordinates(
	const Index &index,
	std::initializer_list<double> coordinates,
	std::initializer_list<int> specifiers
){
	int basisIndex = hoppingAmplitudeSet->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	const Index &index,
	const std::vector<double> &coordinates,
	const std::vector<int> &specifiers
){
	int basisIndex = hoppingAmplitudeSet->getBasisIndex(index);
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	int basisIndex,
	std::initializer_list<double> coordinates,
	std::initializer_list<int> specifiers
){
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::setCoordinates(
	int basisIndex,
	const std::vector<double> &coordinates,
	const std::vector<int> &specifiers
){
	if(coordinates.size() == (unsigned int)dimensions){
		for(unsigned int n = 0; n < dimensions; n++)
			this->coordinates[dimensions*basisIndex + n] = *(coordinates.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::setCoordinates()",
			"Geometry requires " << dimensions << " coordinates, but " << coordinates.size() << " were supplied.",
			""
		);
	}

	if(specifiers.size() == (unsigned int)numSpecifiers){
		for(unsigned int n = 0; n < numSpecifiers; n++)
			this->specifiers[numSpecifiers*basisIndex + n] = *(specifiers.begin() + n);
	}
	else{
		TBTKExit(
			"Geometry::addPoint()",
			"Geometry requires " << numSpecifiers << " specfiers, but " << specifiers.size() << " were supplied.",
			""
		);
	}
}

void Geometry::translate(initializer_list<double> translation){
	if(translation.size() != dimensions){
		TBTKExit(
			"Geometry::translate()",
			"The number of dimensions of the translation vector (" << translation.size() << ") does not match the dimension of the geometry (" << dimensions << ").",
			""
		);
	}

	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		for(unsigned int c = 0; c < dimensions; c++){
			coordinates[n*dimensions + c] += *(translation.begin() + c);
		}
	}
}

double Geometry::getDistance(const Index &index1, const Index &index2) const{
	int basisIndex1 = hoppingAmplitudeSet->getBasisIndex(index1);
	int basisIndex2 = hoppingAmplitudeSet->getBasisIndex(index2);

	double distanceSquared = 0.;
	for(unsigned int n = 0; n < dimensions; n++){
		double difference = coordinates[dimensions*basisIndex1 + n] - coordinates[dimensions*basisIndex2 + n];
		distanceSquared += difference*difference;
	}

	return sqrt(distanceSquared);
}

string Geometry::serialize(Mode mode) const{
	switch(mode){
	case Mode::Debug:
	{
		stringstream ss;
		ss << "Geometry(";
		ss << Serializable::serialize(dimensions, mode);
		ss << "," << Serializable::serialize(numSpecifiers, mode);
		for(
			unsigned int n = 0;
			n < dimensions*hoppingAmplitudeSet->getBasisSize();
			n++
		){
			ss << "," << Serializable::serialize(
				coordinates[n],
				mode
			);
		}
		for(
			unsigned int n = 0;
			n < numSpecifiers*hoppingAmplitudeSet->getBasisSize();
			n++
		){
			ss << "," << Serializable::serialize(
				specifiers[n],
				mode
			);
		}
		ss << ")";

		return ss.str();
	}
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Geometry";
		j["dimensions"] = dimensions;
		j["numSpecifiers"] = numSpecifiers;
		for(
			unsigned int n = 0;
			n < dimensions*hoppingAmplitudeSet->getBasisSize();
			n++
		){
			j["coordinates"].push_back(coordinates[n]);
		}
		if(numSpecifiers > 0){
			for(
				unsigned int n = 0;
				n < numSpecifiers*hoppingAmplitudeSet->getBasisSize();
				n++
			){
				j["specifiers"].push_back(specifiers[n]);
			}
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"Geometry::Geometry()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
