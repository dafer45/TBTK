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

/** @package TBTKcalc
 *  @file Geometry.h
 *  @brief Contains geometric inforamtion about a model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GEOMETRY
#define COM_DAFER45_TBTK_GEOMETRY

#include "Model.h"
#include <initializer_list>

namespace TBTK{
	class FileReader;

class Geometry{
public:
	/** Constructor. */
	Geometry(int dimensions, int numSpecifiers, Model *parentModel);

	/** Destructor. */
	~Geometry();

	/** Add a coordinate (and specifiers) for an index. */
	void setCoordinates(
		const Index &index,
		std::initializer_list<double> coordinates,
		std::initializer_list<int> specifiers = {}
	);

	/** Add a coordinate (and specifiers) for an index. */
	void setCoordinates(
		const Index &index,
		const std::vector<double> &coordinates,
		const std::vector<int> &specifiers = {}
	);

	/** Add a coordinate (and specifiers) for a Hilbert space index. */
	void setCoordinates(
		int basisIndex,
		std::initializer_list<double> coordinates,
		std::initializer_list<int> specifiers = {}
	);

	/** Add a coordinate (and specifiers) for a Hilbert space index. */
	void setCoordinates(
		int basisIndex,
		const std::vector<double> &coordinates,
		const std::vector<int> &specifiers = {}
	);

	/** Get dimensions. */
	int getDimensions() const;

	/** Get number of specifiers. */
	int getNumSpecifiers() const;

	/** Get Coordinates using a physical index. */
	const double* getCoordinates(const Index &index) const;

	/** Get coordinates using a Hilbert space index. */
	const double* getCoordinates(int index) const;

	/** Get all coordinates. */
	const double* getCoordinates() const;

	/** Translate all coordinates. */
	void translate(std::initializer_list<double> translation);

	/** Get specifier using a physical index. */
	int getSpecifier(const Index &index, int specifier) const;

	/** Get specifier using a Hilbert space index. */
	int getSpecifier(int index, int specifier) const;

	/** Get all specifiers for given physical index. */
	const int* getSpecifiers(const Index &index) const;

	/** Get all speciferis for given Hilbert space index. */
	const int* getSpecifiers(int index) const;

	/** Get all specifiers. */
	const int* getSpecifiers() const;

	/** Get distance between positions corresponding to index1 and index2,
	 *  using physical indices. */
	double getDistance(const Index &index1, const Index &index2) const;

	/** Get distance between positions corresponding to index1 and index2,
	 *  using physical indices. */
	double getDistance(int index1, int index2) const;

	/** Get size of corresponding Hilbert space. */
	int getBasisSize() const;
private:
	/** Number of spatial dimensions. */
	unsigned int dimensions;

	/** Number of specifiers. */
	unsigned int numSpecifiers;

	/** Spatial dimensions. */
	double *coordinates;

	/** Specifiers such as orbital number, spin-species, etc. */
	int *specifiers;

	/** The Model that the geometry corresponds to. */
	Model *parentModel;

	/** FileReader is a friend class to allow it to write Geometry data. */
	friend class FileReader;
};

inline int Geometry::getDimensions() const{
	return dimensions;
}

inline int Geometry::getNumSpecifiers() const{
	return numSpecifiers;
}

inline const double* Geometry::getCoordinates(const Index &index) const{
	return &(coordinates[dimensions*parentModel->getBasisIndex(index)]);
}

inline const double* Geometry::getCoordinates(int index) const{
	return &(coordinates[dimensions*index]);
}

inline const double* Geometry::getCoordinates() const{
	return coordinates;
}

inline int Geometry::getSpecifier(const Index &index, int specifier) const{
	return specifiers[numSpecifiers*parentModel->getBasisIndex(index) + specifier];
}

inline int Geometry::getSpecifier(int index, int specifier) const{
	return specifiers[numSpecifiers*index + specifier];
}

inline const int* Geometry::getSpecifiers(const Index& index) const{
	return &(specifiers[numSpecifiers*parentModel->getBasisIndex(index)]);
}

inline const int* Geometry::getSpecifiers(const int index) const{
	return &(specifiers[numSpecifiers*index]);
}

inline const int* Geometry::getSpecifiers() const{
	return specifiers;
}

inline double Geometry::getDistance(int index1, int index2) const{
	double distanceSquared = 0.;
	for(unsigned int n = 0; n < dimensions; n++){
		double difference = coordinates[index1] - coordinates[index2];
		distanceSquared += difference*difference;
	}

	return sqrt(distanceSquared);
}

inline int Geometry::getBasisSize() const{
	return parentModel->getBasisSize();
}

};	//End of namespace TBTK

#endif
