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
 *  @file BravaisLattice.h
 *  @brief Base class for Bravais lattices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BRAVAIS_LATTICE
#define COM_DAFER45_TBTK_BRAVAIS_LATTICE

#include <vector>

namespace TBTK{
namespace Lattice{

class BravaisLattice{
public:
	/** Constructor. */
	BravaisLattice();

	/** Destructor. */
	~BravaisLattice();

	/** Get number of lattice vectors. */
	inline int getNumLatticeVectors() const;

	/** Get number of additional sites. */
	inline int getNumAdditionalSites() const;

	/** Get lattice vector. */
	const std::vector<double>& getLatticeVector(int n) const;

	/** Get additional site. */
	const std::vector<double>& getAdditionalSite(int n) const;

	/** Get lattice vectors. */
	const std::vector<std::vector<double>>& getLatticeVectors() const;

	/** Get additional sites. */
	const std::vector<std::vector<double>>& getAdditionalSites() const;

	/** Converts the basis vectors to ensure that they span primitive
	 *  cells. */
	virtual void makePrimitive();
protected:
	/** Set lattice vectors. */
	void setLatticeVectors(const std::vector<std::vector<double>> &latticeVectors);

	/** Set additional sites. */
	void setAdditionalSites(const std::vector<std::vector<double>> &additionalSites);
private:
	/** Lattice vectors. */
	std::vector<std::vector<double>> latticeVectors;

	/** Positions of additional sites within the unit cell. */
	std::vector<std::vector<double>> additionalSites;
};

inline int BravaisLattice::getNumLatticeVectors() const{
	return latticeVectors.size();
}

inline int BravaisLattice::getNumAdditionalSites() const{
	return additionalSites.size();
}

inline const std::vector<double>& BravaisLattice::getLatticeVector(int n) const{
	return latticeVectors.at(n);
}

inline const std::vector<double>& BravaisLattice::getAdditionalSite(int n) const{
	return additionalSites.at(n);
}

inline const std::vector<std::vector<double>>& BravaisLattice::getLatticeVectors() const{
	return latticeVectors;
}

inline const std::vector<std::vector<double>>& BravaisLattice::getAdditionalSites() const{
	return additionalSites;
}

};	//End of namespace Lattice
};	//End of namespace TBTK

#endif
