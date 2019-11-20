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

/** @file ParallelepipedArrayState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ParallelepipedArrayState.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

ParallelepipedArrayState::ParallelepipedArrayState(
	const vector<vector<double>> &basisVectors,
	initializer_list<unsigned int> numMeshPoints
) :
	ArrayState(numMeshPoints),
	Field(true),
	parallelepiped(basisVectors, SpacePartition::MeshType::Nodal)
{
}

ParallelepipedArrayState::~ParallelepipedArrayState(){
}

ParallelepipedArrayState* ParallelepipedArrayState::clone() const{
	ParallelepipedArrayState* clonedState = new ParallelepipedArrayState(*this);

	return clonedState;
}

complex<double> ParallelepipedArrayState::getOverlap(const AbstractState &bra) const{
	TBTKNotYetImplemented("ArrayState::getOverlap()");
}

complex<double> ParallelepipedArrayState::getMatrixElement(
	const AbstractState &bra,
	const AbstractOperator &o
) const{
	TBTKNotYetImplemented("ArrayState::getOverlap()");
}

void ParallelepipedArrayState::setAmplitude(
	std::complex<double> amplitude,
	std::initializer_list<double> coordinate
){
	const std::vector<unsigned int> &resolution = getResolution();
	Index index = parallelepiped.getMinorCellIndex(coordinate, resolution);
	for(unsigned int n = 0; n < resolution.size(); n++)
		index.at(n) += resolution.at(n)/2;
	for(unsigned int n = 0; n < resolution.size(); n++){
		if(index.at(n) < 0 || index.at(n) >= (int)resolution.at(n)){
			std::stringstream ss;
			ss << "(";
			for(unsigned int n = 0; n < coordinate.size(); n++){
				if(n != 0)
					ss << ", ";
				ss << *(coordinate.begin() + n);
			}
			ss << ")";
			TBTKExit(
				"ParallelepipedArrayState::setAmplitude()",
				"Coordinate " << ss.str() << " out of bound.",
				"" << index.at(0) << "\t" << index.at(1) << "\t" << index.at(2)
			);
		}
	}

	ArrayState::setAmplitude(amplitude, index);
}

const complex<double>& ParallelepipedArrayState::getAmplitude(
	std::initializer_list<double> coordinate
) const{
	static constexpr complex<double> zero = 0;

	const vector<double> &localCenter = getCoordinates();
	TBTKAssert(
		localCenter.size() == coordinate.size(),
		"ParallelepipedArrayState::operator()",
		"Coordinate dimension does not agree with the dimension of the"
		<< " coordinate of the state. 'coordinate' has "
		<< coordinate.size() << " components, but the states"
		<< " coordinate has " << localCenter.size() << " components.",
		""
	);
	vector<double> c;
	c.reserve(coordinate.size());	//Optimization
	for(unsigned int n = 0; n < coordinate.size(); n++)
		c.push_back(*(coordinate.begin() + n) - localCenter.at(n));

	const std::vector<unsigned int> &resolution = getResolution();
	Index index = parallelepiped.getMinorCellIndex(c, resolution);
	for(unsigned int n = 0; n < resolution.size(); n++)
		index.at(n) += resolution.at(n)/2;
	for(unsigned int n = 0; n < resolution.size(); n++){
		if(index.at(n) < 0 || index.at(n) >= (int)resolution.at(n))
			return zero;
	}

	return ArrayState::getAmplitude(index);
}

const complex<double>& ParallelepipedArrayState::getAmplitude(
	const std::vector<double> &coordinate
) const{
	static constexpr complex<double> zero = 0;

	const vector<double> &localCenter = ArrayState::getCoordinates();
	TBTKAssert(
		localCenter.size() == coordinate.size(),
		"ParallelepipedArrayState::operator()",
		"Coordinate dimension does not agree with the dimension of the"
		<< " coordinate for the state. 'coordinate' has "
		<< coordinate.size() << " components, but the states"
		<< " coordinate has " << localCenter.size() << " components.",
		""
	);

	vector<double> c;
	c.reserve(coordinate.size());	//Optimization
	for(unsigned int n = 0; n < coordinate.size(); n++)
		c.push_back(coordinate.at(n) - localCenter.at(n));

	const std::vector<unsigned int> &resolution = getResolution();
	Index index = parallelepiped.getMinorCellIndex(c, resolution);
	for(unsigned int n = 0; n < resolution.size(); n++)
		index.at(n) += resolution.at(n)/2;
	for(unsigned int n = 0; n < resolution.size(); n++){
		if(index.at(n) < 0 || index.at(n) >= (int)resolution.at(n))
			return zero;
	}

	return ArrayState::getAmplitude(index);
}

complex<double> ParallelepipedArrayState::operator()(
	initializer_list<double> argument
) const{
	return ParallelepipedArrayState::getAmplitude(argument);
}

const vector<double>& ParallelepipedArrayState::getCoordinates() const{
	return ArrayState::getCoordinates();
}

double ParallelepipedArrayState::getExtent() const{
	return ArrayState::getExtent();
}

};	//End of namespace TBTK
