/* Copyright 2019 Kristofer Björnson
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

/** @file SquareLattice.cpp
 *  @author Kristofer Björnson
 */

#include "TBTK/AbstractIndexFilter.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Models{

class IndexFilter : public AbstractIndexFilter{
public:
	IndexFilter(
		unsigned int sizeX,
		unsigned int sizeY,
		int spinIndex = -1
	){
		this->sizeX = sizeX;
		this->sizeY = sizeY;
		this->spinIndex = spinIndex;
	}

	virtual IndexFilter* clone() const{
		return new IndexFilter(sizeX, sizeY, spinIndex);
	}

	virtual bool isIncluded(const Index &index) const{
		Index modifiedIndex = index;
		if(spinIndex != -1)
			modifiedIndex.erase(spinIndex);
		Subindex x = modifiedIndex[0];
		Subindex y = modifiedIndex[1];
		if(
			x >= 0 && x < (Subindex)sizeX
			&& y >= 0 && y < (Subindex)sizeY
		){
			return true;
		}
		else{
			return false;
		}
	}
private:
	unsigned int sizeX;
	unsigned int sizeY;
	int spinIndex;
};

SquareLattice::SquareLattice(
	const Index &size,
	const vector<complex<double>> &parameters
){
	TBTKAssert(
		parameters.size() > 1 && parameters.size() < 4,
		"Models::SquareLattice::SquareLattice()",
		"'parameters' must have '2' or '3' components, but '"
		<< parameters.size() << "' components found.",
		""
	);

	bool hasSpinIndex = false;
	for(unsigned int n = 0; n < size.getSize(); n++){
		if(size[n].isSpinIndex()){
			TBTKAssert(
				!hasSpinIndex,
				"Models::SquareLattice::SquareLattice()",
				"Multiple spin subindices detected in 'size'.",
				"At most one IDX_SPIN flag is allowed."
			);

			hasSpinIndex = true;
		}
	}

	if(hasSpinIndex)
		createModelWithSpin(size, parameters);
	else
		createModelWithoutSpin(size, parameters);
}

void SquareLattice::createModelWithSpin(
	const Index &size,
	const vector<complex<double>> &parameters
){
	TBTKAssert(
		size.getSize() == 3,
		"Models::SquareLattice::SquareLattice()",
		"'size' must have exactly '3' components when using IDX_SPIN"
		<< " flag, but '" << size.getSize() << "' found.",
		""
	);

	unsigned int spinPosition;
	for(unsigned int n = 0; n < size.getSize(); n++)
		if(size[n].isSpinIndex())
			spinPosition = n;

	Index modifiedSize = size;
	modifiedSize.erase(spinPosition);

	IndexFilter indexFilter(size[0], size[1]);
	setFilter(IndexFilter(size[0], size[1]));
	for(Subindex x = 0; x < size[0]; x++){
		for(Subindex y = 0; y < size[1]; y++){
			for(Subindex spin = 0; spin < 2; spin++){
				Index site = {x, y};
				Index sitePlusX = {x+1, y};
				Index sitePlusY = {x, y+1};
				Index sitePlusXPlusY = {x+1, y+1};
				Index sitePlusXMinusY = {x+1, ((int)y)-1};
				site.insert(spinPosition, spin);
				sitePlusX.insert(spinPosition, spin);
				sitePlusY.insert(spinPosition, spin);
				sitePlusXPlusY.insert(spinPosition, spin);
				sitePlusXMinusY.insert(spinPosition, spin);

				*this << HoppingAmplitude(
					parameters[0],
					site,
					site
				);
				*this << HoppingAmplitude(
					parameters[1],
					sitePlusX,
					site
				) + HC;
				*this << HoppingAmplitude(
					parameters[1],
					sitePlusY,
					site
				) + HC;

				if(parameters.size() > 2){
					*this << HoppingAmplitude(
						parameters[2],
						sitePlusXPlusY,
						site
					) + HC;
					*this << HoppingAmplitude(
						parameters[2],
						sitePlusXMinusY,
						site
					) + HC;
				}
			}
		}
	}
}

void SquareLattice::createModelWithoutSpin(
	const Index &size,
	const vector<complex<double>> &parameters
){
	TBTKAssert(
		size.getSize() == 2,
		"Models::SquareLattice::SquareLattice()",
		"'size' must have exactly '2' components when not using"
		<< " IDX_SPIN flag, but '" << size.getSize() << "' found.",
		""
	);
	setFilter(IndexFilter(size[0], size[1]));
	for(Subindex x = 0; x < size[0]; x++){
		for(Subindex y = 0; y < size[1]; y++){
			*this << HoppingAmplitude(
				parameters[0],
				{x, y},
				{x, y}
			);
			*this << HoppingAmplitude(
				parameters[1],
				{x+1, y},
				{x, y}
			) + HC;
			*this << HoppingAmplitude(
				parameters[1],
				{x, y+1},
				{x, y}
			) + HC;

			if(parameters.size() > 2){
				*this << HoppingAmplitude(
					parameters[2],
					{x+1, y+1},
					{x, y}
				) + HC;
				*this << HoppingAmplitude(
					parameters[2],
					{x+1, ((int)y)-1},
					{x, y}
				) + HC;
			}
		}
	}
}

};	//End of namespace Models
};	//End of namespace TBTK
