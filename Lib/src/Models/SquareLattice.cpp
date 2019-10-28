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
	IndexFilter(unsigned int sizeX, unsigned int sizeY){
		this->sizeX = sizeX;
		this->sizeY = sizeY;
	}

	virtual IndexFilter* clone() const{
		return new IndexFilter(sizeX, sizeY);
	}

	virtual bool isIncluded(const Index &index) const{
		unsigned int x = index[0];
		unsigned int y = index[1];
		if(x >= 0 && x < sizeX && y >= 0 && y < sizeY)
			return true;
		else
			return false;
	}
private:
	unsigned int sizeX;
	unsigned int sizeY;
};

SquareLattice::SquareLattice(
	const vector<unsigned int> &size,
	const vector<complex<double>> &parameters,
	bool includeSpinSubindex
){
	TBTKAssert(
		size.size() == 2,
		"Models::SquareLattice::SquareLattice()",
		"'size' must have exactly '2' components, but '" << size.size()
		<< "' found.",
		""
	);
	TBTKAssert(
		parameters.size() > 1 && parameters.size() < 4,
		"Models::SquareLattice::SquareLattice()",
		"'parameters' must have '2' or '3' components, but '"
		<< size.size() << "' components found.",
		""
	);
	setFilter(IndexFilter(size[0], size[1]));
	if(includeSpinSubindex){
		for(unsigned int x = 0; x < size[0]; x++){
			for(unsigned int y = 0; y < size[1]; y++){
				for(unsigned int spin = 0; spin < 2; spin++){
					*this << HoppingAmplitude(
						parameters[0],
						{x, y, spin},
						{x, y, spin}
					);
					*this << HoppingAmplitude(
						parameters[1],
						{x+1, y, spin},
						{x, y, spin}
					) + HC;
					*this << HoppingAmplitude(
						parameters[1],
						{x, y+1, spin},
						{x, y, spin}
					) + HC;

					if(parameters.size() > 2){
						*this << HoppingAmplitude(
							parameters[2],
							{x+1, y+1, spin},
							{x, y, spin}
						) + HC;
						*this << HoppingAmplitude(
							parameters[2],
							{x, ((int)y)-1, spin},
							{x, y, spin}
						) + HC;
					}
				}
			}
		}
	}
	else{
		for(unsigned int x = 0; x < size[0]; x++){
			for(unsigned int y = 0; y < size[1]; y++){
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
						{x, ((int)y)-1},
						{x, y}
					) + HC;
				}
			}
		}
	}
}

}
};
