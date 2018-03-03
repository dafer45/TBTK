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

/** @file BandDiagramGenerator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/BandDiagramGenerator.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/ParametrizedLine.h"
#include "TBTK/Timer.h"
#include "TBTK/VectorNd.h"

using namespace std;

namespace TBTK{

BandDiagramGenerator::BandDiagramGenerator(){
	reciprocalLattice = nullptr;
}

vector<vector<double>> BandDiagramGenerator::generateBandDiagram(
	initializer_list<initializer_list<double>> kPoints,
	unsigned int resolution,
	initializer_list<initializer_list<double>> nestingVectors
) const{
	TBTKAssert(
		reciprocalLattice != nullptr,
		"BandDiagramGenerator::BandDiagramGenerator()",
		"ReciprocalLattice not set.",
		"Use BandDiagramGenerator::setReciprocalLattice() to set"
		<< " ReciprocalLattice."
	);

	unsigned int numDimensions = kPoints.begin()->size();
	for(unsigned int n = 1; n < kPoints.size(); n++){
		TBTKAssert(
			(kPoints.begin() + n)->size() == numDimensions,
			"BandDiagramGenerator::BandDiagramGenerator()",
			"Incompatible k-point dimensions. First k-point has "
			<< numDimensions << " dimensions, but kPoint " << n
			<< " has " << (kPoints.begin() + n)->size()
			<< " dimensions.",
			""
		);
	}
	for(unsigned int n = 0; n < nestingVectors.size(); n++){
		TBTKAssert(
			(nestingVectors.begin() + n)->size() == numDimensions,
			"BandDiagramGenerator::BandDiagramGenerator()",
			"Incompatible nesting vector dimension. The k-points"
			<< " has " << numDimensions << " dimensions, but"
			<< " nestingVector " << n << " has "
			<< (nestingVectors.begin() + n)->size()
			<< " dimensions.",
			""
		);
	}
	vector<vector<double>> nesting;
	nesting.push_back(vector<double>());
	for(unsigned int n = 0; n < numDimensions; n++)
		nesting.at(0).push_back(0);
	for(unsigned int n = 0; n < nestingVectors.size(); n++){
		nesting.push_back(vector<double>());
		for(unsigned int c = 0; c < numDimensions; c++)
			nesting.back().push_back(*((nestingVectors.begin() + n)->begin() + c));
	}

	unsigned int numBands = reciprocalLattice->getNumBands();

	vector<vector<double>> bandDiagram;
	for(unsigned int n = 0; n < numBands*nesting.size(); n++)
		bandDiagram.push_back(vector<double>());
	for(unsigned int n = 1; n < kPoints.size(); n++){
		const initializer_list<double> kPointStart = *(kPoints.begin() + n - 1);
		const initializer_list<double> kPointEnd = *(kPoints.begin() + n);
		vector<double> start;
		vector<double> direction;
		for(unsigned int c = 0; c < numDimensions; c++){
			double s = *(kPointStart.begin() + c);
			double e = *(kPointEnd.begin() + c);
			start.push_back(s);
			direction.push_back(e - s);
		}
		ParametrizedLine kLine(
			start,
			direction
		);

		for(unsigned int c = 0; c < resolution; c++){
			vector<double> latticePoint = kLine(c/(double)resolution);
			for(unsigned int m = 0; m < nesting.size(); m++){
				vector<double> nestedPoint;
				for(unsigned int i = 0; i < numDimensions; i++){
					nestedPoint.push_back(
						latticePoint.at(i) + nesting.at(m).at(i)
					);
				}
				Timer::tick("Diagonalizing");
				Model *model = reciprocalLattice->generateModel(
					nestedPoint
				);
				model->construct();

				Solver::Diagonalizer solver;
				solver.setModel(*model);
				solver.run();

				for(unsigned int i = 0; i < numBands; i++)
					bandDiagram[i + numBands*m].push_back(solver.getEigenValue(i));

				delete model;
				Timer::tock();
			}
		}
	}

	return bandDiagram;
}

};	//End of namespace TBTK
