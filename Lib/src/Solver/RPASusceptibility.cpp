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

/** @file RPASusceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/Solver/RPASusceptibility.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

RPASusceptibility::RPASusceptibility(
	const MomentumSpaceContext &momentumSpaceContext,
	const Property::Susceptibility &bareSusceptibility
) :
	Communicator(true),
	bareSusceptibility(bareSusceptibility),
	momentumSpaceContext(momentumSpaceContext)
{
}

RPASusceptibility* RPASusceptibility::createSlave(){
	TBTKExit(
		"Solver::RPASusceptibility::createSlave()",
		"This function is not supported by this solver.",
		""
	);
}

extern "C" {
	void zgetrf_(
		int* M,
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		int *info
	);
	void zgetri_(
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		complex<double> *work,
		int *lwork,
		int *info
	);
}

inline void RPASusceptibility::invertMatrix(
	complex<double> *matrix,
	unsigned int dimensions
){
	int numRows = dimensions;
	int numCols = dimensions;

	int *ipiv = new int[min(numRows, numCols)];
	int lwork = numCols*numCols;
	complex<double> *work = new complex<double>[lwork];
	int info;

	zgetrf_(&numRows, &numCols, matrix, &numRows, ipiv, &info);
	zgetri_(&numRows, matrix, &numRows, ipiv, work, &lwork, &info);

	delete [] ipiv;
	delete [] work;
}

vector<vector<vector<complex<double>>>> RPASusceptibility::rpaSusceptibilityMainAlgorithm(
	const Index &index,
	const vector<InteractionAmplitude> &interactionAmplitudes
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	IndexTree intraBlockIndexTree
		= getModel().getHoppingAmplitudeSet().getIndexTree(kIndex);
	vector<Index> intraBlockIndexList;
	for(
		IndexTree::ConstIterator iterator
			= intraBlockIndexTree.cbegin();
		iterator != intraBlockIndexTree.cend();
		++iterator
	){
		Index index = *iterator;
		for(unsigned int n = 0; n < kIndex.getSize(); n++)
			index.popFront();

		intraBlockIndexList.push_back(index);
	}
	unsigned int matrixDimension = pow(intraBlockIndexList.size(), 2);

	//Setup energies.
	vector<complex<double>> energies;
	switch(bareSusceptibility.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real:
		for(
			unsigned int n = 0;
			n < bareSusceptibility.getResolution();
			n++
		){
			energies.push_back(bareSusceptibility.getEnergy(n));
		}
		break;
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::BosonicMatsubara:
		for(
			unsigned int n = 0;
			n < bareSusceptibility.getNumMatsubaraEnergies();
			n++
		){
			energies.push_back(
				bareSusceptibility.getMatsubaraEnergy(n)
			);
		}
		break;
	default:
		TBTKExit(
			"Solver::RPASusceptibility::calculateSusceptibilityMainAlgorithm()",
			"Only the energy types"
			" Property::EnergyResolvedProperty::EnergyType::Real and"
			<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara"
			<< " are supported, but the bare susceptibility has a"
			<< " different energy type.",
			"This should never happen, contact the developer."
		);
	}

	//Denominator in the expression chi_RPA = chi_0/(1 - U\chi_0)
	vector<complex<double>*> denominators;

	//Initialize denominator matrices to unit matrices
	for(
		unsigned int e = 0;
		e < energies.size();
		e++
	){
		//Create denominator matrix
		denominators.push_back(
			new complex<double>[matrixDimension*matrixDimension]
		);
		//Initialize denominator matrices to unit matrices
		for(
			unsigned int c = 0;
			c < matrixDimension*matrixDimension;
			c++
		){
			denominators.at(e)[c] = 0.;
		}
		for(unsigned int c = 0; c < matrixDimension; c++)
			denominators.at(e)[c*matrixDimension + c] = 1.;
	}

	//Calculate denominator = (1 + U\chi_0)
	for(unsigned int n = 0; n < interactionAmplitudes.size(); n++){
		const InteractionAmplitude &interactionAmplitude = interactionAmplitudes.at(n);

		const Index &c0 = interactionAmplitude.getCreationOperatorIndex(0);
		const Index &c1 = interactionAmplitude.getCreationOperatorIndex(1);
		const Index &a0 = interactionAmplitude.getAnnihilationOperatorIndex(0);
		const Index &a1 = interactionAmplitude.getAnnihilationOperatorIndex(1);
		complex<double> amplitude = interactionAmplitude.getAmplitude();

		if(abs(amplitude) < 1e-10)
			continue;

		int c0LinearIntraBlockIndex
			= getModel().getHoppingAmplitudeSet().getBasisIndex(
				Index(kIndex, c0)
			) - getModel().getHoppingAmplitudeSet(
			).getFirstIndexInBlock(kIndex);
		int a1LinearIntraBlockIndex
			= getModel().getHoppingAmplitudeSet().getBasisIndex(
				Index(kIndex, a1)
			) - getModel().getHoppingAmplitudeSet(
			).getFirstIndexInBlock(kIndex);
		int row = intraBlockIndexList.size()*c0LinearIntraBlockIndex
			+ a1LinearIntraBlockIndex;
		for(unsigned int c = 0; c < intraBlockIndexList.size(); c++){
			for(unsigned int d = 0; d < intraBlockIndexList.size(); d++){
				int col = intraBlockIndexList.size()*c + d;

				const vector<complex<double>> &susceptibility
					= bareSusceptibility.getData();
				unsigned int offset
					= bareSusceptibility.getOffset({
						kIndex,
						c1,
						a0,
						intraBlockIndexList[d],
						intraBlockIndexList[c]
					});
				for(
					unsigned int i = 0;
					i < energies.size();
					i++
				){
					denominators.at(i)[
						matrixDimension*col + row
					] += amplitude*susceptibility.at(offset + i);
				}
			}
		}
	}

	//Calculate (1 + U\chi_0)^{-1}
	for(unsigned int n = 0; n < energies.size(); n++)
		invertMatrix(denominators.at(n), matrixDimension);

	//Initialize \chi_RPA
	vector<vector<vector<complex<double>>>> rpaSusceptibility;
	for(unsigned int orbital2 = 0; orbital2 < intraBlockIndexList.size(); orbital2++){
		rpaSusceptibility.push_back(vector<vector<complex<double>>>());
		for(unsigned int orbital3 = 0; orbital3 < intraBlockIndexList.size(); orbital3++){
			rpaSusceptibility[orbital2].push_back(vector<complex<double>>());
			for(
				unsigned int e = 0;
				e < energies.size();
				e++
			){
				rpaSusceptibility[orbital2][orbital3].push_back(0.);
			}
		}
	}

	//Calculate \chi_RPA = \chi_0/(1 + U\chi_0)
	for(unsigned int c = 0; c < intraBlockIndexList.size(); c++){
		for(unsigned int d = 0; d < intraBlockIndexList.size(); d++){
			const vector<complex<double>> &susceptibility
				= bareSusceptibility.getData();
			unsigned int offset = bareSusceptibility.getOffset({
				kIndex,
				intraBlockIndices[0],
				intraBlockIndices[1],
				intraBlockIndexList[d],
				intraBlockIndexList[c]
			});
			for(unsigned int orbital2 = 0; orbital2 < intraBlockIndexList.size(); orbital2++){
				for(unsigned int orbital3 = 0; orbital3 < intraBlockIndexList.size(); orbital3++){
					for(
						unsigned int i = 0;
						i < energies.size();
						i++
					){
						rpaSusceptibility[orbital2][orbital3].at(i) += denominators.at(i)[
							matrixDimension*(
								intraBlockIndexList.size()*orbital2
								+ orbital3
							) + intraBlockIndexList.size()*c + d
						]*susceptibility.at(offset + i);
					}
				}
			}
		}
	}

	//Free memory allocated for denominators
	for(unsigned int n = 0; n < energies.size(); n++)
		delete [] denominators.at(n);

	return rpaSusceptibility;
}

IndexedDataTree<vector<complex<double>>> RPASusceptibility::calculateRPASusceptibility(
	const Index &index
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::calculateRPASusceptibility()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	IndexTree intraBlockIndexTree
		= getModel().getHoppingAmplitudeSet().getIndexTree(kIndex);
	vector<Index> intraBlockIndexList;
	for(
		IndexTree::ConstIterator iterator
			= intraBlockIndexTree.cbegin();
		iterator != intraBlockIndexTree.cend();
		++iterator
	){
		Index index = *iterator;
		for(unsigned int n = 0; n < kIndex.getSize(); n++)
			index.popFront();

		intraBlockIndexList.push_back(index);
	}

	//TODO
	//The way intraBlockIndices[n] are used assumes that they have a single
	//subindex, which limits generality.
	vector<vector<vector<complex<double>>>> result = rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudes
	);
	IndexedDataTree<vector<complex<double>>> indexedDataTree;
	for(unsigned int n = 0; n < result.size(); n++){
		for(unsigned int c = 0; c < result[n].size(); c++){
			indexedDataTree.add(
				result[n][c],
				{
					kIndex,
					intraBlockIndices[0],
					intraBlockIndices[1],
					intraBlockIndexList[n],
					intraBlockIndexList[c]
				}
			);
		}
	}

	return indexedDataTree;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
