/* Copyright 2018 Kristofer Björnson
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

/** @file Greens.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/Greens.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Solver{

Greens::Greens() : Communicator(true){
}

Greens::~Greens(){
}

Property::GreensFunction Greens::calculateInteractingGreensFunction(
	const Property::SelfEnergy &selfEnergy
) const{
	TBTKAssert(
		greensFunction->getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Solver::Greens::addSelfEnergy()",
		"The Green's function must be on the Custom format.",
		"See Property::AbstractProperty for detailed information about"
		<< " the storage formats. Also see the documentation for the"
		<< " PropertyExtractor that was used to calculate the Green's"
		<< " function for details on how to extract it on the Custom"
		<< " format."
	);
	TBTKAssert(
		selfEnergy.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Solver::Greens::addSelfEnergy()",
		"The self-energy must be on the Custom format.",
		"See Property::AbstractProperty for detailed information about"
		<< " the storage formats. Also see the documentation for the"
		<< " PropertyExtractor that was used to calculate the"
		<< " self-energy for details on how to extract it on the"
		<< " Custom format."
	);
	TBTKAssert(
		greensFunction->getEnergyType() == selfEnergy.getEnergyType(),
		"Solver::GreensFunction::addSelfEnergy()",
		"The GreensFunction and SelfEnergy must have the same energy"
		<< " type.",
		""
	);

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Solver::Greens::addSelfEnergy()\n";

	//Get the Green's function and self-energy's  IndexTrees and the
	//HoppingAmplitudeSet from the Model.
	const IndexTree &greensFunctionIndices
		= greensFunction->getIndexDescriptor().getIndexTree();
	const IndexTree &selfEnergyIndices
		= selfEnergy.getIndexDescriptor().getIndexTree();
	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= getModel().getHoppingAmplitudeSet();

	//Only supports Green's functions and self-energies with the exact same
	//Index structure yet. Check this and print error message if this is
	//not satisfied.
	TBTKAssert(
		greensFunctionIndices.equals(selfEnergyIndices),
		"Solver::Greens::addSelfEnergy()",
		"Only GreensFunctions and SelfEnergies with the exact same"
		<< " Index structure are supported yet.",
		""
	);

	//Check whether the Green's function is restricted to intra block Index
	//pairs and extract those blocks.
	bool isBlockRestricted = true;
	bool globalBlockContained = false;
	IndexTree containedBlocks;
	for(
		IndexTree::ConstIterator iterator = greensFunctionIndices.cbegin();
		iterator != greensFunctionIndices.cend();
		++iterator
	){
		vector<Index> components = (*iterator).split();
		Index block0 = hoppingAmplitudeSet.getSubspaceIndex(
			components[0]
		);
		Index block1 = hoppingAmplitudeSet.getSubspaceIndex(
			components[1]
		);

		if(!block0.equals(block1)){
			isBlockRestricted = false;
			break;
		}
		if(block0.equals({})){
			globalBlockContained = true;
			break;
		}

		containedBlocks.add(block0);
	}
	containedBlocks.generateLinearMap();

	Property::GreensFunction interactingGreensFunction;

	if(globalBlockContained){
		//Print whether the Green's function is block restricted.
		if(getGlobalVerbose() && getVerbose()){
			Streams::out << "\tThe Green's function has a single"
				<< " global block.\n";
		}

		//Check that all Index pairs are present.
		IndexTree intraBlockIndices
			= hoppingAmplitudeSet.getIndexTree();
		for(
			IndexTree::ConstIterator iterator0
				= intraBlockIndices.cbegin();
			iterator0 != intraBlockIndices.cend();
			++iterator0
		){
			for(
				IndexTree::ConstIterator iterator1
					= intraBlockIndices.cbegin();
				iterator1 != intraBlockIndices.cend();
				++iterator1
			){
				Index compoundIndex({*iterator0, *iterator1});
				TBTKAssert(
					greensFunction->contains(
						compoundIndex
					),
					"Solver::Greens::addSelfEnergy()",
					"Missing Index. The Index '"
					<< compoundIndex.toString() << "' is"
					<< " missing in the Green's function.",
					""
				);
			}
		}
	}
	else{
		//Print whether the Green's function is block restricted.
		if(getGlobalVerbose() && getVerbose()){
			Streams::out << "\tThe Green's function is block restricted:\t"
				<< (isBlockRestricted ? "True" : "False")
				<< "\n";
		}

		//Only block restricted Green's functions supported yet. Print error
		//message if this is not fulfilled.
		if(!isBlockRestricted){
			TBTKExit(
				"Greens::addSelfEnergy()",
				"Only Green's functions with Index pairs that are"
				<< " restricted to intra block pairs are supported"
				<< " yet.",
				""
			);
		}

		//Check that all intra block Index pairs are present for those blocks
		//for which at least one Index pair is present.
		for(
			IndexTree::ConstIterator iterator = containedBlocks.cbegin();
			iterator != containedBlocks.cend();
			++iterator
		){
			const Index &blockIndex = (*iterator);
			IndexTree intraBlockIndices = hoppingAmplitudeSet.getIndexTree(
				blockIndex
			);
			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndices.cbegin();
				iterator0 != intraBlockIndices.cend();
				++iterator0
			){
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndices.cbegin();
					iterator1 != intraBlockIndices.cend();
					++iterator1
				){
					Index compoundIndex({*iterator0, *iterator1});
					TBTKAssert(
						greensFunction->contains(compoundIndex),
						"Solver::Greens::addSelfEnergy()",
						"Missing Index. The Green's function"
						<< " has at least one component in the"
						<< " block '" << blockIndex.toString()
						<< "', but '"
						<< compoundIndex.toString() << "' is"
						<< " not contained.",
						"Make sure that the GreenFunction"
						<< " contains all intra block Index"
						<< " pairs for those blocks that it"
						<< " contains at least one intra block"
						<< " Index pair."
					);
				}
			}
		}
	}

	switch(greensFunction->getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real:
		interactingGreensFunction = Property::GreensFunction(
			greensFunctionIndices,
			greensFunction->getType(),
			greensFunction->getLowerBound(),
			greensFunction->getUpperBound(),
			greensFunction->getResolution()
		);

		break;
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::FermionicMatsubara:
		interactingGreensFunction = Property::GreensFunction(
			greensFunctionIndices,
			greensFunction->getLowerMatsubaraEnergyIndex(),
			greensFunction->getUpperMatsubaraEnergyIndex(),
			greensFunction->getFundamentalMatsubaraEnergy()
		);

		break;
	default:
		TBTKExit(
			"Solver::GreensFunction::addSelfEnergy()",
			"Unknown energy type",
			"This should never happen, contact the developer."
		);
	}

	if(globalBlockContained){
		IndexTree intraBlockIndices
			= hoppingAmplitudeSet.getIndexTree();
		double blockSize = intraBlockIndices.getSize();

		Matrix<complex<double>> matrix(blockSize, blockSize);
		for(
			unsigned int n = 0;
			n < greensFunction->getResolution();
			n++
		){
			//Convert one block of the Green's function to a
			//matrix.
			unsigned int row = 0;
			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndices.cbegin();
				iterator0 != intraBlockIndices.cend();
				++iterator0
			){
				unsigned int column = 0;
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndices.cbegin();
					iterator1 != intraBlockIndices.cend();
					++iterator1
				){
					Index compoundIndex = {*iterator0, *iterator1};

					matrix.at(row, column)
						= (*greensFunction)(
							compoundIndex,
							n
						);

					column++;
				}

				row++;
			}

			//Invert the matrix.
			matrix.invert();

			//Add the self energy.
			row = 0;
			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndices.cbegin();
				iterator0 != intraBlockIndices.cend();
				++iterator0
			){
				unsigned int column = 0;
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndices.cbegin();
					iterator1 != intraBlockIndices.cend();
					++iterator1
				){
					Index compoundIndex = {*iterator0, *iterator1};

					matrix.at(row, column) += selfEnergy(
						compoundIndex,
						n
					);

					column++;
				}

				row++;
			}

			//Invert matrix.
			matrix.invert();

			//Write the matrix back into the corresponding block of
			//the full Green's function.
			row = 0;
			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndices.cbegin();
				iterator0 != intraBlockIndices.cend();
				++iterator0
			){
				unsigned int column = 0;
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndices.cbegin();
					iterator1 != intraBlockIndices.cend();
					++iterator1
				){
					Index compoundIndex = {*iterator0, *iterator1};

					interactingGreensFunction(
						compoundIndex,
						n
					) = matrix.at(row, column);

					column++;
				}

				row++;
			}
		}
	}
	else{
		for(
			IndexTree::ConstIterator iterator = containedBlocks.cbegin();
			iterator != containedBlocks.cend();
			++iterator
		){
			const Index &blockIndex = (*iterator);
			IndexTree intraBlockIndices = hoppingAmplitudeSet.getIndexTree(
				blockIndex
			);
			double blockSize = intraBlockIndices.getSize();

			Matrix<complex<double>> matrix(blockSize, blockSize);
			for(
				unsigned int n = 0;
				n < greensFunction->getResolution();
				n++
			){
				//Convert one block of the Green's function to a
				//matrix.
				unsigned int row = 0;
				for(
					IndexTree::ConstIterator iterator0
						= intraBlockIndices.cbegin();
					iterator0 != intraBlockIndices.cend();
					++iterator0
				){
					unsigned int column = 0;
					for(
						IndexTree::ConstIterator iterator1
							= intraBlockIndices.cbegin();
						iterator1 != intraBlockIndices.cend();
						++iterator1
					){
						Index compoundIndex = {*iterator0, *iterator1};

						matrix.at(row, column)
							= (*greensFunction)(
								compoundIndex,
								n
							);

						column++;
					}

					row++;
				}

				//Invert the matrix.
				matrix.invert();

				//Add the self energy.
				row = 0;
				for(
					IndexTree::ConstIterator iterator0
						= intraBlockIndices.cbegin();
					iterator0 != intraBlockIndices.cend();
					++iterator0
				){
					unsigned int column = 0;
					for(
						IndexTree::ConstIterator iterator1
							= intraBlockIndices.cbegin();
						iterator1 != intraBlockIndices.cend();
						++iterator1
					){
						Index compoundIndex = {*iterator0, *iterator1};

						matrix.at(row, column) += selfEnergy(
							compoundIndex,
							n
						);

						column++;
					}

					row++;
				}

				//Invert matrix.
				matrix.invert();

				//Write the matrix back into the corresponding block of
				//the full Green's function.
				row = 0;
				for(
					IndexTree::ConstIterator iterator0
						= intraBlockIndices.cbegin();
					iterator0 != intraBlockIndices.cend();
					++iterator0
				){
					unsigned int column = 0;
					for(
						IndexTree::ConstIterator iterator1
							= intraBlockIndices.cbegin();
						iterator1 != intraBlockIndices.cend();
						++iterator1
					){
						Index compoundIndex = {*iterator0, *iterator1};

						interactingGreensFunction(compoundIndex, n)
							= matrix.at(row, column);

						column++;
					}

					row++;
				}
			}
		}
	}

	return interactingGreensFunction;
}

};	//End of namespace Solver
};	//End of namespace TBTK
