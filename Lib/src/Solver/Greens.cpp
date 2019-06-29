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

#include "TBTK/Property/TransmissionRate.h"
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
		"Solver::Greens::calculateInteractingGreensFunction()",
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
		"Solver::Greens::calculateInteractingGreensFunction()",
		"The self-energy must be on the Custom format.",
		"See Property::AbstractProperty for detailed information about"
		<< " the storage formats. Also see the documentation for the"
		<< " PropertyExtractor that was used to calculate the"
		<< " self-energy for details on how to extract it on the"
		<< " Custom format."
	);
	TBTKAssert(
		greensFunction->getEnergyType() == selfEnergy.getEnergyType(),
		"Solver::GreensFunction::calculateInteractingGreensFunction()",
		"The GreensFunction and SelfEnergy must have the same energy"
		<< " type.",
		""
	);

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Solver::Greens::calculateInteractingGreensFunction()\n";

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
		"Solver::Greens::calculateInteractingGreensFunction()",
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
					"Solver::Greens::calculateInteractingGreensFunction()",
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
				"Greens::calculateInteractingGreensFunction()",
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
						"Solver::Greens::calculateInteractingGreensFunction()",
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
			"Solver::GreensFunction::calculateInteractingGreensFunction()",
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
			n < greensFunction->getNumEnergies();
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

					matrix.at(row, column) -= selfEnergy(
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
				n < greensFunction->getNumEnergies();
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

						matrix.at(row, column) -= selfEnergy(
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

Property::TransmissionRate Greens::calculateTransmissionRate(
	const Property::SelfEnergy &selfEnergy0,
	const Property::SelfEnergy &selfEnergy1
) const{
	TBTKAssert(
		greensFunction->getType()
			== Property::GreensFunction::Type::Retarded,
		"Solver::Greens::calculateTransmission()",
		"The Green's function must be of the type"
		<< " Property::GreensFunction::Type::Retarded",
		""
	);
	TBTKAssert(
		greensFunction->getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Solver::Greens::calculateTransmission()",
		"The Green's function must be on the Custom format.",
		"See Property::AbstractProperty for detailed information about"
		<< " the storage formats. Also see the documentation for the"
		<< " PropertyExtractor that was used to calculate the Green's"
		<< " function for details on how to extract it on the Custom"
		<< " format."
	);
	TBTKAssert(
		selfEnergy0.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom
		&& selfEnergy1.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Solver::Greens::calculateTransmission()",
		"The self-energies must be on the Custom format.",
		"See Property::AbstractProperty for detailed information about"
		<< " the storage formats. Also see the documentation for the"
		<< " PropertyExtractor that was used to calculate the"
		<< " self-energy for details on how to extract it on the"
		<< " Custom format."
	);
	TBTKAssert(
		greensFunction->getEnergyType() == selfEnergy0.getEnergyType()
		&& greensFunction->getEnergyType()
			== selfEnergy1.getEnergyType(),
		"Solver::Greens::calculateTransmission()",
		"The GreensFunction and SelfEnergies must have the same energy"
		<< " type.",
		""
	);
	TBTKAssert(
		greensFunction->getEnergyType()
			== Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real,
		"Solver::Greens::calculateTransmission()",
		"Unsupported energy type. Only"
		<< " EnergyResolvedProperty::EnergyType::Real is supported.",
		""
	);
	TBTKAssert(
		greensFunction->getNumEnergies()
			== selfEnergy0.getNumEnergies()
		&& greensFunction->getNumEnergies()
			== selfEnergy1.getNumEnergies(),
		"Solver::Greens::calculateTransmission()",
		"The number of energies must be the same for the Green's"
		<< " function, selfEnergy0, and selfEnergy1. But"
		<< " greensFunction has '" << greensFunction->getNumEnergies()
		<< "' energies, selfEnergy0 has '"
		<< selfEnergy0.getNumEnergies() << "' energies, and"
		<< " selfEnergy1 has '" << selfEnergy1.getNumEnergies()
		<< "'.",
		""
	);
	TBTKAssert(
		abs(
			greensFunction->getLowerBound()
			- selfEnergy0.getLowerBound()
		) < 1e-1*(
			greensFunction->getUpperBound()
			- greensFunction->getLowerBound()
		)/greensFunction->getResolution(),
		"Solver::Greens::calculateTransmission()",
		"Incompatible bounds. The greensFunction and selfEnergy0 has"
		<< " different lower bounds. The Green's functions lower bound"
		<< " is '" << greensFunction->getLowerBound() << "' while"
		<< " self-energies lower bound is '"
		<< selfEnergy0.getLowerBound() << "'.",
		""
	);
	TBTKAssert(
		abs(
			greensFunction->getUpperBound()
			- selfEnergy0.getUpperBound()
		) < 1e-1*(
			greensFunction->getUpperBound()
			- greensFunction->getLowerBound()
		)/greensFunction->getResolution(),
		"Solver::Greens::calculateTransmission()",
		"Incompatible bounds. The greensFunction and selfEnergy0 has"
		<< " different upper bounds. The Green's functions upper bound"
		<< " is '" << greensFunction->getLowerBound() << "' while the"
		<< " self-energies upper bound is '"
		<< selfEnergy0.getLowerBound() << "'.",
		""
	);
	TBTKAssert(
		abs(
			greensFunction->getLowerBound()
			- selfEnergy1.getLowerBound()
		) < 1e-1*(
			greensFunction->getUpperBound()
			- greensFunction->getLowerBound()
		)/greensFunction->getResolution(),
		"Solver::Greens::calculateTransmission()",
		"Incompatible bounds. The greensFunction and selfEnergy1 has"
		<< " different lower bounds. The Green's functions lower bound"
		<< " is '" << greensFunction->getLowerBound() << "' while"
		<< " self-energies lower bound is '"
		<< selfEnergy1.getLowerBound() << "'.",
		""
	);
	TBTKAssert(
		abs(
			greensFunction->getUpperBound()
			- selfEnergy1.getUpperBound()
		) < 1e-1*(
			greensFunction->getUpperBound()
			- greensFunction->getLowerBound()
		)/greensFunction->getResolution(),
		"Solver::Greens::calculateTransmission()",
		"Incompatible bounds. The greensFunction and selfEnergy0 has"
		<< " different upper bounds. The Green's functions upper bound"
		<< " is '" << greensFunction->getLowerBound() << "' while the"
		<< " self-energies upper bound is '"
		<< selfEnergy1.getLowerBound() << "'.",
		""
	);

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Solver::Greens::calculateTransmission()\n";

	//Get the Green's function and self-energyies IndexTrees.
	const IndexTree &greensFunctionIndices
		= greensFunction->getIndexDescriptor().getIndexTree();
	const IndexTree &selfEnergy0Indices
		= selfEnergy0.getIndexDescriptor().getIndexTree();
	const IndexTree &selfEnergy1Indices
		= selfEnergy1.getIndexDescriptor().getIndexTree();

	//Only supports Green's functions and self-energies with the exact same
	//Index structure yet. Check this and print error message if this is
	//not satisfied.
	TBTKAssert(
		greensFunctionIndices.equals(selfEnergy0Indices)
		&& greensFunctionIndices.equals(selfEnergy1Indices),
		"Solver::Greens::calculateTransmission()",
		"Only GreensFunctions and SelfEnergies with the exact same"
		<< " Index structure are supported yet.",
		""
	);

	vector<SparseMatrix<complex<double>>> selfEnergy0Matrices
		= selfEnergy0.toSparseMatrices(getModel());
	vector<SparseMatrix<complex<double>>> selfEnergy1Matrices
		= selfEnergy1.toSparseMatrices(getModel());
	vector<SparseMatrix<complex<double>>> greensFunctionMatrices
		= greensFunction->toSparseMatrices(getModel());

	vector<double> transmissionRateData;
	for(unsigned int n = 0; n < greensFunction->getNumEnergies(); n++){
		//Strictly speaking the broadening should include a factor i.
		//The two i's are moved to multiply the trace to avoid
		//unnecesary multiplications.
		SparseMatrix<complex<double>> broadening0
			= selfEnergy0Matrices[n]
				- selfEnergy0Matrices[n].hermitianConjugate();
		SparseMatrix<complex<double>> broadening1
			= selfEnergy1Matrices[n]
				- selfEnergy1Matrices[n].hermitianConjugate();

		//Gamma_0*G*Gamma_1*Gamma^{\dagger}.
		SparseMatrix<complex<double>> product
			= broadening0
				*greensFunctionMatrices[n]
				*broadening1
				*greensFunctionMatrices[n].hermitianConjugate();

		//Tr[Gamma_0*G*Gamma_1*Gamma^{\dagger}]. i^2 = -1 is taken into
		//account here instead of in the broadenings.
		transmissionRateData.push_back(-real(product.trace()));
	}

	Property::TransmissionRate transmissionRate(
		greensFunction->getLowerBound(),
		greensFunction->getUpperBound(),
		greensFunction->getResolution(),
		transmissionRateData.data()
	);

	return transmissionRate;
}

};	//End of namespace Solver
};	//End of namespace TBTK
