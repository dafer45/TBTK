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

/** @file BlockDiagonalizer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

BlockDiagonalizer::BlockDiagonalizer(){
	maxIterations = 50;
	selfConsistencyCallback = nullptr;

	parallelExecution = false;
}

void BlockDiagonalizer::run(){
	int iterationCounter = 0;
	init();
	eigenVectorsAvailable = false;
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Running Diagonalizer\n";
	while(iterationCounter++ < maxIterations){
		if(getGlobalVerbose() && getVerbose()){
			if(iterationCounter%10 == 1)
				Streams::out << " ";
			if(iterationCounter%50 == 1)
				Streams::out << "\n";
			Streams::out << "." << flush;
		}

		solve();

		if(selfConsistencyCallback){
			if(selfConsistencyCallback->selfConsistencyCallback(*this))
				break;
			else{
				update();
			}
		}
		else{
			break;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";
}

void BlockDiagonalizer::init(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Initializing BlockDiagonalizer\n";

	//Setup the BlockStructureDescriptor.
	blockStructureDescriptor = BlockStructureDescriptor(
		getModel().getHoppingAmplitudeSet()
	);

	/** Calculate block sizes and blockOffsets. */
	blockSizes.clear();
	eigenVectorSizes.clear();
	blockOffsets.clear();
	eigenVectorOffsets.clear();
	for(
		unsigned int n = 0;
		n < blockStructureDescriptor.getNumBlocks();
		n++
	){
		unsigned int numStates
			= blockStructureDescriptor.getNumStatesInBlock(n);
		if(useGPUAcceleration){
			blockSizes.push_back(numStates*numStates);
		}
		else{
			blockSizes.push_back((numStates*(numStates+1))/2);
		}
		eigenVectorSizes.push_back(numStates*numStates);

		if(n == 0){
			blockOffsets.push_back(0);
			eigenVectorOffsets.push_back(0);
		}
		else{
			blockOffsets.push_back(
				blockOffsets.back() + blockSizes.at(n-1)
			);
			eigenVectorOffsets.push_back(
				eigenVectorOffsets.back() + 
				eigenVectorSizes.at(n-1)
			);
		}
	}

	//Calculate amount of memory required to store all the blocks of the
	//Hamiltonian.
	size_t hamiltonianSize = 0;
	size_t eigenVectorsSize = 0;
	for(
		unsigned int n = 0;
		n < blockStructureDescriptor.getNumBlocks();
		n++
	){
		size_t numStatesInBlock
			= blockStructureDescriptor.getNumStatesInBlock(n);
		eigenVectorsSize += numStatesInBlock*numStatesInBlock;
		if(!useGPUAcceleration){
			hamiltonianSize += (numStatesInBlock*(numStatesInBlock + 1))/2;
		}
		else{
			hamiltonianSize += numStatesInBlock*numStatesInBlock;
		}
	}

	if(getGlobalVerbose() && getVerbose()){
		size_t numBytesHamiltonian = hamiltonianSize*sizeof(
			complex<double>
		);
		size_t numBytesEigenVectors = eigenVectorsSize*sizeof(
			complex<double>
		);

		Streams::out << "\tBasis size: " << getModel().getBasisSize()
			<< "\n";
		if(numBytesHamiltonian < 1024){
			Streams::out << "\tHamiltonian size: "
				<< numBytesHamiltonian*sizeof(complex<double>)
				<< "B\n";
		}
		else if(numBytesHamiltonian < 1024*1024){
			Streams::out << "\tHamiltonian size: "
				<< numBytesHamiltonian/1024 << "KB\n";
		}
		else if(numBytesHamiltonian < 1024*1024*1024){
			Streams::out << "\tHamiltonian size: "
				<< numBytesHamiltonian/1024/1024 << "MB\n";
		}
		else{
			Streams::out << "\tHamiltonian size: "
				<< numBytesHamiltonian/1024/1024/1024
				<< "GB\n";
		}
		if(numBytesEigenVectors < 1024){
			Streams::out << "\tEigenvectors size: "
				<< numBytesEigenVectors << "B\n";
		}
		else if(numBytesEigenVectors < 1024*1024){
			Streams::out << "\tEigenvectors size: "
				<< numBytesEigenVectors/1024 << "KB\n";
		}
		else if(numBytesEigenVectors < 1024*1024*1024){
			Streams::out << "\tEigenvectors size: "
				<< numBytesEigenVectors/1024/1024 << "MB\n";
		}
		else{
			Streams::out << "\tEigenvectors size: "
				<< numBytesEigenVectors/1024/1024/1024
				<< "GB\n";
		}
		Streams::out << "\tNumber of blocks: "
			<< blockStructureDescriptor.getNumBlocks() << "\n";
	}
	if(!useGPUAcceleration){ //No hamiltonian needs to be stored as GPU solver 
								//solves it in place of CArray eigenVectors
		hamiltonian = CArray<complex<double>>(hamiltonianSize);
	}
	eigenValues = CArray<double>(getModel().getBasisSize());
	eigenVectors = CArray<complex<double>>(eigenVectorsSize);
	update();
}

void BlockDiagonalizer::update(){
	const Model &model = getModel();

	unsigned int hamiltonianSize = 0;
	for(
		unsigned int n = 0;
		n < blockStructureDescriptor.getNumBlocks();
		n++
	){
		unsigned int numStatesInBlock
			= blockStructureDescriptor.getNumStatesInBlock(n);
		if(!useGPUAcceleration){ //TODO See if one could avoid this
			hamiltonianSize += (numStatesInBlock*(numStatesInBlock + 1))/2;
		}
		else{
			hamiltonianSize += hamiltonianSize*hamiltonianSize;
		}
		
	}
	if(useGPUAcceleration){
		for(unsigned int n = 0; n < eigenVectors.getSize(); n++){
			eigenVectors[n] = 0.;
		}
	}
	else{
		for(unsigned int n = 0; n < hamiltonianSize; n++)
			hamiltonian[n] = 0.;
	}


	IndexTree blockIndices
		= getModel().getHoppingAmplitudeSet().getSubspaceIndices();
	IndexTree::ConstIterator blockIterator = blockIndices.cbegin();
	if(parallelExecution){
		vector<HoppingAmplitudeSet::ConstIterator> iterators;
		vector<HoppingAmplitudeSet::ConstIterator> endIterators;

		while(blockIterator != blockIndices.cend()){
			Index blockIndex = *blockIterator;

			iterators.push_back(
				getModel().getHoppingAmplitudeSet().cbegin(
					blockIndex
				)
			);
			endIterators.push_back(
				getModel().getHoppingAmplitudeSet().cend(
					blockIndex
				)
			);

			++blockIterator;
		}

		#pragma omp parallel for
		for(unsigned int block = 0; block < iterators.size(); block++){
			HoppingAmplitudeSet::ConstIterator &iterator
				= iterators[block];
			HoppingAmplitudeSet::ConstIterator &endIterator
				= endIterators[block];

			int minBasisIndex = iterator.getMinBasisIndex();
			while(iterator != endIterator){
				int from = model.getHoppingAmplitudeSet(
				).getBasisIndex(
					(*iterator).getFromIndex()
				) - minBasisIndex;
				int to = model.getHoppingAmplitudeSet(
				).getBasisIndex(
					(*iterator).getToIndex()
				) - minBasisIndex;
				if(from <= to){
					unsigned int numStates
						= blockStructureDescriptor.getNumStatesInBlock(block);
					if(useGPUAcceleration){
						//Temporarily store hamiltonian in eigenVectors
						eigenVectors[blockOffsets.at(block) +
							to + 
							from*numStates] 
							+= (*iterator).getAmplitude();
					}
					else{
						hamiltonian[
							blockOffsets.at(block)
							+ to
							+ (from*(2*numStates-from-1))/2
						] += (*iterator).getAmplitude();
					}

				}
				++iterator;
			}
		}
	}
	else{
		unsigned int blockCounter = 0;
		while(blockIterator != blockIndices.cend()){
			Index blockIndex = *blockIterator;

			HoppingAmplitudeSet::ConstIterator iterator
				= getModel().getHoppingAmplitudeSet().cbegin(
					blockIndex
				);
			int minBasisIndex = iterator.getMinBasisIndex();
			unsigned int numStates
				= blockStructureDescriptor.getNumStatesInBlock(blockCounter);
			while(
				iterator != getModel().getHoppingAmplitudeSet(
				).cend(blockIndex)
			){
				int from = model.getHoppingAmplitudeSet(
				).getBasisIndex(
					(*iterator).getFromIndex()
				) - minBasisIndex;
				int to = model.getHoppingAmplitudeSet(
				).getBasisIndex(
					(*iterator).getToIndex()
				) - minBasisIndex;
				if(from <= to){
					if(useGPUAcceleration){
						//Temporarily store hamiltonian in eigenVectors
						eigenVectors[blockOffsets.at(blockCounter) +
							to + 
							from*numStates] 
							+= (*iterator).getAmplitude();
					}
					else{
						hamiltonian[
							blockOffsets.at(blockCounter)
							+ to
							+ (from*(2*numStates-from-1))/2
						] += (*iterator).getAmplitude();
					}
				}
				++iterator;
			}
			blockCounter++;
			++blockIterator;
		}
	}
}

// //Lapack function for matrix diagonalization of banded triangular matrix
// extern "C" void zhbeb_(
// 	char *jobz,		//'E' = Eigenvalues only, 'V' = Eigenvalues and eigenvectors.
// 	char *uplo,		//'U' = Stored as upper triangular, 'L' = Stored as lower triangular.
// 	int *n,			//n*n = Matrix size
// 	int *kd,		//Number of (sub/super)diagonal elements
// 	complex<double> *ab,	//Input matrix
// 	int *ldab,		//Leading dimension of array ab. ldab >= kd + 1
// 	double *w,		//Eigenvalues, is in accending order if info = 0
// 	complex<double> *z,	//Eigenvectors
// 	int *ldz,		//
// 	complex<double> *work,	//Workspace, dimension = max(1, 2*N-1)
// 	double *rwork,		//Workspace, dimension = max(1, 3*N-2)
// 	int *info);		//0 = successful, <0 = -info value was illegal, >0 = info number of off-diagonal elements failed to converge.

void BlockDiagonalizer::solve(){
	if(true){//Currently no support for banded matrices.
		if(parallelExecution){
			vector<unsigned int> eigenValuesOffsets;
			eigenValuesOffsets.push_back(0);
			for(
				unsigned int b = 1;
				b < blockStructureDescriptor.getNumBlocks();
				b++
			){
				eigenValuesOffsets.push_back(
					eigenValuesOffsets[b-1]
					+ blockStructureDescriptor.getNumStatesInBlock(b-1)
				);
			}

			#pragma omp parallel for
			for(
				unsigned int b = 0;
				b < blockStructureDescriptor.getNumBlocks();
				b++
			){
				if(useGPUAcceleration){
					int n = blockStructureDescriptor.getNumStatesInBlock(b);
					solveGPU(
						eigenVectors.getData() + eigenVectorOffsets.at(b), //Hamiltionian is stored in eigenVectors
							eigenValues.getData()+ eigenValuesOffsets[b],
							n
						);
				}
				else{
					solveCPU(hamiltonian.getData()
							+ blockOffsets.at(b), 
							eigenValues.getData()
							+ eigenValuesOffsets[b],
							eigenVectors.getData()
					 		+ eigenVectorOffsets.at(b),
							blockStructureDescriptor.getNumStatesInBlock(b)
							);
				}
			}
		}
		else{
			unsigned int eigenValuesOffset = 0;
			for(
				unsigned int b = 0;
				b < blockStructureDescriptor.getNumBlocks();
				b++
			){
				int n = blockStructureDescriptor.getNumStatesInBlock(b);	//...nxn-matrix.
				if(useGPUAcceleration){
					solveGPU(
						eigenVectors.getData() + eigenVectorOffsets.at(b), //Hamiltionian is stored in eigenVectors
							eigenValues.getData()+ eigenValuesOffset,
							n
						);
				}
				else{
					solveCPU(hamiltonian.getData()
							+ blockOffsets.at(b), 
							eigenValues.getData()
							+ eigenValuesOffset,
							eigenVectors.getData()
					 		+ eigenVectorOffsets.at(b),
							n
							);
				}

				eigenValuesOffset += blockStructureDescriptor.getNumStatesInBlock(b);
			}
		}
	}
/*	else{
		int kd;
		if(size_z != 1)
			kd = orbitals*size_x*size_y*size_z;
		else if(size_y != 1)
			kd = orbitals*size_x*size_y;
		else
			kd = orbitals*size_x;
		//Setup zhbev to calculate...
		char jobz = 'V';			//...eigenvalues and eigenvectors...
		char uplo = 'U';			//...for and upper triangluar...
		int n = orbitals*size_x*size_y*size_z;	//...nxn-matrix.
		int ldab = kd + 1;
		//Initialize workspaces
		complex<double> *work = new complex<double>[n];
		double *rwork = new double[3*n-2];
		int info;
		//Solve brop
		zhbev_(&jobz, &uplo, &n, &kd, hamiltonian, &ldab, eigenValues, eigenVectors, &n, work, rwork, &info);

		//delete workspaces
		delete [] work;
		delete [] rwork;
	}*/
}

const std::complex<double> BlockDiagonalizer::getAmplitude(
	int state,
	const Index &index
){
	TBTKAssert( eigenVectorsAvailable == true,
		"BlockDiagonalizer::getAmplitude()",
		"Eigenvectors not available.",
		"Use CalculationMode::EigenValuesAndEigenVectors instead."
	);
	const Model &model = getModel();
	unsigned int block = blockStructureDescriptor.getBlockIndex(state);
	unsigned int offset = eigenVectorOffsets.at(block);
	unsigned int linearIndex = model.getBasisIndex(index);
	unsigned int firstStateInBlock
		= blockStructureDescriptor.getFirstStateInBlock(block);
	unsigned int lastStateInBlock = firstStateInBlock
		+ blockStructureDescriptor.getNumStatesInBlock(block)-1;
	offset += (
		state - firstStateInBlock
	)*blockStructureDescriptor.getNumStatesInBlock(block);
	if(
		linearIndex >= firstStateInBlock
		&& linearIndex <= lastStateInBlock
	){
		return eigenVectors[
			offset + (linearIndex - firstStateInBlock)
		];
	}
	else{
		return 0;
	}
}

const std::complex<double> BlockDiagonalizer::getAmplitude(
	const Index &blockIndex,
	int state,
	const Index &intraBlockIndex
){
	TBTKAssert( eigenVectorsAvailable == true,
		"BlockDiagonalizer::getAmplitude()",
		"Eigenvectors not available.",
		"Use CalculationMode::EigenValuesAndEigenVectors instead."
	);
	int firstStateInBlock = getModel().getHoppingAmplitudeSet(
	).getFirstIndexInBlock(blockIndex);
	unsigned int block = blockStructureDescriptor.getBlockIndex(
		firstStateInBlock
	);
	TBTKAssert(
		state >= 0
		&& state < (int)blockStructureDescriptor.getNumStatesInBlock(
			block
		),
		"BlockDiagonalizer::getAmplitude()",
		"Out of bound error. The block with block Index "
		<< blockIndex.toString() << " has "
		<< blockStructureDescriptor.getNumStatesInBlock(block)
		<< " states, but state " << state << " was requested.",
		""
	);
	unsigned int offset = eigenVectorOffsets.at(block)
		+ state*blockStructureDescriptor.getNumStatesInBlock(block);
	unsigned int linearIndex = getModel().getBasisIndex(
		Index(blockIndex, intraBlockIndex)
	);

	return eigenVectors[offset + (linearIndex - firstStateInBlock)];
}

};	//End of namespace Solver
};	//End of namespace TBTK
