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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file BlockStructureDescriptor.h
 *  @brief Describes the block structure of a Hamiltonian.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BLOCK_STRUCTURE_DESCRIPTOR
#define COM_DAFER45_TBTK_BLOCK_STRUCTURE_DESCRIPTOR

#include "TBTK/HoppingAmplitudeSet.h"

namespace TBTK{

/** @brief Describes the block structure of a Hamiltonian. */
class BlockStructureDescriptor{
public:
	/** Constructs an uninitialized BlockStructureDescriptor. */
	BlockStructureDescriptor();

	/** Constructs a BlockStructureDescriptor. */
	BlockStructureDescriptor(
		const HoppingAmplitudeSet &hoppingAmplitudeSet
	);

	/** Get the number of blocks. */
	unsigned int getNumBlocks() const;

	/** Get the number of states in the given block.
	 *
	 *  @param block The block to get the number of states for.
	 *
	 *  @return The number of states in the given block. */
	unsigned int getNumStatesInBlock(unsigned int block) const;

	/** Get the block index for the block that contains the given state.
	 *
	 *  @param state The state to get the block index for.
	 *
	 *  @return The block index for the block that contains the given
	 *  state. */
	unsigned int getBlockIndex(unsigned int state) const;

	/** Get the first state index in the given block.
	 *
	 *  @param block The block to get the first state index for.
	 *
	 *  @return The first state index in the given block. */
	unsigned int getFirstStateInBlock(unsigned int block) const;
private:
	/** Number of states in the given block. */
	std::vector<unsigned int> numStatesInBlock;

	/** Block indices for the given state. */
	std::vector<unsigned int> stateToBlockMap;

	/** The first state index in the given block. */
	std::vector<unsigned int> blockToStateMap;
};

inline unsigned int BlockStructureDescriptor::getNumBlocks() const{
	return numStatesInBlock.size();
}

inline unsigned int BlockStructureDescriptor::getNumStatesInBlock(
	unsigned int block
) const{
	return numStatesInBlock[block];
}

inline unsigned int BlockStructureDescriptor::getBlockIndex(
	unsigned int state
) const{
	return stateToBlockMap[state];
}

inline unsigned int BlockStructureDescriptor::getFirstStateInBlock(
	unsigned int block
) const{
	return blockToStateMap[block];
}

}; //End of namesapce TBTK

#endif
/// @endcond
