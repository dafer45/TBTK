#include "TBTK/BlockStructureDescriptor.h"

#include "gtest/gtest.h"

namespace TBTK{

#define SETUP_BLOCK_STRUCTURE_DESCRIPTORS() \
	HoppingAmplitudeSet hoppingAmplitudeSet0; \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {0, 0}, {0, 0})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {0, 1}, {0, 1})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {0, 1}, {0, 0})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {0, 0}, {0, 1})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {1, 0}, {0, 0})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {0, 0}, {1, 0})); \
	hoppingAmplitudeSet0.add(HoppingAmplitude(1, {1, 0}, {1, 0})); \
	hoppingAmplitudeSet0.construct(); \
	BlockStructureDescriptor blockStructureDescriptor0( \
		hoppingAmplitudeSet0 \
	); \
 \
	HoppingAmplitudeSet hoppingAmplitudeSet1; \
	hoppingAmplitudeSet1.add(HoppingAmplitude(1, {0, 0}, {0, 0})); \
	hoppingAmplitudeSet1.add(HoppingAmplitude(1, {0, 1}, {0, 1})); \
	hoppingAmplitudeSet1.add(HoppingAmplitude(1, {0, 1}, {0, 0})); \
	hoppingAmplitudeSet1.add(HoppingAmplitude(1, {0, 0}, {0, 1})); \
	hoppingAmplitudeSet1.add(HoppingAmplitude(1, {1, 0}, {1, 0})); \
	hoppingAmplitudeSet1.construct(); \
	BlockStructureDescriptor blockStructureDescriptor1( \
		hoppingAmplitudeSet1 \
	);

TEST(BlockStructureDescriptor, Constructor0){
	//Not testable on its own.
}

TEST(BlockStructuredescriptor, Constructor1){
	//Not testable on its own.
}

TEST(BlockStructureDescriptor, getNumBlocks){
	SETUP_BLOCK_STRUCTURE_DESCRIPTORS();

	EXPECT_EQ(blockStructureDescriptor0.getNumBlocks(), 1);
	EXPECT_EQ(blockStructureDescriptor1.getNumBlocks(), 2);
}

TEST(BlockStructureDescriptor, getNumStatesInBlock){
	SETUP_BLOCK_STRUCTURE_DESCRIPTORS();

	EXPECT_EQ(blockStructureDescriptor0.getNumStatesInBlock(0), 3);
	EXPECT_EQ(blockStructureDescriptor1.getNumStatesInBlock(0), 2);
	EXPECT_EQ(blockStructureDescriptor1.getNumStatesInBlock(1), 1);
}

TEST(BlockStructureDescriptor, getBlockIndex){
	SETUP_BLOCK_STRUCTURE_DESCRIPTORS();

	EXPECT_EQ(blockStructureDescriptor0.getBlockIndex(0), 0);
	EXPECT_EQ(blockStructureDescriptor0.getBlockIndex(1), 0);
	EXPECT_EQ(blockStructureDescriptor0.getBlockIndex(2), 0);
	EXPECT_EQ(blockStructureDescriptor1.getBlockIndex(0), 0);
	EXPECT_EQ(blockStructureDescriptor1.getBlockIndex(1), 0);
	EXPECT_EQ(blockStructureDescriptor1.getBlockIndex(2), 1);
}

TEST(BlockStructureDescriptor, getFirstStateInBlock){
	SETUP_BLOCK_STRUCTURE_DESCRIPTORS();

	EXPECT_EQ(blockStructureDescriptor0.getFirstStateInBlock(0), 0);
	EXPECT_EQ(blockStructureDescriptor1.getFirstStateInBlock(0), 0);
	EXPECT_EQ(blockStructureDescriptor1.getFirstStateInBlock(1), 2);
}

};
