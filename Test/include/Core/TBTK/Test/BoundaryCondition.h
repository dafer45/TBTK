#include "TBTK/BoundaryCondition.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(BoundaryCondition, Constructor){
	//Not testable on its own.
}

TEST(BoundaryCondition, SerializeToJSON){
	BoundaryCondition boundaryCondition0;
	boundaryCondition0.add(HoppingAmplitude(1, {0, 0}, {0, 0}));
	boundaryCondition0.add(HoppingAmplitude(2, {0, 0}, {0, 1}));
	boundaryCondition0.add(HoppingAmplitude(3, {0, 0}, {0, 2}));
	boundaryCondition0.set(SourceAmplitude(1, {0, 0}));
	boundaryCondition0.setEliminationIndex({0, 0});

	BoundaryCondition boundaryCondition1(
		boundaryCondition0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//HoppingAmplitudes.
	const HoppingAmplitudeList &hoppingAmplitudeList
		= boundaryCondition1.getHoppingAmplitudeList();
	EXPECT_EQ(hoppingAmplitudeList.getSize(), 3);

	//SourceAmplitude.
	const SourceAmplitude &sourceAmplitude
		= boundaryCondition1.getSourceAmplitude();
	EXPECT_DOUBLE_EQ(real(sourceAmplitude.getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitude.getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitude.getIndex().equals({0, 0}));

	//EliminationIndex.
	EXPECT_TRUE(boundaryCondition1.getEliminationIndex().equals({0, 0}));
}

TEST(BoundaryCondition, add){
	//Tested through BoundaryCondition::getHoppingAmplitudeSet().
}

TEST(BoundaryCondition, getHoppingAmplitudeSet){
	BoundaryCondition boundaryCondition;
	boundaryCondition.add(HoppingAmplitude(1, {0, 0}, {0, 0}));
	boundaryCondition.add(HoppingAmplitude(2, {0, 0}, {0, 1}));
	boundaryCondition.add(HoppingAmplitude(3, {0, 0}, {0, 2}));
	const HoppingAmplitudeList &hoppingAmplitudeList
		= boundaryCondition.getHoppingAmplitudeList();
	EXPECT_EQ(hoppingAmplitudeList.getSize(), 3);
}

TEST(BoundaryCondition, set){
	//Tested through BoundaryCondition::getSourceAmplitude().
}

TEST(BoundaryCondition, getSourceAmplitude){
	BoundaryCondition boundaryCondition;
	boundaryCondition.set(SourceAmplitude(1, {0, 0}));
	const SourceAmplitude &sourceAmplitude
		= boundaryCondition.getSourceAmplitude();
	EXPECT_DOUBLE_EQ(real(sourceAmplitude.getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitude.getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitude.getIndex().equals({0, 0}));
}

TEST(BoundaryCondition, setEliminationIndex){
	//Tested through BoundaryCondition::getEliminationIndex().
}

TEST(BoundaryCondition, getEliminationIndex){
	BoundaryCondition boundaryCondition;
	boundaryCondition.setEliminationIndex({0, 0});
	EXPECT_TRUE(boundaryCondition.getEliminationIndex().equals({0, 0}));
}

TEST(BoundaryCondition, serialize){
	//Tested through SerializeToJSON.
}

TEST(BoundaryCondition, getSizeInBytes){
	BoundaryCondition boundaryCondition;
	EXPECT_TRUE(boundaryCondition.getSizeInBytes() > 0);
}

};
