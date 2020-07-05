#include "TBTK/BoundaryCondition.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(BoundaryCondition, Constructor){
	//Not testable on its own.
}

TEST(BoundaryCondition, SerializeToJSON0){
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

TEST(BoundaryCondition, SerializeToJSON1){
	BoundaryCondition boundaryCondition;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boundaryCondition.serialize(
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(BoundaryCondition, add){
	BoundaryCondition boundaryCondition0;

	//Fail to add HoppingAmplitudes with different to-Indices.
	boundaryCondition0.add(HoppingAmplitude(0, {0}, {2}));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boundaryCondition0.add(HoppingAmplitude(0, {1}, {2}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add HoppingAmplitude with different to-Index than the
	//SourceAmplitudes Index.
	BoundaryCondition boundaryCondition1;
	boundaryCondition1.set(SourceAmplitude(0, {0}));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boundaryCondition1.add(HoppingAmplitude(0, {1}, {2}));
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed to add HoppingAmplitude with the same to-Index as the
	//SourceAmplitudes Index.
	boundaryCondition1.add(HoppingAmplitude(0, {0}, {2}));
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
	//Fail to set SourceAmplitude with an Index that disagrees with the
	//to-Indices of the already added HoppingAmplitudes.
	BoundaryCondition boundaryCondition;
	boundaryCondition.add(HoppingAmplitude(0, {0}, {2}));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boundaryCondition.set(SourceAmplitude(0, {1}));
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed to add SourceAmplitude with an Index that agrees with the
	//to-Indices of the already added HoppingAmplitudes.
	boundaryCondition.set(SourceAmplitude(0, {0}));
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
	BoundaryCondition boundaryCondition;
	boundaryCondition.add(HoppingAmplitude(0, {0}, {1}));
	//Fail to set Elimination Index that is not present among the
	//from-Indices of the added HoppingAmplitudes.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boundaryCondition.setEliminationIndex({2});
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed at adding Elimination Index that is present among the
	//from-Indices of the added HoppingAmplitudes.
	boundaryCondition.setEliminationIndex({1});
}

TEST(BoundaryCondition, getEliminationIndex){
	BoundaryCondition boundaryCondition;
	boundaryCondition.add(HoppingAmplitude(0, {0, 0}, {0, 0}));
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
