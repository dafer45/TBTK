#include "TBTK/HoppingAmplitudeTree.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(HoppingAmplitudeTree, Constructor){
	//Not testable
}

TEST(HoppingAmplitudeTree, ConstructorCapacity){
	//Not testable
}

TEST(HoppingAmplitudeTree, SerializeToJSON){
//	HoppingAmplitudeTree hoppingAmplitudeTree;
	//TODO...
}

TEST(HoppingAmplitudeTree, add){
	EXPECT_EXIT(
		{
			HoppingAmplitudeTree hoppingAmplitudeTree;
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4}));
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 5}));
			std::cerr << "Test completed.";
			exit(0);
		},
		::testing::ExitedWithCode(0),
		"Test completed."
	);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeTree hoppingAmplitudeTree;
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4}));
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4, 5}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeTree hoppingAmplitudeTree;
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4, 5}));
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeTree hoppingAmplitudeTree;
			hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, -1, 5}));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(HoppingAmplitudeTree, getBasisSize){
	std::string errorMessage = "getBasisSize() failed.";

	HoppingAmplitudeTree hoppingAmplitudeTree;
	EXPECT_EQ(hoppingAmplitudeTree.getBasisSize(), -1) << errorMessage;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2}, {3, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {3, 4}, {1, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {3, 2}, {1, 2}));
	EXPECT_EQ(hoppingAmplitudeTree.getBasisSize(), -1) << errorMessage;
	hoppingAmplitudeTree.generateBasisIndices();
	EXPECT_EQ(hoppingAmplitudeTree.getBasisSize(), 3) << errorMessage;
}

TEST(HoppingAmplitudeTree, getPhysicsIndex){
	//TODO
	//...
}

TEST(HoppingAmplitudeTree, generateBasisIndices){
	//Already tested together with HoppingAmplitudeTree::getBasisSize()
}

};
