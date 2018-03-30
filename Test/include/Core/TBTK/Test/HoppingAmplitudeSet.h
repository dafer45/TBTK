#include "TBTK/HoppingAmplitudeSet.h"

#include "gtest/gtest.h"

namespace TBTK{

//TODO
//Sparse matrix format stored through the variables numMatrixElements,
//cooRowIndices, cooColIndices, and cooValues are not tested since the
//HoppingAmplitudeSet should not be responsible for storing these in the
//future.

TEST(HoppingAmplitudeSet, Constructor){
	//Not testable on its own
}

TEST(HoppingAmplitudeSet, ConstructorCapacity){
	//Not testable on its own
}

//TODO
//This test remains unimplemented as long as the HoppingAmplitudeSet stores the
//sparse matrices. Once sparse matrices are no longer stored, the
//copy-constructor itself can be removed.
TEST(HoppingAmplitudeSet, CopyConstructor){
}

//TODO
//This test remains unimplemented as long as the HoppingAmplitudeSet stores the
//sparse matrices. Once sparse matrices are no longer stored, the
//move-constructor itself can be removed.
TEST(HoppingAmplitudeSet, MoveConstructor){
}

TEST(HoppingAmplitudeSet, SerializeToJSON){
	HoppingAmplitudeSet hoppingAmplitudeSet0;
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet0.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet0.construct();

	HoppingAmplitudeSet hoppingAmplitudeSet1(
		hoppingAmplitudeSet0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(hoppingAmplitudeSet1.getBasisSize(), 5);

	std::vector<Index> indices = hoppingAmplitudeSet1.getIndexList({0, 0, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({0, 0, 0}));
	EXPECT_TRUE(indices[1].equals({0, 0, 1}));
	EXPECT_TRUE(indices[2].equals({0, 0, 2}));

	indices = hoppingAmplitudeSet1.getIndexList({1, 1, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({1, 1, 0}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));

	indices = hoppingAmplitudeSet1.getIndexList({IDX_ALL, IDX_ALL, 1});
	EXPECT_TRUE(indices[0].equals({0, 0, 1}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));
}

//TODO
//This test remains unimplemented as long as the HoppingAmplitudeSet stores the
//sparse matrices. Once sparse matrices are no longer stored, the
//destructor itself can be removed.
TEST(HoppingAmplitudeSet, destructor){
}

//TODO
//This test remains unimplemented as long as the HoppingAmplitudeSet stores the
//sparse matrices. Once sparse matrices are no longer stored, the
//assignment operator itself can be removed.
TEST(HoppingAmplitudeSet, operatorAssignment){
}

//TODO
//This test remains unimplemented as long as the HoppingAmplitudeSet stores the
//sparse matrices. Once sparse matrices are no longer stored, the
//move-assignment operator itself can be removed.
TEST(HoppingAmplitudeSet, operatorMoveAssignment){
}

TEST(HoppingAmplitudeSet, addHoppingAmplitude){
	//Test normal function
	EXPECT_EXIT(
		{
			HoppingAmplitudeSet hoppingAmplitudeSet;
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4}));
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 5}));
			std::cerr << "Test completed.";
			exit(0);
		},
		::testing::ExitedWithCode(0),
		"Test completed."
	);

	//Fail to add HoppingAmplitude with conflicting Index structures (short first).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeSet hoppingAmplitudeSet;
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4}));
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4, 5}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add HoppingAmplitude with conflicting Index structures (long first).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeSet hoppingAmplitudeSet;
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4, 5}));
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add HoppingAmplitude with negative subindex.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			HoppingAmplitudeSet hoppingAmplitudeSet;
			hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, -1, 5}));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(HoppingAmplitudeSet, addHoppingAmplitudeAndHermitianConjugate){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitudeAndHermitianConjugate(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeSet.construct();

	//Check that the HoppingAmplitude itself was added.
	const std::vector<HoppingAmplitude> *hoppingAmplitudes0 = hoppingAmplitudeSet.getHAs({3, 4});
	EXPECT_EQ(hoppingAmplitudes0->size(), 1);

	//Check that the Hermitian conjugate was added.
	const std::vector<HoppingAmplitude> *hoppingAmplitudes1 = hoppingAmplitudeSet.getHAs({1, 2});
	EXPECT_EQ(hoppingAmplitudes1->size(), 1);
}

TEST(HoppingAmplitudeSet, getHAs){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet.construct();

	const std::vector<HoppingAmplitude> *hoppingAmplitudes = hoppingAmplitudeSet.getHAs({0, 0, 1});
	EXPECT_EQ(hoppingAmplitudes->size(), 2);
	for(unsigned int n = 0; n < hoppingAmplitudes->size(); n++){
		const HoppingAmplitude &hoppingAmplitude = hoppingAmplitudes->at(n);

		EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({0, 0, 1}));
		EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({0, 0, 1}) || hoppingAmplitude.getToIndex().equals({0, 0, 2}));
	}

	hoppingAmplitudes = hoppingAmplitudeSet.getHAs({0, 0, 2});
	EXPECT_EQ(hoppingAmplitudes->size(), 1);
	for(unsigned int n = 0; n < hoppingAmplitudes->size(); n++){
		const HoppingAmplitude &hoppingAmplitude = hoppingAmplitudes->at(n);

		EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({0, 0, 2}));
		EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({0, 0, 1}));
	}
}

TEST(HoppingAmplitudeSet, getBasisIndex){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));

	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 0}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 2}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({1, 1, 0}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({1, 1, 1}), -1);

	hoppingAmplitudeSet.construct();

	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 0}), 0);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 1}), 1);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({0, 0, 2}), 2);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({1, 1, 0}), 3);
	EXPECT_EQ(hoppingAmplitudeSet.getBasisIndex({1, 1, 1}), 4);
}

TEST(HoppingAmplitudeSet, getPhysicsIndex){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.getPhysicalIndex(0);
		},
		::testing::ExitedWithCode(1),
		""
	);

	hoppingAmplitudeSet.construct();

	EXPECT_TRUE(hoppingAmplitudeSet.getPhysicalIndex(0).equals({0, 0, 0}));
	EXPECT_TRUE(hoppingAmplitudeSet.getPhysicalIndex(1).equals({0, 0, 1}));
	EXPECT_TRUE(hoppingAmplitudeSet.getPhysicalIndex(2).equals({0, 0, 2}));
	EXPECT_TRUE(hoppingAmplitudeSet.getPhysicalIndex(3).equals({1, 1, 0}));
	EXPECT_TRUE(hoppingAmplitudeSet.getPhysicalIndex(4).equals({1, 1, 1}));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.getPhysicalIndex(-1);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.getPhysicalIndex(5);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(HoppingAmplitudeSet, getBasisSize){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	EXPECT_EQ(hoppingAmplitudeSet.getBasisSize(), -1);
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2}, {3, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {3, 4}, {1, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {3, 2}, {1, 2}));
	EXPECT_EQ(hoppingAmplitudeSet.getBasisSize(), -1);
	hoppingAmplitudeSet.construct();
	EXPECT_EQ(hoppingAmplitudeSet.getBasisSize(), 3);
}

TEST(HoppingAmplitudeSet, isProperSubspace){
	HoppingAmplitudeSet hoppingAmplitudeSet;

	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2, 3, 4, 5}, {1, 2, 3, 5, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2, 3, 5, 5}, {1, 2, 3, 4, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2, 3, 6, 5}, {1, 2, 3, 7, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 2, 3, 7, 5}, {1, 2, 3, 6, 5}));

	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {2, 2, 2, 4, 5}, {2, 2, 2, 4, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {2, 2, 2, 4, 5}, {2, 2, 2, 5, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {2, 2, 2, 5, 5}, {2, 2, 2, 4, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {2, 2, 2, 6, 5}, {2, 2, 2, 7, 5}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {2, 2, 2, 7, 5}, {2, 2, 2, 6, 5}));

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.isProperSubspace({1, -1});
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_TRUE(hoppingAmplitudeSet.isProperSubspace({1, 2}));
	EXPECT_TRUE(hoppingAmplitudeSet.isProperSubspace({2, 2}));
	EXPECT_TRUE(hoppingAmplitudeSet.isProperSubspace({1, 1}));
	EXPECT_TRUE(hoppingAmplitudeSet.isProperSubspace({1, 2, 3}));
	EXPECT_FALSE(hoppingAmplitudeSet.isProperSubspace({1, 2, 3, 4}));
	EXPECT_FALSE(hoppingAmplitudeSet.isProperSubspace({1, 2, 3, 4, 5}));
	EXPECT_TRUE(hoppingAmplitudeSet.isProperSubspace({1, 3}));
}

TEST(HoppingAmplitudeSet, getSubspaceIndices){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet.construct();

	IndexTree subspaceIndices = hoppingAmplitudeSet.getSubspaceIndices();

	EXPECT_EQ(subspaceIndices.getSize(), 2);

	IndexTree::Iterator iterator = subspaceIndices.begin();
	EXPECT_TRUE(iterator.getIndex().equals({0, 0}));
	iterator.searchNext();
	EXPECT_TRUE(iterator.getIndex().equals({1, 1}));
}

TEST(HoppingAmplitudeTree, construct){
	//Already tested through
	//HoppingAmplitudeSet::getBasisSize()
	//HoppingAmplitudeSet::getBasisIndex()
	//HoppingAmplitudeSet::getPhysicalIndex()
}

TEST(HoppingAmplitudeSet, getIsConstructed){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	EXPECT_FALSE(hoppingAmplitudeSet.getIsConstructed());
	hoppingAmplitudeSet.construct();
	EXPECT_TRUE(hoppingAmplitudeSet.getIsConstructed());
}

TEST(HoppingAmplitudeSet, getIndexList){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet.construct();

	std::vector<Index> indices = hoppingAmplitudeSet.getIndexList({0, 0, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({0, 0, 0}));
	EXPECT_TRUE(indices[1].equals({0, 0, 1}));
	EXPECT_TRUE(indices[2].equals({0, 0, 2}));

	indices = hoppingAmplitudeSet.getIndexList({1, 1, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({1, 1, 0}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));

	indices = hoppingAmplitudeSet.getIndexList({IDX_ALL, IDX_ALL, 1});
	EXPECT_TRUE(indices[0].equals({0, 0, 1}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));
}

TEST(HoppingAmplitudeSet, getFirstIndexInBlock){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet.construct();

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.getFirstIndexInBlock({0, 0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EQ(hoppingAmplitudeSet.getFirstIndexInBlock({0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getFirstIndexInBlock({0, 0}), 0);
	EXPECT_EQ(hoppingAmplitudeSet.getFirstIndexInBlock({1, 1}), 3);
}

TEST(HoppingAmplitudeSet, getLastIndexInBlock){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeSet.addHoppingAmplitude(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeSet.construct();

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeSet.getLastIndexInBlock({0, 0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EQ(hoppingAmplitudeSet.getLastIndexInBlock({0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeSet.getLastIndexInBlock({0, 0}), 2);
	EXPECT_EQ(hoppingAmplitudeSet.getLastIndexInBlock({1, 1}), 4);
}

TEST(HoppingAmplitudeSet, serialize){
	//Already tested through serializeToJSON
}

TEST(HoppingAmplitudeSet, getSizeInBytes){
	HoppingAmplitudeSet hoppingAmplitudeSet;
	EXPECT_TRUE(hoppingAmplitudeSet.getSizeInBytes() > 0);
}

//TODO
//...
/*TEST(HoppingAmplitudeTree, begin){
}*/

};
