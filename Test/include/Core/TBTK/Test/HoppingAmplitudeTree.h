#include "TBTK/HoppingAmplitudeTree.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(HoppingAmplitudeTree, Constructor){
	//Not testable on its own
}

TEST(HoppingAmplitudeTree, ConstructorCapacity){
	//Not testable on its own
}

TEST(HoppingAmplitudeTree, SerializeToJSON){
	HoppingAmplitudeTree hoppingAmplitudeTree0;
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree0.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree0.generateBasisIndices();

	HoppingAmplitudeTree hoppingAmplitudeTree1(
		hoppingAmplitudeTree0.serialize(Serializeable::Mode::JSON),
		Serializeable::Mode::JSON
	);

	EXPECT_EQ(hoppingAmplitudeTree1.getBasisSize(), 5);

	std::vector<Index> indices = hoppingAmplitudeTree1.getIndexList({0, 0, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({0, 0, 0}));
	EXPECT_TRUE(indices[1].equals({0, 0, 1}));
	EXPECT_TRUE(indices[2].equals({0, 0, 2}));

	indices = hoppingAmplitudeTree1.getIndexList({1, 1, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({1, 1, 0}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));

	indices = hoppingAmplitudeTree1.getIndexList({IDX_ALL, IDX_ALL, 1});
	EXPECT_TRUE(indices[0].equals({0, 0, 1}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));
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

TEST(HoppingAmplitudeTree, getSubTree){
	std::string errorMessage = "getSubTree() failed.";

	HoppingAmplitudeTree hoppingAmplitudeTree;

	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 4}, {1, 2, 3, 4}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 4}, {1, 2, 3, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 5}, {1, 2, 3, 4}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 6}, {1, 2, 3, 7}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 7}, {1, 2, 3, 6}));

	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 4}, {2, 2, 2, 4}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 4}, {2, 2, 2, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 5}, {2, 2, 2, 4}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 6}, {2, 2, 2, 7}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 7}, {2, 2, 2, 6}));

	hoppingAmplitudeTree.generateBasisIndices();

	EXPECT_EXIT(
		{
			const HoppingAmplitudeTree *hoppingAmplitudeTree0 = hoppingAmplitudeTree.getSubTree({1, 2});
			const HoppingAmplitudeTree *hoppingAmplitudeTree1 = hoppingAmplitudeTree.getSubTree({2, 2});
			const HoppingAmplitudeTree *hoppingAmplitudeTree2 = hoppingAmplitudeTree.getSubTree({1, 1});
			const HoppingAmplitudeTree *hoppingAmplitudeTree3 = hoppingAmplitudeTree.getSubTree({1, 3});

			HoppingAmplitudeTree::Iterator iterator = hoppingAmplitudeTree0->begin();
			int counter = 0;
			const HoppingAmplitude *hoppingAmplitude;
			while((hoppingAmplitude = iterator.getHA())){
				counter++;
				if(!hoppingAmplitude->getFromIndex().equals({1, 2, 3, IDX_ALL}, true))
					exit(1);

				iterator.searchNextHA();
			}
			if(counter != 5)
				exit(1);

			iterator = hoppingAmplitudeTree1->begin();
			counter = 0;
			while((hoppingAmplitude = iterator.getHA())){
				counter++;
				if(!hoppingAmplitude->getFromIndex().equals({2, 2, 2, IDX_ALL}, true))
					exit(1);

				iterator.searchNextHA();
			}
			if(counter != 5)
				exit(1);

			iterator = hoppingAmplitudeTree2->begin();
			counter = 0;
			while((hoppingAmplitude = iterator.getHA())){
				counter++;
				iterator.searchNextHA();
			}
			if(counter != 0)
				exit(1);

			iterator = hoppingAmplitudeTree3->begin();
			counter = 0;
			while((hoppingAmplitude = iterator.getHA())){
				counter++;
				iterator.searchNextHA();
			}
			if(counter != 0)
				exit(1);

			std::cerr << "Test completed.";
			exit(0);
		},
		::testing::ExitedWithCode(0),
		"Test completed."
	);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getSubTree({1, -1});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(HoppingAmplitudeTree, isProperSubspace){
	std::string errorMessage = "getSubTree() failed.";

	HoppingAmplitudeTree hoppingAmplitudeTree;

	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 4, 5}, {1, 2, 3, 5, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 5, 5}, {1, 2, 3, 4, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 6, 5}, {1, 2, 3, 7, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 2, 3, 7, 5}, {1, 2, 3, 6, 5}));

	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 4, 5}, {2, 2, 2, 4, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 4, 5}, {2, 2, 2, 5, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 5, 5}, {2, 2, 2, 4, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 6, 5}, {2, 2, 2, 7, 5}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {2, 2, 2, 7, 5}, {2, 2, 2, 6, 5}));

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.isProperSubspace({1, -1});
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_TRUE(hoppingAmplitudeTree.isProperSubspace({1, 2})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitudeTree.isProperSubspace({2, 2})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitudeTree.isProperSubspace({1, 1})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitudeTree.isProperSubspace({1, 2, 3})) << errorMessage;
	EXPECT_FALSE(hoppingAmplitudeTree.isProperSubspace({1, 2, 3, 4})) << errorMessage;
	EXPECT_FALSE(hoppingAmplitudeTree.isProperSubspace({1, 2, 3, 4, 5})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitudeTree.isProperSubspace({1, 3}));
}

TEST(HoppingAmplitudeTree, getSubspaceIndices){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree.generateBasisIndices();

	IndexTree subspaceIndices = hoppingAmplitudeTree.getSubspaceIndices();

	EXPECT_EQ(subspaceIndices.getSize(), 2);

	IndexTree::Iterator iterator = subspaceIndices.begin();
	EXPECT_TRUE(iterator.getIndex().equals({0, 0}));
	iterator.searchNext();
	EXPECT_TRUE(iterator.getIndex().equals({1, 1}));
}

TEST(HoppingAmplitudeTree, getFirstIndexInSubspace){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree.generateBasisIndices();

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getFirstIndexInSubspace({0, 0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EQ(hoppingAmplitudeTree.getFirstIndexInSubspace({0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getFirstIndexInSubspace({0, 0}), 0);
	EXPECT_EQ(hoppingAmplitudeTree.getFirstIndexInSubspace({1, 1}), 3);
}

TEST(HoppingAmplitudeTree, getLastIndexInSubspace){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree.generateBasisIndices();

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getLastIndexInSubspace({0, 0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	EXPECT_EQ(hoppingAmplitudeTree.getLastIndexInSubspace({0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getLastIndexInSubspace({0, 0}), 2);
	EXPECT_EQ(hoppingAmplitudeTree.getLastIndexInSubspace({1, 1}), 4);
}

TEST(HoppingAmplitudeTree, getHAs){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree.generateBasisIndices();

	const std::vector<HoppingAmplitude> *hoppingAmplitudes = hoppingAmplitudeTree.getHAs({0, 0, 1});
	EXPECT_EQ(hoppingAmplitudes->size(), 2);
	for(unsigned int n = 0; n < hoppingAmplitudes->size(); n++){
		const HoppingAmplitude &hoppingAmplitude = hoppingAmplitudes->at(n);

		EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({0, 0, 1}));
		EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({0, 0, 1}) || hoppingAmplitude.getToIndex().equals({0, 0, 2}));
	}

	hoppingAmplitudes = hoppingAmplitudeTree.getHAs({0, 0, 2});
	EXPECT_EQ(hoppingAmplitudes->size(), 1);
	for(unsigned int n = 0; n < hoppingAmplitudes->size(); n++){
		const HoppingAmplitude &hoppingAmplitude = hoppingAmplitudes->at(n);

		EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({0, 0, 2}));
		EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({0, 0, 1}));
	}
}

TEST(HoppingAmplitudeTree, getBasisIndex){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));

	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 0}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 1}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 2}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({1, 1, 0}), -1);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({1, 1, 1}), -1);

	hoppingAmplitudeTree.generateBasisIndices();

	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 0}), 0);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 1}), 1);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({0, 0, 2}), 2);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({1, 1, 0}), 3);
	EXPECT_EQ(hoppingAmplitudeTree.getBasisIndex({1, 1, 1}), 4);
}

TEST(HoppingAmplitudeTree, getPhysicsIndex){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getPhysicalIndex(0);
		},
		::testing::ExitedWithCode(1),
		""
	);

	hoppingAmplitudeTree.generateBasisIndices();

	EXPECT_TRUE(hoppingAmplitudeTree.getPhysicalIndex(0).equals({0, 0, 0}));
	EXPECT_TRUE(hoppingAmplitudeTree.getPhysicalIndex(1).equals({0, 0, 1}));
	EXPECT_TRUE(hoppingAmplitudeTree.getPhysicalIndex(2).equals({0, 0, 2}));
	EXPECT_TRUE(hoppingAmplitudeTree.getPhysicalIndex(3).equals({1, 1, 0}));
	EXPECT_TRUE(hoppingAmplitudeTree.getPhysicalIndex(4).equals({1, 1, 1}));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getPhysicalIndex(-1);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitudeTree.getPhysicalIndex(5);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(HoppingAmplitudeTree, generateBasisIndices){
	//Already tested through
	//HoppingAmplitudeTree::getBasisSize()
	//HoppingAmplitudeTree::getBasisIndex()
	//HoppingAmplitudeTree::getPhysicalIndex()
}

TEST(HoppingAmplitudeTree, getIndexList){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 1}, {0, 0, 2}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {0, 0, 2}, {0, 0, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 0}, {1, 1, 1}));
	hoppingAmplitudeTree.add(HoppingAmplitude(1, {1, 1, 1}, {1, 1, 0}));
	hoppingAmplitudeTree.generateBasisIndices();

	std::vector<Index> indices = hoppingAmplitudeTree.getIndexList({0, 0, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({0, 0, 0}));
	EXPECT_TRUE(indices[1].equals({0, 0, 1}));
	EXPECT_TRUE(indices[2].equals({0, 0, 2}));

	indices = hoppingAmplitudeTree.getIndexList({1, 1, IDX_ALL});
	EXPECT_TRUE(indices[0].equals({1, 1, 0}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));

	indices = hoppingAmplitudeTree.getIndexList({IDX_ALL, IDX_ALL, 1});
	EXPECT_TRUE(indices[0].equals({0, 0, 1}));
	EXPECT_TRUE(indices[1].equals({1, 1, 1}));
}

//TODO
//...
/*TEST(HoppingAmplitudeTree, sort){
}*/

//TODO
//...
/*TEST(HoppingAmplitudeTree, print){
}*/

TEST(HoppingAmplitudeTree, getSizeInBytes){
	HoppingAmplitudeTree hoppingAmplitudeTree;
	EXPECT_TRUE(hoppingAmplitudeTree.getSizeInBytes() > 0);
}

//TODO
//...
/*TEST(HoppingAmplitudeTree, begin){
}*/

TEST(HoppingAmplitudeTree, serialize){
	//Already tested through serializeToJSON
}

};
