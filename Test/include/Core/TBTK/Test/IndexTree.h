#include "TBTK/IndexTree.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(IndexTree, Constructor){
	//Not testable on its own.
}

//TODO
//...
//TEST(IndexedDataTree, SerializeToJSON){
	/**************************/
	/* Serializable elements. */
	/**************************/

/*	Streams::setStdMuteOut();

	//Setup Models that will work as test elements.
	Model model0;
	model0 << HoppingAmplitude(1, {0}, {0});
	model0.construct();

	Model model1;
	model1 << HoppingAmplitude(1, {0}, {0});
	model1 << HoppingAmplitude(1, {1}, {1});
	model1.construct();

	Model model2;
	model2 << HoppingAmplitude(1, {0}, {0});
	model2 << HoppingAmplitude(1, {1}, {1});
	model2 << HoppingAmplitude(1, {2}, {2});
	model2.construct();

	Model model3;
	model3 << HoppingAmplitude(1, {0}, {0});
	model3 << HoppingAmplitude(1, {1}, {1});
	model3 << HoppingAmplitude(1, {2}, {2});
	model3 << HoppingAmplitude(1, {3}, {3});
	model3.construct();

	//Actual tests
	IndexedDataTree<Model> indexedDataTree0;
	indexedDataTree0.add(model0, {1, 2, 3});
	indexedDataTree0.add(model1, {1, 2, 4});
	indexedDataTree0.add(model2, {1, 2, 3});
	indexedDataTree0.add(model3, {{1, 2, 5}, {1, 2, 3}});

	IndexedDataTree<Model> indexedDataTree1(
		indexedDataTree0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//Access existing elements.
	EXPECT_EQ(indexedDataTree1.get({1, 2, 3}).getBasisSize(), 3);
	EXPECT_EQ(indexedDataTree1.get({1, 2, 4}).getBasisSize(), 2);
	EXPECT_EQ(indexedDataTree1.get({{1, 2, 5}, {1, 2, 3}}).getBasisSize(), 4);

	//Access non-existing element
	EXPECT_THROW(indexedDataTree1.get({1, 2, 2}), ElementNotFoundException);
	EXPECT_THROW(indexedDataTree1.get({1, 1, 3}), ElementNotFoundException);*/

	/*************************************/
	/* Non/pseudo-serializable elements. */
	/*************************************/
/*	IndexedDataTree<int> indexedDataTree2;
	indexedDataTree2.add(1, {1, 2, 3});
	indexedDataTree2.add(2, {1, 2, 4});
	indexedDataTree2.add(3, {1, 2, 3});
	indexedDataTree2.add(4, {{1, 2, 5}, {1, 2, 3}});

	IndexedDataTree<int> indexedDataTree3(
		indexedDataTree2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//Access existing elements.
	EXPECT_EQ(indexedDataTree3.get({1, 2, 3}), 3);
	EXPECT_EQ(indexedDataTree3.get({1, 2, 4}), 2);
	EXPECT_EQ(indexedDataTree3.get({{1, 2, 5}, {1, 2, 3}}), 4);

	//Access non-existing element
	EXPECT_THROW(indexedDataTree3.get({1, 2, 2}), ElementNotFoundException);
	EXPECT_THROW(indexedDataTree3.get({1, 1, 3}), ElementNotFoundException);
}*/

TEST(IndexTree, Destructor){
	//Not testable on its own.
}

TEST(IndexTree, add){
	IndexTree indexTree;

	//Adding normal Index.
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});

	//Adding already existing Index.
	indexTree.add({1, 2, 3});

	//Adding compund Index.
	indexTree.add({{1, 2, 5}, {1, 2, 3}});
	indexTree.add({{1, 2, 5}, {1, 2, 4}});

	//Adding already existing compund Index.
	indexTree.add({{1, 2, 5}, {1, 2, 3}});

	//Fail to add compund Index if a non-compund Index with the same
	//structure already have been added.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({{1, 2, 3}, {1, 2, 3}});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with a compund Index structure that
	//already have been added.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2, 5});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with an Index structure that
	//already has been added (longer version).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with an Index structure that
	//already has been added (shorter version).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with a compund Index structure that
	//already has been added.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2, 5, 6});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index with negative Index (Except for the negative
	//IDX_SEPARATOR already tested above).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, -1, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(IndexTree, generateLinearMap){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({1, 3, 3});
	indexTree.generateLinearMap();

	EXPECT_EQ(indexTree.getSize(), 3);
}

//TODO
//...
TEST(IndexTree, getLinearIndex){
}

//TODO
//...
TEST(IndexTree, getPhysicalIndex){
}

//TODO
//...
TEST(IndexTree, getSize){
}

//TODO
//...
TEST(IndexTree, getSubindicesMatching){
}

//TODO
//...
TEST(IndexTree, getIndexList){
}

//TODO
//...
TEST(IndexTree, serialize){
}

//TODO
//...
TEST(IndexTree, Iterator){
}

};
