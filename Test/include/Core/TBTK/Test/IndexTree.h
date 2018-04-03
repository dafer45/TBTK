#include "TBTK/IndexException.h"
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

	//Adding Index with wildcard.
	indexTree.add({2, IDX_ALL, 3});

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

	//Fail to add compund Index if it clashes with a compund Index
	//structure that have already been added.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({{1, 2, 5}});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with a compund Index structure that
	//have already been added.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2, 5});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with an Index structure that has
	//already been added (longer version).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, 2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index if it clashes with an Index structure that has
	//already been added (shorter version).
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

	//Fail to add Index with wildcard Index in a position that clashes with
	//a non-wildcard Index structure
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({1, IDX_ALL, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index that that has a specific subindex in a position
	//previously marked as a wildcard.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({2, 1, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add Index that has a similar Index structure to an already
	//added Index, but which differs in wildcard type at the shared
	//wildcard position.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({2, IDX_SPIN, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Succed at adding an Index that has a similar Index structure to an already
	//added Index up to a wildcard Index if it has the same wildcard type.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.add({2, IDX_ALL, 4});
			std::cerr << "Test completed.";
			exit(0);
		},
		::testing::ExitedWithCode(0),
		"Test completed."
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
	IndexTree indexTree;
	indexTree.add({0, 0, 0});
	indexTree.add({0, 0, 1});
	indexTree.add({0, 1, 0});
	indexTree.add({1, 0, 0});
	indexTree.add({0, 1, 2, 3});
	indexTree.add({{1, 2, 3}, {4, 5, 6}});
	indexTree.add({3, IDX_SPIN, 4});
	indexTree.add({4, IDX_ALL, 5});
	indexTree.generateLinearMap();

	///////////////////////
	// Strict match mode //
	///////////////////////

	//Test normal requests.
	EXPECT_EQ(indexTree.getLinearIndex({0, 0, 0}), 0);
	EXPECT_EQ(indexTree.getLinearIndex({0, 0, 1}), 1);
	EXPECT_EQ(indexTree.getLinearIndex({0, 1, 0}), 2);
	EXPECT_EQ(indexTree.getLinearIndex({0, 1, 2, 3}), 3);
	EXPECT_EQ(indexTree.getLinearIndex({1, 0, 0}), 4);
	EXPECT_EQ(indexTree.getLinearIndex({{1, 2, 3}, {4, 5, 6}}), 5);
	EXPECT_EQ(indexTree.getLinearIndex({3, IDX_SPIN, 4}), 6);
	EXPECT_EQ(indexTree.getLinearIndex({4, IDX_ALL, 5}), 7);

	//Fail to request wildcard Indices using wildcards replaced by zero.
	//(Zero is used internally for wildcard indices, so this is meant to
	//test that the internal structure does not leak through).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.getLinearIndex({3, 0, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to request wildcard Indices using wrong wildcard. replaced by zero.
	//(Zero is used internally for wildcard indices, so this is meant to
	//test that the internal structure does not leak through).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.getLinearIndex({3, IDX_ALL, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Throw IndexException when accessing non-existing Index.
	EXPECT_THROW(indexTree.getLinearIndex({9, 8, 7}), IndexException);
	EXPECT_THROW(indexTree.getLinearIndex({0, 1, 2, 0}), IndexException);

	//Return -1 when accessing non-existing Index.
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{9, 8, 7},
			IndexTree::SearchMode::StrictMatch,
			true
		),
		-1
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 1, 2, 0},
			IndexTree::SearchMode::StrictMatch,
			true
		),
		-1
	);

	////////////////////////
	// MatchWildcard mode //
	////////////////////////

	//Test normal requests.
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 0, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		0
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 0, 1},
			IndexTree::SearchMode::MatchWildcards
		),
		1
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 1, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		2
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 1, 2, 3},
			IndexTree::SearchMode::MatchWildcards
		),
		3
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{1, 0, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		4
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{{1, 2, 3}, {4, 5, 6}},
			IndexTree::SearchMode::MatchWildcards
		),
		5
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{3, IDX_SPIN, 4},
			IndexTree::SearchMode::MatchWildcards
		),
		6
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{4, IDX_ALL, 5},
			IndexTree::SearchMode::MatchWildcards
		),
		7
	);

	//Test wildcard requests.
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{3, 1, 4},
			IndexTree::SearchMode::MatchWildcards
		),
		6
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{4, 2, 5},
			IndexTree::SearchMode::MatchWildcards
		),
		7
	);

	//Fail to request wildcard Indices using wrong wildcard. replaced by zero.
	//(Zero is used internally for wildcard indices, so this is meant to
	//test that the internal structure does not leak through).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.getLinearIndex(
				{3, IDX_ALL, 4},
				IndexTree::SearchMode::MatchWildcards
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Throw IndexException when accessing non-existing Index.
	EXPECT_THROW(
		indexTree.getLinearIndex(
			{9, 8, 7},
			IndexTree::SearchMode::MatchWildcards
		),
		IndexException
	);
	EXPECT_THROW(
		indexTree.getLinearIndex(
			{0, 1, 2, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		IndexException
	);

	//Return -1 when accessing non-existing Index.
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{9, 8, 7},
			IndexTree::SearchMode::MatchWildcards,
			true
		),
		-1
	);
	EXPECT_EQ(
		indexTree.getLinearIndex(
			{0, 1, 2, 0},
			IndexTree::SearchMode::MatchWildcards,
			true
		),
		-1
	);
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
