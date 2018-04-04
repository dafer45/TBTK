#include "TBTK/IndexException.h"
#include "TBTK/IndexTree.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(IndexTree, Constructor){
	//Not testable on its own.
}

TEST(IndexTree, SerializeToJSON){
	IndexTree indexTree0;
	indexTree0.add({0, 0, 0});
	indexTree0.add({0, 0, 1});
	indexTree0.add({0, 1, 0});
	indexTree0.add({1, 0, 0});
	indexTree0.add({0, 1, 2, 3});
	indexTree0.add({{1, 2, 3}, {4, 5, 6}});
	indexTree0.add({3, IDX_SPIN, 4});
	indexTree0.add({4, IDX_ALL, 5});
	indexTree0.generateLinearMap();

	IndexTree indexTree1(
		indexTree0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	///////////////////////
	// Strict match mode //
	///////////////////////

	//Test normal requests.
	EXPECT_EQ(indexTree1.getLinearIndex({0, 0, 0}), 0);
	EXPECT_EQ(indexTree1.getLinearIndex({0, 0, 1}), 1);
	EXPECT_EQ(indexTree1.getLinearIndex({0, 1, 0}), 2);
	EXPECT_EQ(indexTree1.getLinearIndex({0, 1, 2, 3}), 3);
	EXPECT_EQ(indexTree1.getLinearIndex({1, 0, 0}), 4);
	EXPECT_EQ(indexTree1.getLinearIndex({{1, 2, 3}, {4, 5, 6}}), 5);
	EXPECT_EQ(indexTree1.getLinearIndex({3, IDX_SPIN, 4}), 6);
	EXPECT_EQ(indexTree1.getLinearIndex({4, IDX_ALL, 5}), 7);

	////////////////////////
	// MatchWildcard mode //
	////////////////////////

	//Test normal requests.
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{0, 0, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		0
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{0, 0, 1},
			IndexTree::SearchMode::MatchWildcards
		),
		1
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{0, 1, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		2
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{0, 1, 2, 3},
			IndexTree::SearchMode::MatchWildcards
		),
		3
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{1, 0, 0},
			IndexTree::SearchMode::MatchWildcards
		),
		4
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{{1, 2, 3}, {4, 5, 6}},
			IndexTree::SearchMode::MatchWildcards
		),
		5
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{3, IDX_SPIN, 4},
			IndexTree::SearchMode::MatchWildcards
		),
		6
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{4, IDX_ALL, 5},
			IndexTree::SearchMode::MatchWildcards
		),
		7
	);

	//Test wildcard requests.
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{3, 1, 4},
			IndexTree::SearchMode::MatchWildcards
		),
		6
	);
	EXPECT_EQ(
		indexTree1.getLinearIndex(
			{4, 2, 5},
			IndexTree::SearchMode::MatchWildcards
		),
		7
	);
}

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

TEST(IndexTree, getPhysicalIndex){
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

	//Normal access.
	EXPECT_TRUE(indexTree.getPhysicalIndex(0).equals({0, 0, 0}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(1).equals({0, 0, 1}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(2).equals({0, 1, 0}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(3).equals({0, 1, 2, 3}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(4).equals({1, 0, 0}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(5).equals({{1, 2, 3}, {4, 5, 6}}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(6).equals({{3, IDX_SPIN, 4}}));
	EXPECT_TRUE(indexTree.getPhysicalIndex(7).equals({{4, IDX_ALL, 5}}));

	//Index out of bound.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.getPhysicalIndex(-1);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexTree.getPhysicalIndex(8);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(IndexTree, getSize){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({1, 3, 3});

	EXPECT_EQ(indexTree.getSize(), -1);
	indexTree.generateLinearMap();
	EXPECT_EQ(indexTree.getSize(), 3);
}

TEST(IndexTree, getSubindicesMatching){
	IndexTree indexTree;
	indexTree.add({1, 2, 1});
	indexTree.add({1, 2, 3});
	indexTree.add({2, IDX_ALL, 4, IDX_SPIN, 6, IDX_ALL});
	indexTree.generateLinearMap();

	//Match agains normal Index in StrictMatch mode.
	std::vector<unsigned int> subindices0
		= indexTree.getSubindicesMatching(
			1,
			{1, 2, 1},
			IndexTree::SearchMode::StrictMatch
		);
	EXPECT_EQ(subindices0.size(), 2);
	EXPECT_EQ(subindices0[0], 0);
	EXPECT_EQ(subindices0[1], 2);

	//Match agains normal Index in MatchWildcards mode.
	std::vector<unsigned int> subindices1
		= indexTree.getSubindicesMatching(
			1,
			{1, 2, 1},
			IndexTree::SearchMode::MatchWildcards
		);
	EXPECT_EQ(subindices1.size(), 2);
	EXPECT_EQ(subindices1[0], 0);
	EXPECT_EQ(subindices1[1], 2);

	//Match against wildcard indices in StrictMatch mode.
	std::vector<unsigned int> subindices2
		= indexTree.getSubindicesMatching(
			IDX_ALL,
			{2, IDX_ALL, 4, IDX_SPIN, 6, IDX_ALL},
			IndexTree::SearchMode::StrictMatch
		);
	EXPECT_EQ(subindices2.size(), 2);
	EXPECT_EQ(subindices2[0], 1);
	EXPECT_EQ(subindices2[1], 5);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			std::vector<unsigned int> subindices3
				= indexTree.getSubindicesMatching(
					IDX_ALL,
					{2, 3, 4, 5, 6, 7},
					IndexTree::SearchMode::StrictMatch
				);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Match against wildcard indices in MatchWildcards mode.
	std::vector<unsigned int> subindices4
		= indexTree.getSubindicesMatching(
			IDX_ALL,
			{2, 3, 4, 5, 6, 7},
			IndexTree::SearchMode::MatchWildcards
		);
	EXPECT_EQ(subindices4.size(), 2);
	EXPECT_EQ(subindices4[0], 1);
	EXPECT_EQ(subindices4[1], 5);
}

//TODO
//...
TEST(IndexTree, getIndexList){
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

	//Match without IDX_ALL wildcards
	std::vector<Index> indexList0 = indexTree.getIndexList({0, 0, 0});
	std::vector<Index> indexList1 = indexTree.getIndexList({0, 1, 0});
	std::vector<Index> indexList2 = indexTree.getIndexList(
		{{1, 2, 3}, {4, 5, 6}}
	);
	std::vector<Index> indexList3 = indexTree.getIndexList(
		{3, IDX_SPIN, 4}
	);
	ASSERT_EQ(indexList0.size(), 1);
	ASSERT_EQ(indexList1.size(), 1);
	ASSERT_EQ(indexList2.size(), 1);
	ASSERT_EQ(indexList3.size(), 1);
	EXPECT_TRUE(indexList0[0].equals({0, 0, 0}));
	EXPECT_TRUE(indexList1[0].equals({0, 1, 0}));
	EXPECT_TRUE(indexList2[0].equals({{1, 2, 3}, {4, 5, 6}}));
	EXPECT_TRUE(indexList3[0].equals({3, IDX_SPIN, 4}));

	//Match with wildcard
	std::vector<Index> indexList4 = indexTree.getIndexList(
		{0, IDX_ALL, IDX_ALL}
	);
	std::vector<Index> indexList5 = indexTree.getIndexList(
		{{IDX_ALL, IDX_ALL, IDX_ALL}, {IDX_ALL, IDX_ALL, IDX_ALL}}
	);
	ASSERT_EQ(indexList4.size(), 3);
	ASSERT_EQ(indexList5.size(), 1);
	EXPECT_TRUE(indexList4[0].equals({0, 0, 0}));
	EXPECT_TRUE(indexList4[1].equals({0, 0, 1}));
	EXPECT_TRUE(indexList4[2].equals({0, 1, 0}));
	EXPECT_TRUE(indexList5[0].equals({{1, 2, 3}, {4, 5, 6}}));

	//Match against IDX_ALL wildcard in the IndexTree by specifying IDX_ALL
	//also in the pattern.
	std::vector<Index> indexList6 = indexTree.getIndexList(
		{4, IDX_ALL, 5}
	);
	ASSERT_EQ(indexList6.size(), 1);
	EXPECT_TRUE(indexList6[0].equals({4, IDX_ALL, 5}));

	//Do not match against IDX_ALL wildcard in the IndexTree when not
	//specifying IDX_ALL also in the pattern.
	std::vector<Index> indexList7 = indexTree.getIndexList(
		{4, 0, 5}
	);
	ASSERT_EQ(indexList7.size(), 0);
}

TEST(IndexTree, serialize){
	//Already tested through SerializeToJSON.
}

//TODO
//...
TEST(IndexTree, Iterator){
}

};
