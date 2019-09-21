#include "TBTK/Index.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Core.Index.Construction.1 2019-09-19
TEST(Index, ConstructorEmpty){
	Index index;
	EXPECT_EQ(index.getSize(), 0) << "Empty Index does not have size 0.";
}

//TBTKFeature Core.Index.Construction.2 2019-09-19
TEST(Index, ConstructorInitializerListInt){
	Index index({1, 2, 3});
	EXPECT_EQ(index.getSize(), 3) << "Index({1, 2, 3}) does not have size 3.";
	EXPECT_EQ(index[0], 1) << "Index({1, 2, 3}) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index({1, 2, 3}) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index({1, 2, 3}) does not have '3' as third subindex.";
}

//TBTKFeature Core.Index.Construction.2 2019-09-19
TEST(Index, ConstructorInitializerListUnsignedInt){
	Index index({(unsigned int)1, (unsigned int)2, (unsigned int)3});
	EXPECT_EQ(index.getSize(), 3) << "Index({1, 2, 3}) does not have size 3.";
	EXPECT_EQ(index[0], 1) << "Index({1, 2, 3}) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index({1, 2, 3}) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index({1, 2, 3}) does not have '3' as third subindex.";
}

//TBTKFeature Core.Index.Construction.3.C++ 2019-09-19
TEST(Index, ConstructorVectorInt){
	std::vector<Subindex> myVector({1, 2, 3});
	Index index(myVector);
	EXPECT_EQ(index.getSize(), 3) << "Index(std::vector<int>({1, 2, 3})) does not have size 3.";
	EXPECT_EQ(index[0], 1)<< "Index(std::vector<int>({1, 2, 3})) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index(std::vector<int>({1, 2, 3})) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index(std::vector<int>({1, 2, 3})) does not have '3' as third subindex.";
}

/*TEST(Index, ConstructorVectorUnsignedInt){
	std::vector<unsigned int> myVector({1, 2, 3});
	Index index(myVector);
	EXPECT_EQ(index.getSize(), 3) << "Index(std::vector<int>({1, 2, 3})) does not have size 3.";
	EXPECT_EQ(index[0], 1)<< "Index(std::vector<int>({1, 2, 3})) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index(std::vector<int>({1, 2, 3})) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index(std::vector<int>({1, 2, 3})) does not have '3' as third subindex.";
}*/

//TBTKFeature Core.Index.Copy.1 2019-09-19
TEST(Index, CopyConstructor){
	Index index({1, 2, 3});
	Index indexCopy = index;
	EXPECT_EQ(indexCopy.getSize(), 3) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[0], 1) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[1], 2) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[2], 3) << "Copy constructor failed.";
}

//TBTKFeature Core.Index.Construction.4 2019-09-19
TEST(Index, ConstructorConcatenationInitializerList){
	std::string errorMessage = "Index concatenation filed.";

	Index head({1, 2});
	Index tail({3});
	Index index(head, tail);
	EXPECT_EQ(index.getSize(), 3) << errorMessage;
	EXPECT_EQ(index[0], 1) << errorMessage;
	EXPECT_EQ(index[1], 2) << errorMessage;
	EXPECT_EQ(index[2], 3) << errorMessage;
}

//TBTKFeature Core.Index.Construction.5.C++ 2019-09-19
TEST(Index, ConstructorConcatenationVector){
	std::string errorMessage = "Index concatenation filed.";

	std::vector<Subindex> head({1, 2});
	std::vector<Subindex> tail({3});
	Index index(head, tail);
	EXPECT_EQ(index.getSize(), 3) << errorMessage;
	EXPECT_EQ(index[0], 1) << errorMessage;
	EXPECT_EQ(index[1], 2) << errorMessage;
	EXPECT_EQ(index[2], 3) << errorMessage;
}

TEST(Index, ConstructorCompundInitializerList){
	std::string errorMessage = "Compund Index construction failed.";

	//TBTKFeature Core.Index.Construction.6 2019-09-19
	//Using an initializer list.
	Index index0({{1}, {2, 3}, {4, 5, 6}});
	EXPECT_EQ(index0.getSize(), 8) << errorMessage;
	EXPECT_EQ(index0[0], 1) << errorMessage;
	EXPECT_EQ(index0[1], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index0[2], 2) << errorMessage;
	EXPECT_EQ(index0[3], 3) << errorMessage;
	EXPECT_EQ(index0[4], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index0[5], 4) << errorMessage;
	EXPECT_EQ(index0[6], 5) << errorMessage;
	EXPECT_EQ(index0[7], 6) << errorMessage;

	//TBTKFeature Core.Index.Construction.7.C++ 2019-09-19
	//Using a vector.
	Index index1(std::vector<Index>({{1}, {2, 3}, {4, 5, 6}}));
	EXPECT_EQ(index1.getSize(), 8) << errorMessage;
	EXPECT_EQ(index1[0], 1) << errorMessage;
	EXPECT_EQ(index1[1], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index1[2], 2) << errorMessage;
	EXPECT_EQ(index1[3], 3) << errorMessage;
	EXPECT_EQ(index1[4], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index1[5], 4) << errorMessage;
	EXPECT_EQ(index1[6], 5) << errorMessage;
	EXPECT_EQ(index1[7], 6) << errorMessage;
}

//TBTKFeature Core.Index.Construction.8.C++ 2019-09-19
TEST(Index, ConstructorCompundVector){
	std::string errorMessage = "Compund Index construction failed.";

	std::vector<std::vector<Subindex>> indices;
	indices.push_back({1});
	indices.push_back({2, 3});
	indices.push_back({4, 5, 6});
	Index index(indices);
	EXPECT_EQ(index.getSize(), 8) << errorMessage;
	EXPECT_EQ(index[0], 1) << errorMessage;
	EXPECT_EQ(index[1], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index[2], 2) << errorMessage;
	EXPECT_EQ(index[3], 3) << errorMessage;
	EXPECT_EQ(index[4], IDX_SEPARATOR) << errorMessage;
	EXPECT_EQ(index[5], 4) << errorMessage;
	EXPECT_EQ(index[6], 5) << errorMessage;
	EXPECT_EQ(index[7], 6) << errorMessage;
}

//TBTKFeature Core.Index.Construction.9 2019-09-19
TEST(Index, ConstructorString){
	std::string errorMessage = "Index construction from string failed.";

	Index index0("{1, 2, 3}");
	EXPECT_TRUE(index0.equals({1, 2, 3})) << errorMessage;
}

//TBTKFeature Core.Index.Serialization.1 2019-09-19
TEST(Index, SerializeToJSON){
	Index index0({1, 2, 3});
	Index index1(
		index0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_TRUE(index1.equals(index0)) << "JSON serialization failed.";
}

TEST(Index, equals){
	std::string errorMessage = "Index comparison failed.";

	//TBTKFeature Core.Index.Equals.1 2019-09-19
	EXPECT_TRUE(Index({1, 2, 3}).equals(Index({1, 2, 3}))) << errorMessage;
	//TBTKFeature Core.Index.Equals.2 2019-09-19
	EXPECT_FALSE(Index({1, 2, 2}).equals(Index({1, 2, 3}))) << errorMessage;
	//TBTKFeature Core.Index.Equals.3 2019-09-19
	EXPECT_FALSE(Index({1, 2, IDX_ALL}).equals(Index({1, 2, 3}))) << errorMessage;
	//TBTKFeature Core.Index.Equals.4 2019-09-21
	EXPECT_TRUE(Index({1, 2, IDX_ALL}).equals(Index({1, 2, 3}), true)) << errorMessage;
	//TBTKFeature Core.Index.Equals.5 2019-09-19
	EXPECT_TRUE(Index({1, -1, 3}).equals(Index({1, -1, 3}))) << errorMessage;
	//TBTKFeature Core.Index.Equals.6 2019-09-19
	EXPECT_TRUE(Index({{1, 2, 3}, {4, 5, 6}}).equals(Index({{1, 2, 3}, {4, 5, 6}}))) << errorMessage;
	//TBTKFeature Core.Index.Equals.7 2019-09-19
	EXPECT_TRUE(
		Index(
			{0, IDX_ALL_(0), IDX_ALL_(0), IDX_ALL_(1), 3, IDX_ALL_(1)}
		).equals(Index({0, 1, 1, 2, 3, 2}), true)
	) << errorMessage;
	//TBTKFeature Core.Index.Equals.8 2019-09-19
	EXPECT_TRUE(
		Index(
			Index({0, 1, 1, 2, 3, 2})
		).equals(
			Index({
				0,
				IDX_ALL_(0),
				IDX_ALL_(0),
				IDX_ALL_(1),
				3,
				IDX_ALL_(1)
			}),
			true
		)
	) << errorMessage;
	//TBTKFeature Core.Index.Equals.9 2019-09-19
	EXPECT_FALSE(
		Index(
			{0, IDX_ALL_(0), IDX_ALL_(0), IDX_ALL_(1), 3, IDX_ALL_(1)}
		).equals(Index({0, 1, 1, 2, 3, 2}))
	) << errorMessage;
	//TBTKFeature Core.Index.Equals.10 2019-09-19
	EXPECT_FALSE(
		Index(
			Index({0, 1, 1, 2, 3, 2})
		).equals(
			Index({
				0,
				IDX_ALL_(0),
				IDX_ALL_(0),
				IDX_ALL_(1),
				3,
				IDX_ALL_(1)
			})
		)
	) << errorMessage;
}

//TBTKFeature Core.Index.at.1 2019-09-19
//TBTKFeature Core.Index.operator[].1.C++ 2019-09-19
TEST(Index, atAndSubscriptOperator){
	std::string errorMessage = "at() and operator[] gives different results.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.at(0), index[0]) << errorMessage;
	EXPECT_EQ(index.at(1), index[1]) << errorMessage;
	EXPECT_EQ(index.at(2), index[2]) << errorMessage;
}

//TBTKFeature Core.Index.at.2.C++ 2019-09-19
//TBTKFeature Core.Index.operator[].2.C++ 2019-09-19
TEST(Index, atAndSubscriptOperatorConst){
	std::string errorMessage = "at() and operator[] gives different results.";

	const Index index({1, 2, 3});
	EXPECT_EQ(index.at(0), index[0]) << errorMessage;
	EXPECT_EQ(index.at(1), index[1]) << errorMessage;
	EXPECT_EQ(index.at(2), index[2]) << errorMessage;
}

TEST(Index, getSize){
	std::string errorMessage = "getSize() failed.";

	//TBTKFeature Core.Index.getSize.1 2019-09-19
	EXPECT_EQ(Index().getSize(), 0) << errorMessage;
	//TBTKFeature Core.Index.getSize.2 2019-09-19
	EXPECT_EQ(Index({1}).getSize(), 1) << errorMessage;
	//TBTKFeature Core.Index.getSize.3 2019-09-19
	EXPECT_EQ(Index({1, 2, 3}).getSize(), 3) << errorMessage;
	//TBTKFeature Core.Index.getSize.4 2019-09-19
	EXPECT_EQ(Index({1, 2}, {3, 4}).getSize(), 4) << errorMessage;
	//TBTKFeature Core.Index.getSize.5 2019-09-19
	EXPECT_EQ(Index({{1, 2}, {3, 4}}).getSize(), 5) << errorMessage;
}

//TBTKFeature Core.Index.pushBack.1 2019-09-19
TEST(Index, pushBack){
	Index index;
	index.pushBack(1);
	index.pushBack(2);
	index.pushBack(3);
	EXPECT_TRUE(index.equals({1, 2, 3})) << "push_back failed.";
}

//TBTKFeature Core.Index.popFront.1 2019-09-19
TEST(Index, popFront){
	std::string errorMessage = "popFront() failed.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.popFront(), 1) << errorMessage;
	EXPECT_TRUE(index.equals({2, 3})) << errorMessage;
}

//TBTKFeature Core.Index.popBack.1 2019-09-19
TEST(Index, popBack){
	std::string errorMessage = "popBack() failed.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.popBack(), 3) << errorMessage;
	EXPECT_TRUE(index.equals({1, 2})) << errorMessage;
}

//TBTKFeature Core.Index.getUnitRange.1 2019-09-19
TEST(Index, getUnitRange){
	Index index({1, 2, 3});
	EXPECT_TRUE(index.getUnitRange().equals({1, 1, 1})) << "getUnitRange() failed.";
}

//TBTKFeature Core.Index.getSubIndex.1 2019-09-21
TEST(Index, getSubIndex){
	Index index({1, 2, 3, 4, 5, 6, 7, 8, 9});
	EXPECT_TRUE(index.getSubIndex(3, 5).equals({4, 5, 6})) << "getSubIndex() failed.";
}

//TBTKFeature Core.Index.split.1 2019-09-19
TEST(Index, split){
	Index index({{1, 2, 3}, {4, 5}, {6, 7, 8}});
	std::vector<Index> indices = index.split();
	ASSERT_EQ(indices.size(), 3);
	ASSERT_TRUE(indices[0].equals({1, 2, 3}));
	ASSERT_TRUE(indices[1].equals({4, 5}));
	ASSERT_TRUE(indices[2].equals({6, 7, 8}));
}

TEST(Index, isPatternIndex){
	std::string errorMessage = "isPatternIndex() failed.";

	//TBTKFeature Core.Index.isPatternIndex.1 2019-09-19
	EXPECT_FALSE(Index({1, 2, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.2 2019-09-19
	EXPECT_TRUE(Index({1, _a_, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.2 2019-09-19
	EXPECT_TRUE(Index({1, IDX_ALL, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.3 2019-09-19
	EXPECT_TRUE(Index({1, IDX_SUM_ALL, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.4 2019-09-19
	EXPECT_TRUE(Index({1, IDX_X, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.5 2019-09-19
	EXPECT_TRUE(Index({1, IDX_Y, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.6 2019-09-19
	EXPECT_TRUE(Index({1, IDX_Z, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.7 2019-09-19
	EXPECT_TRUE(Index({1, IDX_SPIN, 3}).isPatternIndex()) << errorMessage;
	//TBTKFeature Core.Index.isPatternIndex.8 2019-09-19
	EXPECT_TRUE(Index({1, IDX_SEPARATOR, 3}).isPatternIndex()) << errorMessage;
}

TEST(Index, toString){
	std::string errorMessage = "toString() failed.";

	//TBTKFeature Core.Index.toString.1 2019-09-19
	EXPECT_TRUE(Index({1, 2, 3}).toString().compare("{1, 2, 3}") == 0) << errorMessage;
	//TBTKFeature Core.Index.toString.2 2019-09-19
	EXPECT_TRUE(Index({{1}, {2, 3}, {4, 5, 6}}).toString().compare("{1}, {2, 3}, {4, 5, 6}") == 0) << errorMessage;
}

TEST(Index, operatorLessThan){
	std::string errorMessage = "operator<() failed.";

	//TBTKFeature Core.Index.operator<.1.C++ 2019-09-19
	EXPECT_FALSE(Index({1, 2, 3}) < Index({1, 2, 3}));
	//TBTKFeature Core.Index.operator<.2.C++ 2019-09-19
	EXPECT_TRUE(Index({1, 2, 3}) < Index({1, 2, 4}));
	//TBTKFeature Core.Index.operator<.3.C++ 2019-09-19
	EXPECT_FALSE(Index({1, 2, 3}) < Index({1, 1, 4}));
	//TBTKFeature Core.Index.operator<.4.C++ 2019-09-19
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Index({1, 2}) < Index({1, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
	//TBTKFeature Core.Index.operator<.5.C++ 2019-09-19
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Index({1, 2, 3}) < Index({1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Index, operatorGreaterThan){
	std::string errorMessage = "operator>() failed.";

	//TBTKFeature Core.Index.operator>.1.C++ 2019-09-19
	EXPECT_FALSE(Index({1, 2, 3}) > Index({1, 2, 3}));
	//TBTKFeature Core.Index.operator>.2.C++ 2019-09-19
	EXPECT_FALSE(Index({1, 2, 3}) > Index({1, 2, 4}));
	//TBTKFeature Core.Index.operator>.3.C++ 2019-09-19
	EXPECT_TRUE(Index({1, 2, 3}) > Index({1, 1, 4}));
	//TBTKFeature Core.Index.operator>.4.C++ 2019-09-19
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Index({1, 2}) > Index({1, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
	//TBTKFeature Core.Index.operator>.5.C++ 2019-09-19
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Index({1, 2, 3}) > Index({1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Index, getSizeInBytes){
	EXPECT_TRUE(Index().getSizeInBytes() > 0) << "getSizeInBytes() failed.";
}

};
