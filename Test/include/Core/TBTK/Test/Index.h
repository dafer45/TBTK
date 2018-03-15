#include "TBTK/Index.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Index, ConstructorEmpty){
	Index index;
	EXPECT_EQ(index.getSize(), 0) << "Empty Index does not have size 0.";
}

TEST(Index, ConstructorInitializerList){
	Index index({1, 2, 3});
	EXPECT_EQ(index.getSize(), 3) << "Index({1, 2, 3}) does not have size 3.";
	EXPECT_EQ(index[0], 1) << "Index({1, 2, 3}) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index({1, 2, 3}) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index({1, 2, 3}) does not have '3' as third subindex.";
}

TEST(Index, ConstructorVector){
	std::vector<int> myVector({1,2,3});
	Index index(myVector);
	EXPECT_EQ(index.getSize(), 3) << "Index(std::vector<int>({1, 2, 3})) does not have size 3.";
	EXPECT_EQ(index[0], 1)<< "Index(std::vector<int>({1, 2, 3})) does not have '1' as first subindex.";
	EXPECT_EQ(index[1], 2) << "Index(std::vector<int>({1, 2, 3})) does not have '2' as second subindex.";
	EXPECT_EQ(index[2], 3) << "Index(std::vector<int>({1, 2, 3})) does not have '3' as third subindex.";
}

TEST(Index, CopyConstructor){
	Index index({1, 2, 3});
	Index indexCopy = index;
	EXPECT_EQ(indexCopy.getSize(), 3) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[0], 1) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[1], 2) << "Copy constructor failed.";
	EXPECT_EQ(indexCopy[2], 3) << "Copy constructor failed.";
}

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

TEST(Index, ConstructorConcatenationVector){
	std::string errorMessage = "Index concatenation filed.";

	std::vector<int> head({1, 2});
	std::vector<int> tail({3});
	Index index(head, tail);
	EXPECT_EQ(index.getSize(), 3) << errorMessage;
	EXPECT_EQ(index[0], 1) << errorMessage;
	EXPECT_EQ(index[1], 2) << errorMessage;
	EXPECT_EQ(index[2], 3) << errorMessage;
}

TEST(Index, ConstructorCompundInitializerList){
	std::string errorMessage = "Compund Index construction failed.";

	Index index({{1}, {2, 3}, {4, 5, 6}});
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

TEST(Index, ConstructorCompundVector){
	std::string errorMessage = "Compund Index construction failed.";

	std::vector<std::vector<int>> indices;
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

TEST(Index, ConstructorString){
	std::string errorMessage = "Index construction from string failed.";

	Index index0("{1, 2, 3}");
	EXPECT_TRUE(index0.equals({1, 2, 3})) << errorMessage;
}

TEST(Index, SerializeToJSON){
	Index index0({1, 2, 3});
	Index index1(
		index0.serialize(Serializeable::Mode::JSON),
		Serializeable::Mode::JSON
	);
	EXPECT_TRUE(index1.equals(index0)) << "JSON serialization failed.";
}

TEST(Index, equals){
	std::string errorMessage = "Index comparison failed.";

	EXPECT_TRUE(Index({1, 2, 3}).equals(Index({1, 2, 3}))) << errorMessage;
	EXPECT_FALSE(Index({1, 2, 2}).equals(Index({1, 2, 3}))) << errorMessage;
	EXPECT_FALSE(Index({1, 2, IDX_ALL}).equals(Index({1, 2, 3}))) << errorMessage;
	EXPECT_TRUE(Index({1, 2, IDX_ALL}).equals(Index({1, 2, 3}), true)) << errorMessage;
	EXPECT_TRUE(Index({1, -1, 3}).equals(Index({1, -1, 3}))) << errorMessage;
	EXPECT_TRUE(Index({{1, 2, 3}, {4, 5, 6}}).equals(Index({{1, 2, 3}, {4, 5, 6}}))) << errorMessage;
}

TEST(Index, atAndSubscriptOperator){
	std::string errorMessage = "at() and operator[] gives different results.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.at(0), index[0]) << errorMessage;
	EXPECT_EQ(index.at(1), index[1]) << errorMessage;
	EXPECT_EQ(index.at(2), index[2]) << errorMessage;
}

TEST(Index, atAndSubscriptOperatorConst){
	std::string errorMessage = "at() and operator[] gives different results.";

	const Index index({1, 2, 3});
	EXPECT_EQ(index.at(0), index[0]) << errorMessage;
	EXPECT_EQ(index.at(1), index[1]) << errorMessage;
	EXPECT_EQ(index.at(2), index[2]) << errorMessage;
}

TEST(Index, getSize){
	std::string errorMessage = "getSize() failed.";

	EXPECT_EQ(Index().getSize(), 0) << errorMessage;
	EXPECT_EQ(Index({1}).getSize(), 1) << errorMessage;
	EXPECT_EQ(Index({1, 2, 3}).getSize(), 3) << errorMessage;
	EXPECT_EQ(Index({1, 2}, {3, 4}).getSize(), 4) << errorMessage;
	EXPECT_EQ(Index({{1, 2}, {3, 4}}).getSize(), 5) << errorMessage;
}

TEST(Index, push_back){
	Index index;
	index.push_back(1);
	index.push_back(2);
	index.push_back(3);
	EXPECT_TRUE(index.equals({1, 2, 3})) << "push_back failed.";
}

TEST(Index, popFront){
	std::string errorMessage = "popFront() failed.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.popFront(), 1) << errorMessage;
	EXPECT_TRUE(index.equals({2, 3})) << errorMessage;
}

TEST(Index, popBack){
	std::string errorMessage = "popBack() failed.";

	Index index({1, 2, 3});
	EXPECT_EQ(index.popBack(), 3) << errorMessage;
	EXPECT_TRUE(index.equals({1, 2})) << errorMessage;
}

TEST(Index, getUnitRange){
	Index index({1, 2, 3});
	EXPECT_TRUE(index.getUnitRange().equals({1, 1, 1})) << "getUnitRange() failed.";
}

TEST(Index, getSubIndex){
	Index index({1, 2, 3, 4, 5, 6, 7, 8, 9});
	EXPECT_TRUE(index.getSubIndex(3, 5).equals({4, 5, 6})) << "getSubIndex() failed.";
}

TEST(Index, isPatternIndex){
	std::string errorMessage = "isPatternIndex() failed.";

	EXPECT_FALSE(Index({1, 2, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, _, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, ___, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_ALL, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_SUM_ALL, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_X, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_Y, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_Z, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_SPIN, 3}).isPatternIndex()) << errorMessage;
	EXPECT_TRUE(Index({1, IDX_SEPARATOR, 3}).isPatternIndex()) << errorMessage;
}

TEST(Index, toString){
	std::string errorMessage = "toString() failed.";

	EXPECT_TRUE(Index({1, 2, 3}).toString().compare("{1, 2, 3}") == 0) << errorMessage;
	EXPECT_TRUE(Index({{1}, {2, 3}, {4, 5, 6}}).toString().compare("{1}, {2, 3}, {4, 5, 6}") == 0) << errorMessage;
}

TEST(Index, operatorLessThan){
	std::string errorMessage = "operator<() failed.";

	Streams::setStdMuteErr();

	EXPECT_FALSE(Index({1, 2, 3}) < Index({1, 2, 3}));
	EXPECT_TRUE(Index({1, 2, 3}) < Index({1, 2, 4}));
	EXPECT_FALSE(Index({1, 2, 3}) < Index({1, 1, 4}));
	EXPECT_EXIT(Index({1, 2}) < Index({1, 2, 3}), ::testing::ExitedWithCode(1), "");
	EXPECT_EXIT(Index({1, 2, 3}) < Index({1, 2}), ::testing::ExitedWithCode(1), "");
}

TEST(Index, operatorGreaterThan){
	std::string errorMessage = "operator>() failed.";

	Streams::setStdMuteErr();

	EXPECT_FALSE(Index({1, 2, 3}) > Index({1, 2, 3}));
	EXPECT_FALSE(Index({1, 2, 3}) > Index({1, 2, 4}));
	EXPECT_TRUE(Index({1, 2, 3}) > Index({1, 1, 4}));
	EXPECT_EXIT(Index({1, 2}) > Index({1, 2, 3}), ::testing::ExitedWithCode(1), "");
	EXPECT_EXIT(Index({1, 2, 3}) > Index({1, 2}), ::testing::ExitedWithCode(1), "");
}

TEST(Index, getSizeInBytes){
	EXPECT_TRUE(Index().getSizeInBytes() > 0) << "getSizeInBytes() failed.";
}

};
