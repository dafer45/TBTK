#include "TBTK/Property/IndexDescriptor.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(IndexDescriptor, ConstructorNone){
	//Not possible to test on its own.
}

TEST(IndexDescriptor, ConstructorRanges){
	//Not possible to test on its own.
}

TEST(IndexDescriptor, ConstructorCustom){
	//Fail if the linear map has not been generated for the IndexTree.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexTree indexTree;
			IndexDescriptor indexDescriptor(indexTree);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TODO
//...
TEST(IndexDescriptor, CopyConstructor){
}

//TODO
//...
TEST(IndexDescriptor, MoveConstructor){
}

//TODO
//...
TEST(IndexDescriptor, SerializeToJSON){
}

//TODO
//...
TEST(IndexDescriptor, Destructor){
}

//TODO
//...
TEST(IndexDescriptor, operatorAssignment){
}

//TODO
//...
TEST(IndexDescriptor, operatorMoveAssignment){
}

TEST(IndexDescriptor, operatorComparison){
	//Format::None.
	IndexDescriptor indexDescriptorNone0;
	IndexDescriptor indexDescriptorNone1;

	EXPECT_TRUE(indexDescriptorNone0 == indexDescriptorNone1);
	EXPECT_FALSE(indexDescriptorNone0 != indexDescriptorNone1);

	//Format::Ranges.
	IndexDescriptor indexDescriptorRanges0({1, 2, 3});
	IndexDescriptor indexDescriptorRanges1({1, 2, 3});
	IndexDescriptor indexDescriptorRanges2({1, 2});
	IndexDescriptor indexDescriptorRanges3({1, 3, 2});

	EXPECT_TRUE(indexDescriptorRanges0 == indexDescriptorRanges1);
	EXPECT_FALSE(indexDescriptorRanges0 != indexDescriptorRanges1);
	EXPECT_FALSE(indexDescriptorRanges0 == indexDescriptorRanges2);
	EXPECT_TRUE(indexDescriptorRanges0 != indexDescriptorRanges2);
	EXPECT_FALSE(indexDescriptorRanges0 == indexDescriptorRanges3);
	EXPECT_TRUE(indexDescriptorRanges0 != indexDescriptorRanges3);

	//Format custom.
	IndexTree indexTree0;
	indexTree0.add({1, 2, 3});
	indexTree0.add({2, 2, 3});
	indexTree0.add({2, 3, 1});
	indexTree0.generateLinearMap();

	IndexTree indexTree1;
	indexTree1.add({1, 2, 3});
	indexTree1.add({2, 2, 3});
	indexTree1.add({2, 3, 1});
	indexTree1.generateLinearMap();

	IndexTree indexTree2;
	indexTree2.add({1, 2, 3});
	indexTree2.add({2, 3, 1});
	indexTree2.generateLinearMap();

	IndexDescriptor indexDescriptorCustom0(indexTree0);
	IndexDescriptor indexDescriptorCustom1(indexTree1);
	IndexDescriptor indexDescriptorCustom2(indexTree2);

	EXPECT_TRUE(indexDescriptorCustom0 == indexDescriptorCustom1);
	EXPECT_FALSE(indexDescriptorCustom0 != indexDescriptorCustom1);
	EXPECT_FALSE(indexDescriptorCustom0 == indexDescriptorCustom2);
	EXPECT_TRUE(indexDescriptorCustom0 != indexDescriptorCustom2);

	//Different formats.
	EXPECT_FALSE(indexDescriptorNone0 == indexDescriptorRanges0);
	EXPECT_TRUE(indexDescriptorNone0 != indexDescriptorRanges0);
	EXPECT_FALSE(indexDescriptorNone0 == indexDescriptorCustom0);
	EXPECT_TRUE(indexDescriptorNone0 != indexDescriptorCustom0);
	EXPECT_FALSE(indexDescriptorRanges0 == indexDescriptorCustom0);
	EXPECT_TRUE(indexDescriptorRanges0 != indexDescriptorCustom0);
}

TEST(IndexDescriptor, operatorInequality){
	//Tested through IndexDescriptor::operatorComparison
}

TEST(IndexDescriptor, getFormat){
	IndexDescriptor indexDescriptor0;
	EXPECT_EQ(indexDescriptor0.getFormat(), IndexDescriptor::Format::None);
	IndexDescriptor indexDescriptor1({2, 3});
	EXPECT_EQ(
		indexDescriptor1.getFormat(),
		IndexDescriptor::Format::Ranges
	);
	IndexTree indexTree;
	indexTree.add({1, 2});
	indexTree.generateLinearMap();
	IndexDescriptor indexDescriptor2(indexTree);
	EXPECT_EQ(
		indexDescriptor2.getFormat(),
		IndexDescriptor::Format::Custom
	);

	//TODO:
	//Replace commented out code with valid code for the creation of the
	//IndexDescriptor on the Dynamic format.
/*	IndexDescriptor indexDescriptor3(IndexDescriptor::Format::Dynamic);
	EXPECT_EQ(
		indexDescriptor3.getFormat(),
		IndexDescriptor::Format::Dynamic
	);*/
}

TEST(IndexDescriptor, getRanges){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor;
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Ranges.
	IndexDescriptor indexDescriptor({2, 3, 4});
	std::vector<int> ranges = indexDescriptor.getRanges();
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);

	//Fail for Format::Custom.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexTree indexTree;
			indexTree.add({1, 2});
			indexTree.generateLinearMap();
			IndexDescriptor indexDescriptor(indexTree);
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Dynamic.
	//TODO:
	//Replace commented out code with valid code for the creation of the
	//IndexDescriptor on the Dynamic format.
/*	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);*/
}

TEST(IndexDescriptor, getIndexTree){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor;
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor({2, 3, 4});
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	IndexDescriptor indexDescriptor(indexTreeInput);
	const IndexTree &indexTree = indexDescriptor.getIndexTree();
	EXPECT_EQ(indexTree.getSize(), 3);
	EXPECT_EQ(indexTree.getLinearIndex({0}), 0);
	EXPECT_EQ(indexTree.getLinearIndex({1}), 1);
	EXPECT_EQ(indexTree.getLinearIndex({2}), 2);

	//Fail for Format::Dynamic.
	//TODO:
	//Replace commented out code with valid code for the creation of the
	//IndexDescriptor on the Dynamic format.
/*	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);*/
}

TEST(IndexDescriptor, add){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor;
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor({2, 3, 4});
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexTree indexTree;
			indexTree.add({1, 2});
			indexTree.generateLinearMap();
			IndexDescriptor indexDescriptor(indexTree);
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Dynamic.
	//TODO:
	//Replace commented out code with valid code for the creation of the
	//IndexDescriptor on the Dynamic format.
/*	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Dynamic);
	indexDescriptor.add({0});
	indexDescriptor.add({2});
	indexDescriptor.add({1});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);*/
}

TEST(IndexDescriptor, getLinearIndex){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor;
			indexDescriptor.getLinearIndex({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor({2, 3, 4});
			indexDescriptor.getLinearIndex({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	IndexDescriptor indexDescriptor0(indexTreeInput);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({0}), 0);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({1}), 1);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({2}), 2);
	EXPECT_THROW(indexDescriptor0.getLinearIndex({3}), IndexException);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({3}, true), -1);

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
/*	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Dynamic);
	indexDescriptor1.add({0});
	indexDescriptor1.add({2});
	indexDescriptor1.add({1});
	EXPECT_EQ(indexDescriptor1.getLinearIndex({0}), 0);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({1}), 2);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({2}), 1);
	EXPECT_THROW(indexDescriptor1.getLinearIndex({3}), IndexException);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({3}, true), -1);*/
}

TEST(IndexDescriptor, getSize){
	//Format::None.
	IndexDescriptor indexDescriptor0;
	EXPECT_EQ(indexDescriptor0.getSize(), 1);

	//Format::Ranges.
	IndexDescriptor indexDescriptor1({2, 3, 4});
	EXPECT_EQ(indexDescriptor1.getSize(), 2*3*4);

	//Format::Custom.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	IndexDescriptor indexDescriptor2(indexTree);
	EXPECT_EQ(indexDescriptor2.getSize(), 3);

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
/*	IndexDescriptor indexDescriptor3(IndexDescriptor::Format::Dynamic);
	indexDescriptor3.add({0});
	indexDescriptor3.add({2});
	indexDescriptor3.add({1});
	EXPECT_EQ(indexDescriptor3.getSize(), 3);*/
}

TEST(IndexDescriptor, contains){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor;
			indexDescriptor.contains({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor({2, 3, 4});
			indexDescriptor.contains({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom.
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	IndexDescriptor indexDescriptor0(indexTreeInput);
	EXPECT_TRUE(indexDescriptor0.contains({0}));
	EXPECT_TRUE(indexDescriptor0.contains({1}));
	EXPECT_TRUE(indexDescriptor0.contains({2}));
	EXPECT_FALSE(indexDescriptor0.contains({3}));

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
/*	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Dynamic);
	indexDescriptor1.add({0});
	indexDescriptor1.add({2});
	indexDescriptor1.add({1});
	EXPECT_TRUE(indexDescriptor1.contains({0}));
	EXPECT_TRUE(indexDescriptor1.contains({1}));
	EXPECT_TRUE(indexDescriptor1.contains({2}));
	EXPECT_FALSE(indexDescriptor1.contains({3}));*/
}

TEST(IndexDescriptor, serialize){
	//Already tested through SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
