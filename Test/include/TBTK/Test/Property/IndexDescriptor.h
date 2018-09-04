#include "TBTK/Property/IndexDescriptor.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(IndexDescriptor, Constructor){
	//Not possible to test on its own.
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

TEST(IndexDescriptor, getFormat){
	IndexDescriptor indexDescriptor0(IndexDescriptor::Format::None);
	EXPECT_EQ(indexDescriptor0.getFormat(), IndexDescriptor::Format::None);
	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Ranges);
	EXPECT_EQ(
		indexDescriptor1.getFormat(),
		IndexDescriptor::Format::Ranges
	);
	IndexDescriptor indexDescriptor2(IndexDescriptor::Format::Custom);
	EXPECT_EQ(
		indexDescriptor2.getFormat(),
		IndexDescriptor::Format::Custom
	);
	IndexDescriptor indexDescriptor3(IndexDescriptor::Format::Dynamic);
	EXPECT_EQ(
		indexDescriptor3.getFormat(),
		IndexDescriptor::Format::Dynamic
	);
}

TEST(IndexDescriptor, setRanges){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.setRanges({2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Ranges.
	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Ranges);
	indexDescriptor.setRanges({2, 3, 4});

	//Fail for Format::Custom.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Custom
			);
			indexDescriptor.setRanges({2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Dynamic.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.setRanges({2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(IndexDescriptor, getRanges){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Ranges.
	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Ranges);
	indexDescriptor.setRanges({2, 3, 4});
	std::vector<int> ranges = indexDescriptor.getRanges();
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);

	//Fail for Format::Custom.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Custom
			);
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Dynamic.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(IndexDescriptor, setIndexTree){
	IndexTree indexTree0;
	indexTree0.add({0});
	indexTree0.add({1});
	indexTree0.add({2});
	indexTree0.generateLinearMap();

	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.setIndexTree(indexTree0);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Ranges
			);
			indexDescriptor.setIndexTree(indexTree0);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom
	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Custom);
	indexDescriptor.setIndexTree(indexTree0);
	//Fail if the linear map has not been constructed.
	IndexTree indexTree1;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			indexDescriptor.setIndexTree(indexTree1);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Dynamic.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.setIndexTree(indexTree0);
		},
		::testing::ExitedWithCode(1),
		""
	);

}

TEST(IndexDescriptor, getIndexTree){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Ranges
			);
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom
	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Custom);
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	indexDescriptor.setIndexTree(indexTreeInput);
	const IndexTree &indexTree = indexDescriptor.getIndexTree();
	EXPECT_EQ(indexTree.getSize(), 3);
	EXPECT_EQ(indexTree.getLinearIndex({0}), 0);
	EXPECT_EQ(indexTree.getLinearIndex({1}), 1);
	EXPECT_EQ(indexTree.getLinearIndex({2}), 2);

	//Fail for Format::Dynamic.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Dynamic
			);
			indexDescriptor.getIndexTree();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(IndexDescriptor, add){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Ranges
			);
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Custom
			);
			indexDescriptor.add({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Dynamic.
	IndexDescriptor indexDescriptor(IndexDescriptor::Format::Dynamic);
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
	);
}

TEST(IndexDescriptor, getLinearIndex){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.getLinearIndex({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Ranges
			);
			indexDescriptor.getLinearIndex({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom
	IndexDescriptor indexDescriptor0(IndexDescriptor::Format::Custom);
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	indexDescriptor0.setIndexTree(indexTreeInput);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({0}), 0);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({1}), 1);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({2}), 2);
	EXPECT_THROW(indexDescriptor0.getLinearIndex({3}), IndexException);
	EXPECT_EQ(indexDescriptor0.getLinearIndex({3}, true), -1);

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Dynamic);
	indexDescriptor1.add({0});
	indexDescriptor1.add({2});
	indexDescriptor1.add({1});
	EXPECT_EQ(indexDescriptor1.getLinearIndex({0}), 0);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({1}), 2);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({2}), 1);
	EXPECT_THROW(indexDescriptor1.getLinearIndex({3}), IndexException);
	EXPECT_EQ(indexDescriptor1.getLinearIndex({3}, true), -1);
}

TEST(IndexDescriptor, getSize){
	//Format::None.
	IndexDescriptor indexDescriptor0(IndexDescriptor::Format::None);
	EXPECT_EQ(indexDescriptor0.getSize(), 1);

	//Format::Ranges.
	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Ranges);
	indexDescriptor1.setRanges({2, 3, 4});
	EXPECT_EQ(indexDescriptor1.getSize(), 2*3*4);

	//Format::Custom.
	IndexDescriptor indexDescriptor2(IndexDescriptor::Format::Custom);
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	indexDescriptor2.setIndexTree(indexTree);
	EXPECT_EQ(indexDescriptor2.getSize(), 3);

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
	IndexDescriptor indexDescriptor3(IndexDescriptor::Format::Dynamic);
	indexDescriptor3.add({0});
	indexDescriptor3.add({2});
	indexDescriptor3.add({1});
	EXPECT_EQ(indexDescriptor3.getSize(), 3);
}

TEST(IndexDescriptor, contains){
	//Fail for Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::None
			);
			indexDescriptor.contains({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for Format::Ranges.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			IndexDescriptor indexDescriptor(
				IndexDescriptor::Format::Ranges
			);
			indexDescriptor.contains({0});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Format::Custom.
	IndexDescriptor indexDescriptor0(IndexDescriptor::Format::Custom);
	IndexTree indexTreeInput;
	indexTreeInput.add({0});
	indexTreeInput.add({1});
	indexTreeInput.add({2});
	indexTreeInput.generateLinearMap();
	indexDescriptor0.setIndexTree(indexTreeInput);
	EXPECT_TRUE(indexDescriptor0.contains({0}));
	EXPECT_TRUE(indexDescriptor0.contains({1}));
	EXPECT_TRUE(indexDescriptor0.contains({2}));
	EXPECT_FALSE(indexDescriptor0.contains({3}));

	//TODO
	//Implement test for Format::Dynamic once a function for adding Indices
	//to the IndexDescriptor is available.
	IndexDescriptor indexDescriptor1(IndexDescriptor::Format::Dynamic);
	indexDescriptor1.add({0});
	indexDescriptor1.add({2});
	indexDescriptor1.add({1});
	EXPECT_TRUE(indexDescriptor1.contains({0}));
	EXPECT_TRUE(indexDescriptor1.contains({1}));
	EXPECT_TRUE(indexDescriptor1.contains({2}));
	EXPECT_FALSE(indexDescriptor1.contains({3}));
}

TEST(IndexDescriptor, serialize){
	//Already tested through SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
