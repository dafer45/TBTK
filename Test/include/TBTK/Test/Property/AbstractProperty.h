#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

//Makes protected members public for testing.
template<typename DataType>
class Property : public AbstractProperty<DataType>{
public:
	Property() : AbstractProperty<DataType>(){};

	Property(
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(blockSize){}

	Property(
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(blockSize, data){}

	Property(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(dimensions, ranges, blockSize){}

	Property(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(
			dimensions,
			ranges,
			blockSize,
			data
		){}

	Property(
		const IndexTree &indexTree,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(indexTree, blockSize){}

	Property(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(indexTree, blockSize, data){}

	Property(
		const Property &property
	) :
		AbstractProperty<DataType>(property){}

	Property(
		Property &&property
	) :
		AbstractProperty<DataType>(property){}

	Property(
		const std::string &serialization,
		Serializable::Mode mode
	) : AbstractProperty<DataType>(serialization, mode){}

	Property& operator=(const Property &property){
		AbstractProperty<DataType>::operator=(property);

		return *this;
	}
	Property& operator=(Property &&property){
		AbstractProperty<DataType>::operator=(property);

		return *this;
	}
};

TEST(AbstractProperty, Constructor0){
	//Construct an uninitialized Property.
	Property<int> property;
	EXPECT_EQ(property.getBlockSize(), 0);
	EXPECT_EQ(property.getSize(), 0);
	EXPECT_EQ(property.getData().size(), 0);
}

TEST(AbstractProperty, Constructor1){
	//Construct a Property with a single data block.
	Property<int> property(10);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_EQ(data[n], 0);
}

TEST(AbstractProperty, Constructor2){
	//Construct a Property with a single data block and initialize it with
	//provided values.
	const int dataInput[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property(10, dataInput);
	ASSERT_EQ(property.getBlockSize(), 10);
	ASSERT_EQ(property.getSize(), 10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	EXPECT_EQ(data[0], 0);
	EXPECT_EQ(data[1], 1);
	EXPECT_EQ(data[2], 2);
	EXPECT_EQ(data[9], 9);
}

TEST(AbstractProperty, Constructor3){
	//Construct a Property with the Ranges format.
	const int ranges[3] = {2, 3, 4};
	Property<int> property(3, ranges, 10);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 2*3*4*10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(AbstractProperty, Constructor4){
	//Construct a Property with the Ranges format and initialize it with
	//provided values.
	const int ranges[3] = {2, 3, 4};
	int dataInput[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput[n] = n;
	Property<int> property(3, ranges, 10, dataInput);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 2*3*4*10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(AbstractProperty, Constructor5){
	//Construct a Property with the Custom format.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});

	//Fail construction if the linnear map has not been generated for the
	//IndexTree.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Property<int> property(indexTree, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed if it has.
	indexTree.generateLinearMap();
	Property<int> property(indexTree, 10);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 3*10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_EQ(data[n], 0);
}

TEST(AbstractProperty, Constructor6){
	//Construct a Property with the Custom format and initialize it with
	//the provided values.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});
	int inputData[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputData[n] = n;

	//Fail construction if the linnear map has not been generated for the
	//IndexTree.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Property<int> property(indexTree, 10, inputData);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed if it has.
	indexTree.generateLinearMap();
	Property<int> property(indexTree, 10, inputData);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 3*10);
	const std::vector<int> &data = property.getData();
	ASSERT_EQ(data.size(), property.getSize());
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_EQ(data[n], n);
}

TEST(AbstractProperty, CopyConstructor){
	//Property with a single block.
	const int dataInput0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property0(10, dataInput0);
	Property<int> property1(property0);
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	const int ranges[3] = {2, 3, 4};
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	Property<int> property2(3, ranges, 10, dataInput2);
	Property<int> property3(property2);
	EXPECT_EQ(property3.getBlockSize(), 10);
	EXPECT_EQ(property3.getSize(), 2*3*4*10);
	const std::vector<int> &data3 = property3.getData();
	ASSERT_EQ(data3.size(), property3.getSize());
	for(unsigned int n = 0; n < data3.size(); n++)
		EXPECT_DOUBLE_EQ(data3[n], n);

	//Property with Custom format.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});
	int inputData4[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputData4[n] = n;
	indexTree.generateLinearMap();
	Property<int> property4(indexTree, 10, inputData4);
	Property<int> property5(property4);
	EXPECT_EQ(property5.getBlockSize(), 10);
	EXPECT_EQ(property5.getSize(), 3*10);
	const std::vector<int> &data5 = property5.getData();
	ASSERT_EQ(data5.size(), property5.getSize());
	for(unsigned int n = 0; n < data5.size(); n++)
		EXPECT_EQ(data5[n], n);
}

TEST(AbstractProperty, MoveConstructor){
	//Property with a single block.
	const int dataInput0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property0(10, dataInput0);
	Property<int> property1(std::move(property0));
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	const int ranges[3] = {2, 3, 4};
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	Property<int> property2(3, ranges, 10, dataInput2);
	Property<int> property3(std::move(property2));
	EXPECT_EQ(property3.getBlockSize(), 10);
	EXPECT_EQ(property3.getSize(), 2*3*4*10);
	const std::vector<int> &data3 = property3.getData();
	ASSERT_EQ(data3.size(), property3.getSize());
	for(unsigned int n = 0; n < data3.size(); n++)
		EXPECT_DOUBLE_EQ(data3[n], n);

	//Property with Custom format.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});
	int inputData4[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputData4[n] = n;
	indexTree.generateLinearMap();
	Property<int> property4(indexTree, 10, inputData4);
	Property<int> property5(std::move(property4));
	EXPECT_EQ(property5.getBlockSize(), 10);
	EXPECT_EQ(property5.getSize(), 3*10);
	const std::vector<int> &data5 = property5.getData();
	ASSERT_EQ(data5.size(), property5.getSize());
	for(unsigned int n = 0; n < data5.size(); n++)
		EXPECT_EQ(data5[n], n);
}

//TODO
//This function is currently only implemented for DataType double and
//std::complex<double>. First implement full support for all primitive data
//types, Serializable, and pseudo-Serializable classes.
TEST(AbstractProperty, SerializeToJSON){
}

TEST(AbstractProperty, Destructor){
	//Not testable on its own.
}

TEST(AbstractProperty, OperatorAssignment){
	//Property with a single block.
	const int dataInput0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property0(10, dataInput0);
	Property<int> property1;
	property1 = property0;
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	const int ranges[3] = {2, 3, 4};
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	Property<int> property2(3, ranges, 10, dataInput2);
	Property<int> property3;
	property3 = property2;
	EXPECT_EQ(property3.getBlockSize(), 10);
	EXPECT_EQ(property3.getSize(), 2*3*4*10);
	const std::vector<int> &data3 = property3.getData();
	ASSERT_EQ(data3.size(), property3.getSize());
	for(unsigned int n = 0; n < data3.size(); n++)
		EXPECT_DOUBLE_EQ(data3[n], n);

	//Property with Custom format.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});
	int inputData4[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputData4[n] = n;
	indexTree.generateLinearMap();
	Property<int> property4(indexTree, 10, inputData4);
	Property<int> property5;
	property5 = property4;
	EXPECT_EQ(property5.getBlockSize(), 10);
	EXPECT_EQ(property5.getSize(), 3*10);
	const std::vector<int> &data5 = property5.getData();
	ASSERT_EQ(data5.size(), property5.getSize());
	for(unsigned int n = 0; n < data5.size(); n++)
		EXPECT_EQ(data5[n], n);
}

TEST(AbstractProperty, OperatorMoveAssignment){
	//Property with a single block.
	const int dataInput0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property0(10, dataInput0);
	Property<int> property1;
	property1 = std::move(property0);
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	const int ranges[3] = {2, 3, 4};
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	Property<int> property2(3, ranges, 10, dataInput2);
	Property<int> property3;
	property3 = std::move(property2);
	EXPECT_EQ(property3.getBlockSize(), 10);
	EXPECT_EQ(property3.getSize(), 2*3*4*10);
	const std::vector<int> &data3 = property3.getData();
	ASSERT_EQ(data3.size(), property3.getSize());
	for(unsigned int n = 0; n < data3.size(); n++)
		EXPECT_DOUBLE_EQ(data3[n], n);

	//Property with Custom format.
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({1, 3});
	int inputData4[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputData4[n] = n;
	indexTree.generateLinearMap();
	Property<int> property4(indexTree, 10, inputData4);
	Property<int> property5;
	property5 = std::move(property4);
	EXPECT_EQ(property5.getBlockSize(), 10);
	EXPECT_EQ(property5.getSize(), 3*10);
	const std::vector<int> &data5 = property5.getData();
	ASSERT_EQ(data5.size(), property5.getSize());
	for(unsigned int n = 0; n < data5.size(); n++)
		EXPECT_EQ(data5[n], n);
}

TEST(AbstractProperty, getBlockSize){
	//Already tested through the constructors.
}

TEST(AbstractProperty, getSize){
	//Already tested through the constructors.
}

TEST(AbstractProperty, getData){
	//Already tested through the constructors.
}

TEST(AbstractProperty, getDataRW){
	Property<int> property(10);
	const std::vector<int> &data0 = property.getData();
	std::vector<int> &data1 = property.getDataRW();
	ASSERT_EQ(data0.size(), 10);
	ASSERT_EQ(data1.size(), 10);
	for(unsigned int n = 0; n < data1.size(); n++)
		data1[n] = n;
	for(unsigned int n = 0; n < data0.size(); n++)
		EXPECT_EQ(data0[n], n);
}

TEST(AbstractProperty, getDimensions){
	//Fail for IndexDescriptor::Format::None.
	Property<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getDimensions();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	const int ranges[3] = {2, 3, 4};
	Property<int> property1(3, ranges, 10);
	EXPECT_EQ(property1.getDimensions(), 3);

	//Fail for IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.generateLinearMap();
	Property<int> property2(indexTree, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property2.getDimensions();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(AbstractProperty, getRanges){
	//Fail for IndexDescriptor::Format::None.
	Property<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	const int rangesIn[3] = {2, 3, 4};
	Property<int> property1(3, rangesIn, 10);
	const std::vector<int> ranges = property1.getRanges();
	ASSERT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);

	//Fail for IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.generateLinearMap();
	Property<int> property2(indexTree, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property2.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(AbstractProperty, getOffset){
	//Fail for IndexDescriptor::Format::None.
	Property<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getOffset({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	const int rangesIn[3] = {2, 3, 4};
	Property<int> property1(3, rangesIn, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property1.getOffset({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	Property<int> property2(indexTree, 10);
	EXPECT_EQ(property2.getOffset({0}), 0);
	EXPECT_EQ(property2.getOffset({1}), 10);
	EXPECT_EQ(property2.getOffset({2}), 20);
}

//TODO
//This function should probably be removed.
TEST(AbstractProperty, getIndexDescriptor){
}

TEST(AbstractProperty, contains){
	//Fail for IndexDescriptor::Format::None.
	Property<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.contains({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	const int rangesIn[3] = {2, 3, 4};
	Property<int> property1(3, rangesIn, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property1.contains({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	Property<int> property2(indexTree, 10);
	EXPECT_TRUE(property2.contains({0}));
	EXPECT_TRUE(property2.contains({1}));
	EXPECT_TRUE(property2.contains({2}));
	EXPECT_FALSE(property2.contains({3}));
}

TEST(AbstractProperty, operatorFunction){
	Property<int> property0(10);
	//Fail to use indexed version for IndexDescriptor::Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			const_cast<const Property<int>&>(property0)({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Non-indexed version.
	property0(0) = 0;
	property0(1) = 1;
	property0(2) = 2;
	EXPECT_EQ(const_cast<const Property<int>&>(property0)(0), 0);
	EXPECT_EQ(const_cast<const Property<int>&>(property0)(1), 1);
	EXPECT_EQ(const_cast<const Property<int>&>(property0)(2), 2);

	//Fail to use indexed version for IndexDescriptor::Format::Ranges.
	const int rangesIn[3] = {2, 3, 4};
	Property<int> property1(3, rangesIn, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property1({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			const_cast<const Property<int>&>(property1)({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Non-indexed version
	property1(0) = 0;
	property1(1) = 1;
	property1(2) = 2;
	EXPECT_EQ(const_cast<const Property<int>&>(property1)(0), 0);
	EXPECT_EQ(const_cast<const Property<int>&>(property1)(1), 1);
	EXPECT_EQ(const_cast<const Property<int>&>(property1)(2), 2);

	//IndexDescriptor::Format::Custom. Normal behavior.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	Property<int> property2(indexTree, 10);
	property2({0}, 1) = 1;
	property2({1}, 2) = 2;
	property2({2}, 3) = 3;
	EXPECT_EQ(const_cast<Property<int>&>(property2)({0}, 1), 1);
	EXPECT_EQ(const_cast<Property<int>&>(property2)({1}, 2), 2);
	EXPECT_EQ(const_cast<Property<int>&>(property2)({2}, 3), 3);
	EXPECT_EQ(const_cast<Property<int>&>(property2)(1), 1);
	EXPECT_EQ(const_cast<Property<int>&>(property2)(12), 2);
	EXPECT_EQ(const_cast<Property<int>&>(property2)(23), 3);
	property2(12) = 12;
	EXPECT_EQ(const_cast<Property<int>&>(property2)(12), 12);
	//IndexDescriptor::Format::Custom. Check index out-of-bounds behavior.
	property2.setDefaultValue(99);
	EXPECT_THROW(property2({3}, 1), IndexException);
	EXPECT_THROW(
		const_cast<Property<int>&>(property2)({3}, 1),
		IndexException
	);
	property2.setAllowIndexOutOfBoundsAccess(true);
	EXPECT_EQ(property2({3}, 0), 99);
	EXPECT_EQ(const_cast<Property<int>&>(property2)({3}, 0), 99);
	property2({3}, 0) = 0;
	EXPECT_EQ(property2({3}, 0), 99);
}

TEST(AbstractProperty, setAllowOutOfBoundsAccess){
	//Already tested through AbstractProperty;;operatorFunction
}

TEST(AbstractProperty, setDefaultValue){
	//Already tested through AbstractProperty;;operatorFunction
}

//TODO
//This function is currently not implemented for all data types. First
//implement full support for all primitive data types, Serializable, and
//pseudo-Serializable classes.
TEST(AbstractProperty, serialize){
}

};	//End of namespace Property
};	//End of namespace TBTK
