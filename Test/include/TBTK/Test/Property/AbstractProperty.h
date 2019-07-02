#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

//Makes protected members public for testing.
template<typename DataType>
class PublicAbstractProperty : public AbstractProperty<DataType>{
public:
	PublicAbstractProperty() : AbstractProperty<DataType>(){};

	PublicAbstractProperty(
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(blockSize){}

	PublicAbstractProperty(
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(blockSize, data){}

	PublicAbstractProperty(
		const std::vector<int> &ranges,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(ranges, blockSize){}

	PublicAbstractProperty(
		const std::vector<int> &ranges,
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(
			ranges,
			blockSize,
			data
		){}

	PublicAbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(indexTree, blockSize){}

	PublicAbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(indexTree, blockSize, data){}

	PublicAbstractProperty(
		const PublicAbstractProperty &publicAbstractProperty
	) :
		AbstractProperty<DataType>(publicAbstractProperty){}

	PublicAbstractProperty(
		PublicAbstractProperty &&publicAbstractProperty
	) :
		AbstractProperty<DataType>(publicAbstractProperty){}

	PublicAbstractProperty(
		const std::string &serialization,
		Serializable::Mode mode
	) : AbstractProperty<DataType>(serialization, mode){}

	PublicAbstractProperty& operator=(
		const PublicAbstractProperty &publicAbstractProperty
	){
		AbstractProperty<DataType>::operator=(publicAbstractProperty);

		return *this;
	}
	PublicAbstractProperty& operator=(
		PublicAbstractProperty &&publicAbstractProperty
	){
		AbstractProperty<DataType>::operator=(publicAbstractProperty);

		return *this;
	}
	PublicAbstractProperty& operator+=(
		const PublicAbstractProperty &publicAbstractProperty
	){
		AbstractProperty<DataType>::operator+=(publicAbstractProperty);

		return *this;
	}
	PublicAbstractProperty& operator-=(
		const PublicAbstractProperty &publicAbstractProperty
	){
		AbstractProperty<DataType>::operator-=(publicAbstractProperty);

		return *this;
	}
	PublicAbstractProperty& operator*=(
		const DataType &rhs
	){
		AbstractProperty<DataType>::operator*=(rhs);

		return *this;
	}
	PublicAbstractProperty& operator/=(
		const DataType &rhs
	){
		AbstractProperty<DataType>::operator/=(rhs);

		return *this;
	}
};

TEST(AbstractProperty, Constructor0){
	//Construct an uninitialized Property.
	PublicAbstractProperty<int> property;
	EXPECT_EQ(property.getBlockSize(), 0);
	EXPECT_EQ(property.getSize(), 0);
	EXPECT_EQ(property.getData().size(), 0);
}

TEST(AbstractProperty, Constructor1){
	//Construct a Property with a single data block.
	PublicAbstractProperty<int> property(10);
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
	PublicAbstractProperty<int> property(10, dataInput);
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
	PublicAbstractProperty<int> property({2, 3, 4}, 10);
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
	int dataInput[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput[n] = n;
	PublicAbstractProperty<int> property({2, 3, 4}, 10, dataInput);
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
			PublicAbstractProperty<int> property(indexTree, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed if it has.
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property(indexTree, 10);
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
			PublicAbstractProperty<int> property(indexTree, 10, inputData);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed if it has.
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property(indexTree, 10, inputData);
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
	PublicAbstractProperty<int> property0(10, dataInput0);
	PublicAbstractProperty<int> property1(property0);
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	PublicAbstractProperty<int> property2({2, 3, 4}, 10, dataInput2);
	PublicAbstractProperty<int> property3(property2);
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
	PublicAbstractProperty<int> property4(indexTree, 10, inputData4);
	PublicAbstractProperty<int> property5(property4);
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
	PublicAbstractProperty<int> property0(10, dataInput0);
	PublicAbstractProperty<int> property1(std::move(property0));
	ASSERT_EQ(property1.getBlockSize(), 10);
	ASSERT_EQ(property1.getSize(), 10);
	const std::vector<int> &data1 = property1.getData();
	ASSERT_EQ(data1.size(), property1.getSize());
	EXPECT_EQ(data1[0], 0);
	EXPECT_EQ(data1[1], 1);
	EXPECT_EQ(data1[2], 2);
	EXPECT_EQ(data1[9], 9);

	//Property with Ranges format.
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	PublicAbstractProperty<int> property2({2, 3, 4}, 10, dataInput2);
	PublicAbstractProperty<int> property3(std::move(property2));
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
	PublicAbstractProperty<int> property4(indexTree, 10, inputData4);
	PublicAbstractProperty<int> property5(std::move(property4));
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
	PublicAbstractProperty<int> property0(10, dataInput0);
	PublicAbstractProperty<int> property1;
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
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	PublicAbstractProperty<int> property2({2, 3, 4}, 10, dataInput2);
	PublicAbstractProperty<int> property3;
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
	PublicAbstractProperty<int> property4(indexTree, 10, inputData4);
	PublicAbstractProperty<int> property5;
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
	PublicAbstractProperty<int> property0(10, dataInput0);
	PublicAbstractProperty<int> property1;
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
	int dataInput2[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInput2[n] = n;
	PublicAbstractProperty<int> property2({2, 3, 4}, 10, dataInput2);
	PublicAbstractProperty<int> property3;
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
	PublicAbstractProperty<int> property4(indexTree, 10, inputData4);
	PublicAbstractProperty<int> property5;
	property5 = std::move(property4);
	EXPECT_EQ(property5.getBlockSize(), 10);
	EXPECT_EQ(property5.getSize(), 3*10);
	const std::vector<int> &data5 = property5.getData();
	ASSERT_EQ(data5.size(), property5.getSize());
	for(unsigned int n = 0; n < data5.size(); n++)
		EXPECT_EQ(data5[n], n);
}

TEST(AbstractProperty, operatorAdditionAssignment){
	//IndexDescriptor::Format::None.
	const int dataInputNone0[3] = {0, 1, 2};
	PublicAbstractProperty<int> propertyNone0(3, dataInputNone0);
	const int dataInputNone1[3] = {2, 2, 3};
	PublicAbstractProperty<int> propertyNone1(3, dataInputNone1);
	const int dataInputNone2[2] = {2, 2};
	PublicAbstractProperty<int> propertyNone2(2, dataInputNone2);

	propertyNone0 += propertyNone1;
	const std::vector<int> &dataNone0 = propertyNone0.getData();
	EXPECT_EQ(dataNone0[0], 2);
	EXPECT_EQ(dataNone0[1], 3);
	EXPECT_EQ(dataNone0[2], 5);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 += propertyNone2;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	int dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	PublicAbstractProperty<int> propertyRanges0(
		{2, 3, 4},
		10,
		dataInputRanges0
	);
	int dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	PublicAbstractProperty<int> propertyRanges1(
		{2, 3, 4},
		10,
		dataInputRanges1
	);
	int dataInputRanges2[2*3*10];
	for(unsigned int n = 0; n < 2*3*10; n++)
		dataInputRanges2[n] = n;
	PublicAbstractProperty<int> propertyRanges2(
		{2, 3},
		10,
		dataInputRanges2
	);

	propertyRanges0 += propertyRanges1;
	const std::vector<int> &dataRanges0 = propertyRanges0.getData();
	for(unsigned int n = 0; n < dataRanges0.size(); n++)
		EXPECT_EQ(dataRanges0[n], 3*n);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyRanges0 += propertyRanges2;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTreeCustom0;
	indexTreeCustom0.add({0, 1});
	indexTreeCustom0.add({1, 2});
	indexTreeCustom0.add({1, 3});
	int inputDataCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom0[n] = n;
	indexTreeCustom0.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom0(
		indexTreeCustom0,
		10,
		inputDataCustom0
	);

	IndexTree indexTreeCustom1;
	indexTreeCustom1.add({0, 1});
	indexTreeCustom1.add({1, 2});
	indexTreeCustom1.add({1, 3});
	int inputDataCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom1[n] = 2*n;
	indexTreeCustom1.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom1(
		indexTreeCustom1,
		10,
		inputDataCustom1
	);

	IndexTree indexTreeCustom2;
	indexTreeCustom2.add({0, 1});
	indexTreeCustom2.add({1, 2});
	int inputDataCustom2[2*10];
	for(unsigned int n = 0; n < 2*10; n++)
		inputDataCustom2[n] = n;
	indexTreeCustom2.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom2(
		indexTreeCustom2,
		10,
		inputDataCustom2
	);

	propertyCustom0 += propertyCustom1;
	const std::vector<int> &dataCustom0 = propertyCustom0.getData();
	for(unsigned int n = 0; n < dataCustom0.size(); n++)
		EXPECT_EQ(dataCustom0[n], 3*n);

	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 += propertyRanges0;
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 += propertyCustom0;
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyRanges0 += propertyCustom0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(AbstractProperty, operatorSubtractionAssignment){
	//IndexDescriptor::Format::None.
	const int dataInputNone0[3] = {0, 1, 2};
	PublicAbstractProperty<int> propertyNone0(3, dataInputNone0);
	const int dataInputNone1[3] = {2, 2, 3};
	PublicAbstractProperty<int> propertyNone1(3, dataInputNone1);
	const int dataInputNone2[2] = {2, 2};
	PublicAbstractProperty<int> propertyNone2(2, dataInputNone2);

	propertyNone0 -= propertyNone1;
	const std::vector<int> &dataNone0 = propertyNone0.getData();
	EXPECT_EQ(dataNone0[0], -2);
	EXPECT_EQ(dataNone0[1], -1);
	EXPECT_EQ(dataNone0[2], -1);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 -= propertyNone2;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	int dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	PublicAbstractProperty<int> propertyRanges0(
		{2, 3, 4},
		10,
		dataInputRanges0
	);
	int dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	PublicAbstractProperty<int> propertyRanges1(
		{2, 3, 4},
		10,
		dataInputRanges1
	);
	int dataInputRanges2[2*3*10];
	for(unsigned int n = 0; n < 2*3*10; n++)
		dataInputRanges2[n] = n;
	PublicAbstractProperty<int> propertyRanges2(
		{2, 3},
		10,
		dataInputRanges2
	);

	propertyRanges0 -= propertyRanges1;
	const std::vector<int> &dataRanges0 = propertyRanges0.getData();
	for(unsigned int n = 0; n < dataRanges0.size(); n++)
		EXPECT_EQ(dataRanges0[n], -n);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyRanges0 -= propertyRanges2;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTreeCustom0;
	indexTreeCustom0.add({0, 1});
	indexTreeCustom0.add({1, 2});
	indexTreeCustom0.add({1, 3});
	int inputDataCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom0[n] = n;
	indexTreeCustom0.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom0(
		indexTreeCustom0,
		10,
		inputDataCustom0
	);

	IndexTree indexTreeCustom1;
	indexTreeCustom1.add({0, 1});
	indexTreeCustom1.add({1, 2});
	indexTreeCustom1.add({1, 3});
	int inputDataCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom1[n] = 2*n;
	indexTreeCustom1.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom1(
		indexTreeCustom1,
		10,
		inputDataCustom1
	);

	IndexTree indexTreeCustom2;
	indexTreeCustom2.add({0, 1});
	indexTreeCustom2.add({1, 2});
	int inputDataCustom2[2*10];
	for(unsigned int n = 0; n < 2*10; n++)
		inputDataCustom2[n] = n;
	indexTreeCustom2.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom2(
		indexTreeCustom2,
		10,
		inputDataCustom2
	);

	propertyCustom0 -= propertyCustom1;
	const std::vector<int> &dataCustom0 = propertyCustom0.getData();
	for(unsigned int n = 0; n < dataCustom0.size(); n++)
		EXPECT_EQ(dataCustom0[n], -n);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyCustom0 -= propertyCustom2;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 -= propertyRanges0;
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyNone0 -= propertyCustom0;
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Fail for different types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyRanges0 -= propertyCustom0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(AbstractProperty, operatorMultiplicationAssignment){
	//IndexDescriptor::Format::None.
	const int dataInputNone[3] = {0, 1, 2};
	PublicAbstractProperty<int> propertyNone(3, dataInputNone);

	propertyNone *= 2;
	const std::vector<int> &dataNone = propertyNone.getData();
	EXPECT_EQ(dataNone[0], 0);
	EXPECT_EQ(dataNone[1], 2);
	EXPECT_EQ(dataNone[2], 4);

	//IndexDescriptor::Format::Ranges.
	int dataInputRanges[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges[n] = n;
	PublicAbstractProperty<int> propertyRanges(
		{2, 3, 4},
		10,
		dataInputRanges
	);

	propertyRanges *= 3;
	const std::vector<int> &dataRanges = propertyRanges.getData();
	for(unsigned int n = 0; n < dataRanges.size(); n++)
		EXPECT_EQ(dataRanges[n], 3*n);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTreeCustom;
	indexTreeCustom.add({0, 1});
	indexTreeCustom.add({1, 2});
	indexTreeCustom.add({1, 3});
	int inputDataCustom[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom[n] = n;
	indexTreeCustom.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom(
		indexTreeCustom,
		10,
		inputDataCustom
	);
	propertyCustom.setAllowIndexOutOfBoundsAccess(true);
	propertyCustom.setDefaultValue(8);

	propertyCustom *= 4;
	const std::vector<int> &dataCustom = propertyCustom.getData();
	for(unsigned int n = 0; n < dataCustom.size(); n++)
		EXPECT_EQ(dataCustom[n], 4*n);
	EXPECT_EQ(propertyCustom({0}), 32);
}

TEST(AbstractProperty, operatorDivisionAssignment){
	//IndexDescriptor::Format::None.
	const int dataInputNone[3] = {0, 1, 2};
	PublicAbstractProperty<int> propertyNone(3, dataInputNone);

	propertyNone /= 2;
	const std::vector<int> &dataNone = propertyNone.getData();
	EXPECT_EQ(dataNone[0], 0);
	EXPECT_EQ(dataNone[1], 0);
	EXPECT_EQ(dataNone[2], 1);

	//IndexDescriptor::Format::Ranges.
	int dataInputRanges[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges[n] = n;
	PublicAbstractProperty<int> propertyRanges(
		{2, 3, 4},
		10,
		dataInputRanges
	);

	propertyRanges /= 3;
	const std::vector<int> &dataRanges = propertyRanges.getData();
	for(unsigned int n = 0; n < dataRanges.size(); n++)
		EXPECT_EQ(dataRanges[n], n/3);

	//IndexDescriptor::Format::Custom.
	IndexTree indexTreeCustom;
	indexTreeCustom.add({0, 1});
	indexTreeCustom.add({1, 2});
	indexTreeCustom.add({1, 3});
	int inputDataCustom[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		inputDataCustom[n] = n;
	indexTreeCustom.generateLinearMap();
	PublicAbstractProperty<int> propertyCustom(
		indexTreeCustom,
		10,
		inputDataCustom
	);
	propertyCustom.setAllowIndexOutOfBoundsAccess(true);
	propertyCustom.setDefaultValue(8);

	propertyCustom /= 4;
	const std::vector<int> &dataCustom = propertyCustom.getData();
	for(unsigned int n = 0; n < dataCustom.size(); n++)
		EXPECT_EQ(dataCustom[n], n/4);
	EXPECT_EQ(propertyCustom({0}), 2);
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
	PublicAbstractProperty<int> property(10);
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
	PublicAbstractProperty<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getDimensions();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({2, 3, 4}, 10);
	EXPECT_EQ(property1.getDimensions(), 3);

	//Fail for IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property2(indexTree, 10);
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
	PublicAbstractProperty<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getRanges();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({2, 3, 4}, 10);
	const std::vector<int> ranges = property1.getRanges();
	ASSERT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);

	//Fail for IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property2(indexTree, 10);
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
	PublicAbstractProperty<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.getOffset({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({2, 3, 4}, 10);
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
	PublicAbstractProperty<int> property2(indexTree, 10);
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
	PublicAbstractProperty<int> property0(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.contains({1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({2, 3, 4}, 10);
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
	PublicAbstractProperty<int> property2(indexTree, 10);
	EXPECT_TRUE(property2.contains({0}));
	EXPECT_TRUE(property2.contains({1}));
	EXPECT_TRUE(property2.contains({2}));
	EXPECT_FALSE(property2.contains({3}));
}

TEST(AbstractProperty, reduce){
	PublicAbstractProperty<int> property0(10);
	//Fail for IndexDescriptor::Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.reduce({{_a0_, _a0_}}, {{_a0_}});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({1, 1}, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property1.reduce({{_a0_, _a0_}}, {{_a0_}});
		},
		::testing::ExitedWithCode(1),
		""
	);

	IndexTree indexTree;
	for(int x = 0; x < 2; x++)
		for(int y = 0; y < 2; y++)
			for(int z = 0; z < 2; z++)
				indexTree.add({0, 1, x, y, z});
	for(int x = 0; x < 2; x++)
		for(int y = 0; y < 2; y++)
			indexTree.add({1, 2, x, y});
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property2(indexTree, 10);
	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(int z = 0; z < 2; z++){
				for(unsigned int n = 0; n < 10; n++){
					property2({0, 1, x, y, z}, n)
						= 10*(2*(2*x + y) + z) + n;
				}
			}
		}
	}
	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int n = 0; n < 10; n++){
				property2({1, 2, x, y}, n)
					= 80 + 10*(2*x + y) + n;
			}
		}
	}
	//Fail because two different Indices reduce to the same Index. For
	//example, here {0, 1, 0, 0, 0} and {1, 2, 0, 0} both reduce to {0, 0}.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property2.reduce(
				{
					{0, 1, _a0_, _a0_, _a1_},
					{1, 2, _a0_, _a0_}
				},
				{
					{_a0_, _a1_},
					{0, _a0_}
				}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Succeed with valid reduction. For example, here {0, 1, 0, 0, 1} ->
	//{0, 1, 0, 1} and {1, 2, 0, 0} -> {2, 0, 3}.
	property2.reduce(
		{
			{0, 1, _a0_, _a0_, _a1_},
			{1, 2, _a0_, _a0_}
		},
		{
			{0, 1, _a0_, _a1_},
			{2, _a0_, 3}
		}
	);
	for(int x = 0; x < 2; x++){
		int y = x;
		for(int z = 0; z < 2; z++){
			for(unsigned int n = 0; n < 10; n++){
				EXPECT_EQ(
					property2({0, 1, x, z}, n),
					10*(2*(2*x + y) + z) + n
				);
			}
		}
	}
	for(int x = 0; x < 2; x++){
		int y = x;
		for(unsigned int n = 0; n < 10; n++){
			EXPECT_EQ(
				property2({2, x, 3}, n),
				80 + 10*(2*x + y) + n
			);
		}
	}
}

TEST(AbstractProperty, hermitianConjugate){
	PublicAbstractProperty<std::complex<double>> property0(10);
	//Fail for IndexDescriptor::Format::None.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property0.hermitianConjugate();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<std::complex<double>> property1({1, 1}, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property1.hermitianConjugate();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail because the IndexTree contains an Index that is not a composit
	//Index with two component Indices.
	IndexTree indexTree2;
	indexTree2.add({0, 1});
	indexTree2.generateLinearMap();
	PublicAbstractProperty<std::complex<double>> property2(indexTree2, 10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property2.hermitianConjugate();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Suceed with valid Hermitian conjugation.
	IndexTree indexTree3;
	indexTree3.add({{0, 1}, {2, 3, 4}});
	indexTree3.add({{0, 2}, {1, 2}});
	indexTree3.add({{1, 2}, {0, 2}});
	indexTree3.generateLinearMap();
	PublicAbstractProperty<std::complex<double>> property3(indexTree3, 10);
	for(unsigned int n = 0; n < 10; n++){
		property3({{0, 1}, {2, 3, 4}}, n) = std::complex<double>(n, n);
		property3({{0, 2}, {1, 2}}, n)
			= std::complex<double>(2*n, 3*n);
		property3({{1, 2}, {0, 2}}, n)
			= std::complex<double>(4*n, 5*n);
	}
	property3.hermitianConjugate();
	for(int n = 0; n < 10; n++){
		EXPECT_DOUBLE_EQ(real(property3({{2, 3, 4}, {0, 1}}, n)), n);
		EXPECT_DOUBLE_EQ(imag(property3({{2, 3, 4}, {0, 1}}, n)), -n);
		EXPECT_DOUBLE_EQ(real(property3({{1, 2}, {0, 2}}, n)), 2*n);
		EXPECT_DOUBLE_EQ(imag(property3({{1, 2}, {0, 2}}, n)), -3*n);
		EXPECT_DOUBLE_EQ(real(property3({{0, 2}, {1, 2}}, n)), 4*n);
		EXPECT_DOUBLE_EQ(imag(property3({{0, 2}, {1, 2}}, n)), -5*n);
	}

	//Fail to access Index present in the orginal property, but not in the
	//Hermitian conjugated property.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			property3({{0, 1}, {2, 3, 4}}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(AbstractProperty, operatorFunction){
	PublicAbstractProperty<int> property0(10);
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
			const_cast<const PublicAbstractProperty<int>&>(
				property0
			)({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Non-indexed version.
	property0(0) = 0;
	property0(1) = 1;
	property0(2) = 2;
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property0)(0),
		0
	);
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property0)(1),
		1
	);
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property0)(2),
		2
	);

	//Fail to use indexed version for IndexDescriptor::Format::Ranges.
	PublicAbstractProperty<int> property1({2, 3, 4}, 10);
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
			const_cast<const PublicAbstractProperty<int>&>(
				property1
			)({1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//Non-indexed version
	property1(0) = 0;
	property1(1) = 1;
	property1(2) = 2;
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property1)(0),
		0
	);
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property1)(1),
		1
	);
	EXPECT_EQ(
		const_cast<const PublicAbstractProperty<int>&>(property1)(2),
		2
	);

	//IndexDescriptor::Format::Custom. Normal behavior.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	PublicAbstractProperty<int> property2(indexTree, 10);
	property2({0}, 1) = 1;
	property2({1}, 2) = 2;
	property2({2}, 3) = 3;
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)({0}, 1),
		1
	);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)({1}, 2),
		2
	);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)({2}, 3),
		3
	);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)(1),
		1
	);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)(12),
		2
	);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)(23),
		3
	);
	property2(12) = 12;
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)(12),
		12
	);
	//IndexDescriptor::Format::Custom. Check index out-of-bounds behavior.
	property2.setDefaultValue(99);
	EXPECT_THROW(property2({3}, 1), IndexException);
	EXPECT_THROW(
		const_cast<PublicAbstractProperty<int>&>(property2)({3}, 1),
		IndexException
	);
	property2.setAllowIndexOutOfBoundsAccess(true);
	EXPECT_EQ(property2({3}, 0), 99);
	EXPECT_EQ(
		const_cast<PublicAbstractProperty<int>&>(property2)({3}, 0),
		99
	);
	property2({3}, 0) = 0;
	EXPECT_EQ(property2({3}, 0), 99);
	//Check that the correct overloaded opeartor is called. An ambiguity
	//can arise since {1} can be cast to 1 instead of Index({1}).
	property2(1) = 100;
	property2(10) = 1000;
	EXPECT_EQ(property2(1), 100);
	EXPECT_EQ(property2({1}), 1000);
}

TEST(AbstractProperty, setAllowOutOfBoundsAccess){
	//Already tested through AbstractProperty;;operatorFunction
}

TEST(AbstractProperty, setDefaultValue){
	//Already tested through AbstractProperty;;operatorFunction
}

TEST(AbstractProperty, replaceValues){
	PublicAbstractProperty<int> property0(10);

	for(unsigned int n = 0; n < 10; n++)
		property0(n) = n;

	property0.replaceValues(5, 25);

	for(unsigned int n = 0; n < 10; n++){
		if(n == 5)
			EXPECT_EQ(property0(n), 25);
		else
			EXPECT_EQ(property0(n), n);
	}

	PublicAbstractProperty<double> property1(4);
	property1(0) = NAN;
	property1(1) = -NAN;
	property1(2) = std::numeric_limits<double>::infinity();
	property1(3) = -std::numeric_limits<double>::infinity();

	property1.replaceValues(NAN, 1);
	property1.replaceValues(std::numeric_limits<double>::infinity(), 2);
	property1.replaceValues(-std::numeric_limits<double>::infinity(), 3);
	EXPECT_DOUBLE_EQ(property1(0), 1);
	EXPECT_DOUBLE_EQ(property1(1), 1);
	EXPECT_DOUBLE_EQ(property1(2), 2);
	EXPECT_DOUBLE_EQ(property1(3), 3);
}

//TODO
//This function is currently not implemented for all data types. First
//implement full support for all primitive data types, Serializable, and
//pseudo-Serializable classes.
TEST(AbstractProperty, serialize){
}

};	//End of namespace Property
};	//End of namespace TBTK
