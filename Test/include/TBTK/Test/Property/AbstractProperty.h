#include "TBTK/Property/AbstractProperty.h"

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
		AbstractProperty<DataType>(
			blockSize
		){
	}
	Property(
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(
			blockSize,
			data
		){
	}
	Property(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(
			dimensions,
			ranges,
			blockSize
		){
	}
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
		){
	}
	Property(
		const IndexTree &indexTree,
		unsigned int blockSize
	) :
		AbstractProperty<DataType>(
			indexTree,
			blockSize
		){
	}
	Property(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const DataType *data
	) :
		AbstractProperty<DataType>(
			indexTree,
			blockSize,
			data
		){
	}
};

TEST(AbstractProperty, Constructor0){
	//Construct an uninitialized Property.
	Property<int> property;
	EXPECT_EQ(property.getBlockSize(), 0);
	EXPECT_EQ(property.getSize(), 0);
	EXPECT_EQ(property.getData(), nullptr);
}

TEST(AbstractProperty, Constructor1){
	//Construct a Property with a single data block.
	Property<int> property(10);
	EXPECT_EQ(property.getBlockSize(), 10);
	EXPECT_EQ(property.getSize(), 10);
	EXPECT_FALSE(property.getData() == nullptr);
}

TEST(AbstractProperty, Constructor2){
	//Construct a Property with a single data block and initialize it with
	//privided values.
	const int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	Property<int> property(10, data);
	ASSERT_EQ(property.getBlockSize(), 10);
	ASSERT_EQ(property.getSize(), 10);
	const int *internalData = property.getData();
	EXPECT_EQ(internalData[0], 0);
	EXPECT_EQ(internalData[1], 1);
	EXPECT_EQ(internalData[2], 2);
	EXPECT_EQ(internalData[9], 9);
}

};	//End of namespace Solver
};	//End of namespace TBTK
