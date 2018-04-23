#include "TBTK/Property/Density.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(Density, Constructor0){
	int ranges[3] = {2, 3, 4};
	Density density(3, ranges);
	ASSERT_EQ(density.getDimensions(), 3);
	ASSERT_EQ(density.getRanges()[0], 2);
	ASSERT_EQ(density.getRanges()[1], 3);
	ASSERT_EQ(density.getRanges()[2], 4);
	ASSERT_EQ(density.getBlockSize(), 1);

	const double *data = density.getData();
	for(unsigned int n = 0; n < 2*3*4; n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(Density, Constructor1){
	int ranges[3] = {2, 3, 4};
	double dataInput[2*3*4];
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInput[n] = n;
	Density density(3, ranges, dataInput);
	ASSERT_EQ(density.getDimensions(), 3);
	ASSERT_EQ(density.getRanges()[0], 2);
	ASSERT_EQ(density.getRanges()[1], 3);
	ASSERT_EQ(density.getRanges()[2], 4);
	ASSERT_EQ(density.getBlockSize(), 1);

	const double *data = density.getData();
	for(unsigned int n = 0; n < 2*3*4; n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(Density, Constructor2){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	Density density(indexTree);
	EXPECT_DOUBLE_EQ(density({1, 2, 3}), 0);
	EXPECT_DOUBLE_EQ(density({1, 2, 4}), 0);
	EXPECT_DOUBLE_EQ(density({2, 2}), 0);
}

TEST(Density, Constructor3){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	double data[3] = {1, 2, 3};
	Density density(indexTree, data);
	EXPECT_DOUBLE_EQ(density({1, 2, 3}), 1);
	EXPECT_DOUBLE_EQ(density({1, 2, 4}), 2);
	EXPECT_DOUBLE_EQ(density({2, 2}), 3);
}

TEST(Density, SerializeToJSON){
	//Ranges format.
	int ranges[3] = {2, 3, 4};
	double dataInput[2*3*4];
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInput[n] = n;
	Density density0(3, ranges, dataInput);
	Density density1(
		density0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(density1.getDimensions(), 3);
	ASSERT_EQ(density1.getRanges()[0], 2);
	ASSERT_EQ(density1.getRanges()[1], 3);
	ASSERT_EQ(density1.getRanges()[2], 4);
	ASSERT_EQ(density1.getBlockSize(), 1);

	const double *data = density1.getData();
	for(unsigned int n = 0; n < 2*3*4; n++)
		EXPECT_DOUBLE_EQ(data[n], n);

	//Custom format
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	double data2[3] = {1, 2, 3};
	Density density2(indexTree, data2);
	Density density3(
		density2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_DOUBLE_EQ(density3({1, 2, 3}), 1);
	EXPECT_DOUBLE_EQ(density3({1, 2, 4}), 2);
	EXPECT_DOUBLE_EQ(density3({2, 2}), 3);
}

TEST(Density, getMin){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	double data[3] = {1, 2, 3};
	Density density(indexTree, data);
	EXPECT_DOUBLE_EQ(density.getMin(), 1);
	EXPECT_DOUBLE_EQ(density.getMax(), 3);
}

TEST(Denity, serialize){
	//Already tested through SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
