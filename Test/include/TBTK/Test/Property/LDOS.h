#include "TBTK/Property/LDOS.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(LDOS, Constructor0){
	int ranges[3] = {2, 3, 4};
	LDOS ldos(3, ranges, -10, 10, 1000);
	ASSERT_EQ(ldos.getDimensions(), 3);
	EXPECT_EQ(ldos.getRanges()[0], 2);
	EXPECT_EQ(ldos.getRanges()[1], 3);
	EXPECT_EQ(ldos.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(ldos.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos.getUpperBound(), 10);
	ASSERT_EQ(ldos.getResolution(), 1000);
	ASSERT_EQ(ldos.getSize(), 1000*2*3*4);
	const std::vector<double> &data = ldos.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(LDOS, Constructor1){
	int ranges[3] = {2, 3, 4};
	double dataInput[1000*2*3*4];
	for(unsigned int n = 0; n < 1000*2*3*4; n++)
		dataInput[n] = n;
	LDOS ldos(3, ranges, -10, 10, 1000, dataInput);
	ASSERT_EQ(ldos.getDimensions(), 3);
	EXPECT_EQ(ldos.getRanges()[0], 2);
	EXPECT_EQ(ldos.getRanges()[1], 3);
	EXPECT_EQ(ldos.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(ldos.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos.getUpperBound(), 10);
	ASSERT_EQ(ldos.getResolution(), 1000);
	ASSERT_EQ(ldos.getSize(), 1000*2*3*4);
	const std::vector<double> &data = ldos.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(LDOS, Constructor2){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	LDOS ldos(indexTree, -10, 10, 1000);
	EXPECT_DOUBLE_EQ(ldos.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos.getUpperBound(), 10);
	ASSERT_EQ(ldos.getResolution(), 1000);
	ASSERT_EQ(ldos.getSize(), 1000*3);
	for(int n = 0; n < ldos.getResolution(); n++){
		EXPECT_DOUBLE_EQ(ldos({0}, n), 0);
		EXPECT_DOUBLE_EQ(ldos({1}, n), 0);
		EXPECT_DOUBLE_EQ(ldos({2}, n), 0);
	}
}

TEST(LDOS, Constructor3){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	double dataInput[1000*3];
	for(unsigned int n = 0; n < 1000*3; n++)
		dataInput[n] = n;
	LDOS ldos(indexTree, -10, 10, 1000, dataInput);
	EXPECT_DOUBLE_EQ(ldos.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos.getUpperBound(), 10);
	ASSERT_EQ(ldos.getResolution(), 1000);
	ASSERT_EQ(ldos.getSize(), 1000*3);
	for(int n = 0; n < ldos.getResolution(); n++){
		EXPECT_DOUBLE_EQ(ldos({0}, n), n);
		EXPECT_DOUBLE_EQ(ldos({1}, n), n+1000);
		EXPECT_DOUBLE_EQ(ldos({2}, n), n+2000);
	}
}

TEST(LDOS, SerializeToJSON){
	//IndexDescriptor::Format::Ranges.
	int ranges[3] = {2, 3, 4};
	double dataInput0[1000*2*3*4];
	for(unsigned int n = 0; n < 1000*2*3*4; n++)
		dataInput0[n] = n;
	LDOS ldos0(3, ranges, -10, 10, 1000, dataInput0);
	LDOS ldos1(
		ldos0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(ldos1.getDimensions(), 3);
	EXPECT_EQ(ldos1.getRanges()[0], 2);
	EXPECT_EQ(ldos1.getRanges()[1], 3);
	EXPECT_EQ(ldos1.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(ldos1.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos1.getUpperBound(), 10);
	ASSERT_EQ(ldos1.getResolution(), 1000);
	ASSERT_EQ(ldos1.getSize(), 1000*2*3*4);
	const std::vector<double> &data1 = ldos1.getData();
	for(unsigned int n = 0; n < data1.size(); n++)
		EXPECT_DOUBLE_EQ(data1[n], n);

	//IndexDescriptor::Format::Custom
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	double dataInput2[1000*3];
	for(unsigned int n = 0; n < 1000*3; n++)
		dataInput2[n] = n;
	LDOS ldos2(indexTree, -10, 10, 1000, dataInput2);
	LDOS ldos3(
		ldos2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_DOUBLE_EQ(ldos3.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(ldos3.getUpperBound(), 10);
	ASSERT_EQ(ldos3.getResolution(), 1000);
	ASSERT_EQ(ldos3.getSize(), 1000*3);
	for(int n = 0; n < ldos3.getResolution(); n++){
		EXPECT_DOUBLE_EQ(ldos3({0}, n), n);
		EXPECT_DOUBLE_EQ(ldos3({1}, n), n+1000);
		EXPECT_DOUBLE_EQ(ldos3({2}, n), n+2000);
	}
}

TEST(LDOS, getLowerBound){
	//Already tested through
	//LDOS::Constructor0
	//LDOS::Constructor1
	//LDOS::Constructor2
	//LDOS::Constructor3
	//LDOS::SerializeToJSON
}

TEST(LDOS, getUpperBound){
	//Already tested through
	//LDOS::Constructor0
	//LDOS::Constructor1
	//LDOS::Constructor2
	//LDOS::Constructor3
	//LDOS::SerializeToJSON
}

TEST(LDOS, getResolution){
	//Already tested through
	//LDOS::Constructor0
	//LDOS::Constructor1
	//LDOS::Constructor2
	//LDOS::Constructor3
	//LDOS::SerializeToJSON
}

TEST(LDOS, serialize){
	//Already tested through
	//LDOS::SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
