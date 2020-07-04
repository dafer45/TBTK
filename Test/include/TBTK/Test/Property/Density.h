#include "TBTK/Property/Density.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(Density, Constructor0){
	Density density({2, 3, 4});
	ASSERT_EQ(density.getDimensions(), 3);
	ASSERT_EQ(density.getRanges()[0], 2);
	ASSERT_EQ(density.getRanges()[1], 3);
	ASSERT_EQ(density.getRanges()[2], 4);
	ASSERT_EQ(density.getBlockSize(), 1);
	ASSERT_EQ(density.getSize(), 2*3*4);

	const std::vector<double> &data = density.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(Density, Constructor1){
	CArray<double> dataInput(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInput[n] = n;
	Density density({2, 3, 4}, dataInput);
	ASSERT_EQ(density.getDimensions(), 3);
	ASSERT_EQ(density.getRanges()[0], 2);
	ASSERT_EQ(density.getRanges()[1], 3);
	ASSERT_EQ(density.getRanges()[2], 4);
	ASSERT_EQ(density.getBlockSize(), 1);
	ASSERT_EQ(density.getSize(), 2*3*4);

	const std::vector<double> &data = density.getData();
	for(unsigned int n = 0; n < data.size(); n++)
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
	CArray<double> data = {1, 2, 3};
	Density density(indexTree, data);
	EXPECT_DOUBLE_EQ(density({1, 2, 3}), 1);
	EXPECT_DOUBLE_EQ(density({1, 2, 4}), 2);
	EXPECT_DOUBLE_EQ(density({2, 2}), 3);
}

TEST(Density, SerializeToJSON){
	//Ranges format.
	CArray<double> dataInput(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInput[n] = n;
	Density density0({2, 3, 4}, dataInput);
	Density density1(
		density0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(density1.getDimensions(), 3);
	ASSERT_EQ(density1.getRanges()[0], 2);
	ASSERT_EQ(density1.getRanges()[1], 3);
	ASSERT_EQ(density1.getRanges()[2], 4);
	ASSERT_EQ(density1.getBlockSize(), 1);
	ASSERT_EQ(density1.getSize(), 2*3*4);

	const std::vector<double> &data = density1.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);

	//Custom format
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	CArray<double> data2 = {1, 2, 3};
	Density density2(indexTree, data2);
	Density density3(
		density2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_DOUBLE_EQ(density3({1, 2, 3}), 1);
	EXPECT_DOUBLE_EQ(density3({1, 2, 4}), 2);
	EXPECT_DOUBLE_EQ(density3({2, 2}), 3);
}

TEST(Density, operatorAdditionAssignment){
	//Ranges format.
	CArray<double> dataInputRanges0(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges0[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges0);

	CArray<double> dataInputRanges1(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges1[n] = 2*n;
	Density densityRanges1({2, 3, 4}, dataInputRanges1);

	densityRanges0 += densityRanges1;
	const std::vector<double> &dataRanges0 = densityRanges0.getData();
	for(unsigned int n = 0; n < dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], 3*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom0 = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom0);

	CArray<double> dataCustom1 = {2, 4, 6};
	Density densityCustom1(indexTree, dataCustom1);

	densityCustom0 += densityCustom1;
	EXPECT_DOUBLE_EQ(densityCustom0({1, 2, 3}), 3);
	EXPECT_DOUBLE_EQ(densityCustom0({1, 2, 4}), 6);
	EXPECT_DOUBLE_EQ(densityCustom0({2, 2}), 9);
}

TEST(Density, operatorAddition){
	//Ranges format.
	CArray<double> dataInputRanges0(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges0[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges0);

	CArray<double> dataInputRanges1(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges1[n] = 2*n;
	Density densityRanges1({2, 3, 4}, dataInputRanges1);

	Density densityRanges2 = densityRanges0 + densityRanges1;
	const std::vector<double> &dataRanges2 = densityRanges2.getData();
	for(unsigned int n = 0; n < dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], 3*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom0 = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom0);

	CArray<double> dataCustom1 = {2, 4, 6};
	Density densityCustom1(indexTree, dataCustom1);

	Density densityCustom2 = densityCustom0 + densityCustom1;
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 3}), 3);
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 4}), 6);
	EXPECT_DOUBLE_EQ(densityCustom2({2, 2}), 9);
}

TEST(Density, operatorSubtractionAssignment){
	//Ranges format.
	CArray<double> dataInputRanges0(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges0[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges0);

	CArray<double> dataInputRanges1(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges1[n] = 2*n;
	Density densityRanges1({2, 3, 4}, dataInputRanges1);

	densityRanges0 -= densityRanges1;
	const std::vector<double> &dataRanges0 = densityRanges0.getData();
	for(int n = 0; n < (int)dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], -n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom0 = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom0);

	CArray<double> dataCustom1 = {2, 4, 6};
	Density densityCustom1(indexTree, dataCustom1);

	densityCustom0 -= densityCustom1;
	EXPECT_DOUBLE_EQ(densityCustom0({1, 2, 3}), -1);
	EXPECT_DOUBLE_EQ(densityCustom0({1, 2, 4}), -2);
	EXPECT_DOUBLE_EQ(densityCustom0({2, 2}), -3);
}

TEST(Density, operatorSubtraction){
	//Ranges format.
	CArray<double> dataInputRanges0(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges0[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges0);

	CArray<double> dataInputRanges1(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges1[n] = 2*n;
	Density densityRanges1({2, 3, 4}, dataInputRanges1);

	Density densityRanges2 = densityRanges0 - densityRanges1;
	const std::vector<double> &dataRanges2 = densityRanges2.getData();
	for(int n = 0; n < (int)dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], -n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom0 = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom0);

	CArray<double> dataCustom1 = {2, 4, 6};
	Density densityCustom1(indexTree, dataCustom1);

	Density densityCustom2 = densityCustom0 - densityCustom1;
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 3}), -1);
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 4}), -2);
	EXPECT_DOUBLE_EQ(densityCustom2({2, 2}), -3);
}

TEST(Density, operatorMultiplicationAssignment){
	//Ranges format.
	CArray<double> dataInputRanges(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges[n] = n;
	Density densityRanges({2, 3, 4}, dataInputRanges);

	densityRanges *= 2;
	const std::vector<double> &dataRanges = densityRanges.getData();
	for(unsigned int n = 0; n < dataRanges.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges[n], 2*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom = {1, 2, 3};
	Density densityCustom(indexTree, dataCustom);

	densityCustom *= 3;
	EXPECT_DOUBLE_EQ(densityCustom({1, 2, 3}), 3);
	EXPECT_DOUBLE_EQ(densityCustom({1, 2, 4}), 6);
	EXPECT_DOUBLE_EQ(densityCustom({2, 2}), 9);
}

TEST(Density, operatorMultiplication){
	//Ranges format.
	CArray<double> dataInputRanges(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges);

	Density densityRanges1 = densityRanges0*2;
	Density densityRanges2 = 2*densityRanges0;
	const std::vector<double> &dataRanges1 = densityRanges1.getData();
	const std::vector<double> &dataRanges2 = densityRanges2.getData();
	for(unsigned int n = 0; n < dataRanges1.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges1[n], 2*n);
	for(unsigned int n = 0; n < dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], 2*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom);

	Density densityCustom1 = densityCustom0*3;
	Density densityCustom2 = 3*densityCustom0;
	EXPECT_DOUBLE_EQ(densityCustom1({1, 2, 3}), 3);
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 3}), 3);
	EXPECT_DOUBLE_EQ(densityCustom1({1, 2, 4}), 6);
	EXPECT_DOUBLE_EQ(densityCustom2({1, 2, 4}), 6);
	EXPECT_DOUBLE_EQ(densityCustom1({2, 2}), 9);
	EXPECT_DOUBLE_EQ(densityCustom2({2, 2}), 9);
}

TEST(Density, operatorDivisionAssignment){
	//Ranges format.
	CArray<double> dataInputRanges(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges[n] = n;
	Density densityRanges({2, 3, 4}, dataInputRanges);

	densityRanges /= 2;
	const std::vector<double> &dataRanges = densityRanges.getData();
	for(unsigned int n = 0; n < dataRanges.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges[n], n/2.);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom = {1, 2, 3};
	Density densityCustom(indexTree, dataCustom);

	densityCustom /= 3;
	EXPECT_DOUBLE_EQ(densityCustom({1, 2, 3}), 1/3.);
	EXPECT_DOUBLE_EQ(densityCustom({1, 2, 4}), 2/3.);
	EXPECT_DOUBLE_EQ(densityCustom({2, 2}), 1);
}

TEST(Density, operatorDivision){
	//Ranges format.
	CArray<double> dataInputRanges(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++)
		dataInputRanges[n] = n;
	Density densityRanges0({2, 3, 4}, dataInputRanges);

	Density densityRanges1 = densityRanges0/2;
	const std::vector<double> &dataRanges1 = densityRanges1.getData();
	for(unsigned int n = 0; n < dataRanges1.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges1[n], n/2.);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	CArray<double> dataCustom = {1, 2, 3};
	Density densityCustom0(indexTree, dataCustom);

	Density densityCustom1 = densityCustom0/3;
	EXPECT_DOUBLE_EQ(densityCustom1({1, 2, 3}), 1/3.);
	EXPECT_DOUBLE_EQ(densityCustom1({1, 2, 4}), 2/3.);
	EXPECT_DOUBLE_EQ(densityCustom1({2, 2}), 1);
}

TEST(Density, getMin){
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();
	CArray<double> data = {1, 2, 3};
	Density density(indexTree, data);
	EXPECT_DOUBLE_EQ(density.getMin(), 1);
	EXPECT_DOUBLE_EQ(density.getMax(), 3);
}

TEST(Density, toString){
	//Difficult to formulate a good test criterium for.
}

TEST(Denity, serialize){
	//Already tested through SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
