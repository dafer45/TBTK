#include "TBTK/Property/LDOS.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(LDOS, Constructor0){
	LDOS ldos({2, 3, 4}, -10, 10, 1000);
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
	double dataInput[1000*2*3*4];
	for(unsigned int n = 0; n < 1000*2*3*4; n++)
		dataInput[n] = n;
	LDOS ldos({2, 3, 4}, -10, 10, 1000, dataInput);
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
	for(unsigned int n = 0; n < ldos.getResolution(); n++){
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
	for(unsigned int n = 0; n < ldos.getResolution(); n++){
		EXPECT_DOUBLE_EQ(ldos({0}, n), n);
		EXPECT_DOUBLE_EQ(ldos({1}, n), n+1000);
		EXPECT_DOUBLE_EQ(ldos({2}, n), n+2000);
	}
}

TEST(LDOS, operatorAdditionAssignment){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	double dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	LDOS ldosRanges1({2, 3, 4}, -10, 10, 10, dataInputRanges1);

	ldosRanges0 += ldosRanges1;
	const std::vector<double> dataRanges0 = ldosRanges0.getData();
	for(unsigned int n = 0; n < dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], 3*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	double dataInputCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom1[n] = 2*n;
	LDOS ldosCustom1(indexTree, -10, 10, 10, dataInputCustom1);

	ldosCustom0 += ldosCustom1;
	const std::vector<double> dataCustom0 = ldosCustom0.getData();
	for(unsigned int n = 0; n < dataCustom0.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom0[n], 3*n);
}

TEST(LDOS, operatorAddition){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	double dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	LDOS ldosRanges1({2, 3, 4}, -10, 10, 10, dataInputRanges1);

	LDOS ldosRanges2 = ldosRanges0 + ldosRanges1;
	const std::vector<double> dataRanges2 = ldosRanges2.getData();
	for(unsigned int n = 0; n < dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], 3*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	double dataInputCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom1[n] = 2*n;
	LDOS ldosCustom1(indexTree, -10, 10, 10, dataInputCustom1);

	LDOS ldosCustom2 = ldosCustom0 + ldosCustom1;
	const std::vector<double> dataCustom2 = ldosCustom2.getData();
	for(unsigned int n = 0; n < dataCustom2.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom2[n], 3*n);
}

TEST(LDOS, operatorSubtractionAssignment){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	double dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	LDOS ldosRanges1({2, 3, 4}, -10, 10, 10, dataInputRanges1);

	ldosRanges0 -= ldosRanges1;
	const std::vector<double> dataRanges0 = ldosRanges0.getData();
	for(int n = 0; n < (int)dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], -n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	double dataInputCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom1[n] = 2*n;
	LDOS ldosCustom1(indexTree, -10, 10, 10, dataInputCustom1);

	ldosCustom0 -= ldosCustom1;
	const std::vector<double> dataCustom0 = ldosCustom0.getData();
	for(int n = 0; n < (int)dataCustom0.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom0[n], -n);
}

TEST(LDOS, operatorSubtraction){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	double dataInputRanges1[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges1[n] = 2*n;
	LDOS ldosRanges1({2, 3, 4}, -10, 10, 10, dataInputRanges1);

	LDOS ldosRanges2 = ldosRanges0 - ldosRanges1;
	const std::vector<double> dataRanges2 = ldosRanges2.getData();
	for(int n = 0; n < (int)dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], -n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	double dataInputCustom1[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom1[n] = 2*n;
	LDOS ldosCustom1(indexTree, -10, 10, 10, dataInputCustom1);

	LDOS ldosCustom2 = ldosCustom0 - ldosCustom1;
	const std::vector<double> dataCustom2 = ldosCustom2.getData();
	for(int n = 0; n < (int)dataCustom2.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom2[n], -n);
}

TEST(LDOS, operatorMultiplicationAssignment){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	ldosRanges0 *= 2;
	const std::vector<double> dataRanges0 = ldosRanges0.getData();
	for(int n = 0; n < (int)dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], 2*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	ldosCustom0 *= 3;
	const std::vector<double> dataCustom0 = ldosCustom0.getData();
	for(int n = 0; n < (int)dataCustom0.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom0[n], 3*n);
}

TEST(LDOS, operatorMultiplication){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	LDOS ldosRanges1 = ldosRanges0*2;
	LDOS ldosRanges2 = 2*ldosRanges0;
	const std::vector<double> dataRanges1 = ldosRanges1.getData();
	for(int n = 0; n < (int)dataRanges1.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges1[n], 2*n);
	const std::vector<double> dataRanges2 = ldosRanges2.getData();
	for(int n = 0; n < (int)dataRanges2.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges2[n], 2*n);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	LDOS ldosCustom1 = ldosCustom0*3;
	LDOS ldosCustom2 = 3*ldosCustom0;
	const std::vector<double> dataCustom2 = ldosCustom2.getData();
	for(int n = 0; n < (int)dataCustom2.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom2[n], 3*n);
}

TEST(LDOS, operatorDivisionAssignment){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	ldosRanges0 /= 2;
	const std::vector<double> dataRanges0 = ldosRanges0.getData();
	for(int n = 0; n < (int)dataRanges0.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges0[n], n/2.);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	ldosCustom0 /= 3;
	const std::vector<double> dataCustom0 = ldosCustom0.getData();
	for(int n = 0; n < (int)dataCustom0.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom0[n], n/3.);
}

TEST(LDOS, operatorDivision){
	//Ranges format.
	double dataInputRanges0[2*3*4*10];
	for(unsigned int n = 0; n < 2*3*4*10; n++)
		dataInputRanges0[n] = n;
	LDOS ldosRanges0({2, 3, 4}, -10, 10, 10, dataInputRanges0);

	LDOS ldosRanges1 = ldosRanges0/2;
	const std::vector<double> dataRanges1 = ldosRanges1.getData();
	for(int n = 0; n < (int)dataRanges1.size(); n++)
		EXPECT_DOUBLE_EQ(dataRanges1[n], n/2.);

	//Custom format.
	IndexTree indexTree;
	indexTree.add({1, 2, 3});
	indexTree.add({1, 2, 4});
	indexTree.add({2, 2});
	indexTree.generateLinearMap();

	double dataInputCustom0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		dataInputCustom0[n] = n;
	LDOS ldosCustom0(indexTree, -10, 10, 10, dataInputCustom0);

	LDOS ldosCustom1 = ldosCustom0/3;
	const std::vector<double> dataCustom1 = ldosCustom1.getData();
	for(int n = 0; n < (int)dataCustom1.size(); n++)
		EXPECT_DOUBLE_EQ(dataCustom1[n], n/3.);
}

TEST(LDOS, SerializeToJSON){
	//IndexDescriptor::Format::Ranges.
	double dataInput0[1000*2*3*4];
	for(unsigned int n = 0; n < 1000*2*3*4; n++)
		dataInput0[n] = n;
	LDOS ldos0({2, 3, 4}, -10, 10, 1000, dataInput0);
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
	for(unsigned int n = 0; n < ldos3.getResolution(); n++){
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
