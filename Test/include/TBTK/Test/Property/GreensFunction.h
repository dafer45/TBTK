#include "TBTK/Property/GreensFunction.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(GreensFunction, Constructor0){
	//Just verify that this compiles.
//	GreensFunction greensFunction;
}

TEST(GreensFunction, Constructor1){
	IndexTree indexTree;
	indexTree.add({{0, 1}, {0, 1}});
	indexTree.add({{0, 2}, {0, 1}});
	indexTree.add({{0, 1}, {0, 3}});
	indexTree.generateLinearMap();
	GreensFunction greensFunction0(
		indexTree,
		GreensFunction::Type::Retarded,
		-10,
		10,
		1000
	);
	ASSERT_EQ(greensFunction0.getBlockSize(), 1000);
	ASSERT_EQ(greensFunction0.getResolution(), 1000);
	ASSERT_EQ(greensFunction0.getSize(), 3*1000);
	for(unsigned int n = 0; n < greensFunction0.getResolution(); n++){
		EXPECT_DOUBLE_EQ(
			real(greensFunction0({{0, 1}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(greensFunction0({{0, 1}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(greensFunction0({{0, 2}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(greensFunction0({{0, 2}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(greensFunction0({{0, 1}, {0, 3}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(greensFunction0({{0, 1}, {0, 3}}, n)),
			0
		);
	}
	EXPECT_DOUBLE_EQ(greensFunction0.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(greensFunction0.getUpperBound(), 10);
	EXPECT_EQ(greensFunction0.getType(), GreensFunction::Type::Retarded);

	GreensFunction greensFunction1(
		indexTree,
		GreensFunction::Type::Principal,
		-10,
		10,
		1000
	);
	EXPECT_EQ(greensFunction1.getType(), GreensFunction::Type::Principal);
}

TEST(GreensFunction, operatorAdditionAssignment){
	IndexTree indexTree;
	indexTree.add({{0, 1}, {0, 1}});
	indexTree.add({{0, 2}, {0, 1}});
	indexTree.add({{0, 1}, {0, 3}});
	indexTree.generateLinearMap();

	std::complex<double> inputData0[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData0[n] = n;
	GreensFunction greensFunction0(
		indexTree,
		GreensFunction::Type::Retarded,
		-10,
		10,
		1000,
		inputData0
	);

	std::complex<double> inputData1[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData1[n] = 2*n;
	GreensFunction greensFunction1(
		indexTree,
		GreensFunction::Type::Retarded,
		-10,
		10,
		1000,
		inputData1
	);

	std::complex<double> inputData2[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData2[n] = 3*n;
	GreensFunction greensFunction2(
		indexTree,
		GreensFunction::Type::Advanced,
		-10,
		10,
		1000,
		inputData2
	);

	greensFunction0 += greensFunction1;
	std::vector<std::complex<double>> data0 = greensFunction0.getData();
	for(unsigned int n = 0; n < 3000; n++){
		EXPECT_DOUBLE_EQ(real(data0[n]), 3*n);
		EXPECT_DOUBLE_EQ(imag(data0[n]), 0);
	}

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			greensFunction0 += greensFunction2;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(GreensFunction, operatorSubtractionAssignment){
	IndexTree indexTree;
	indexTree.add({{0, 1}, {0, 1}});
	indexTree.add({{0, 2}, {0, 1}});
	indexTree.add({{0, 1}, {0, 3}});
	indexTree.generateLinearMap();

	std::complex<double> inputData0[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData0[n] = n;
	GreensFunction greensFunction0(
		indexTree,
		GreensFunction::Type::Retarded,
		-10,
		10,
		1000,
		inputData0
	);

	std::complex<double> inputData1[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData1[n] = 2*n;
	GreensFunction greensFunction1(
		indexTree,
		GreensFunction::Type::Retarded,
		-10,
		10,
		1000,
		inputData1
	);

	std::complex<double> inputData2[3*1000];
	for(unsigned int n = 0; n < 3000; n++)
		inputData2[n] = 3*n;
	GreensFunction greensFunction2(
		indexTree,
		GreensFunction::Type::Advanced,
		-10,
		10,
		1000,
		inputData2
	);

	greensFunction0 -= greensFunction1;
	std::vector<std::complex<double>> data0 = greensFunction0.getData();
	for(int n = 0; n < 3000; n++){
		EXPECT_DOUBLE_EQ(real(data0[n]), -n);
		EXPECT_DOUBLE_EQ(imag(data0[n]), 0);
	}

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			greensFunction0 -= greensFunction2;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(GreensFunction, getType){
	//Already tested through
	//GreensFunction::Constructor0
	//GreensFunction::Constructor1
}

};	//End of namespace Property
};	//End of namespace TBTK
