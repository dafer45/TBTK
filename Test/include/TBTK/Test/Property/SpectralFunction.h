#include "TBTK/Property/SpectralFunctionn.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

//TBTKFeature Property.SpectralFunction.construction.1 2020-02-08
TEST(SpectralFunction, Constructor0){
	//Just verify that this compiles.
	SpectralFunction spectralFunction;
}

//TBTKFeature Property.SpectralFunction.construction.2 2020-02-08
TEST(GreensFunction, Constructor2){
	IndexTree indexTree;
	indexTree.add({{0, 1}, {0, 1}});
	indexTree.add({{0, 2}, {0, 1}});
	indexTree.add({{0, 1}, {0, 3}});
	indexTree.generateLinearMap();
	SpectralFunction spectralFunction(
		indexTree,
		-10,
		10,
		1000
	);
	ASSERT_EQ(spectralFunction.getBlockSize(), 1000);
	ASSERT_EQ(spectralFunction.getResolution(), 1000);
	ASSERT_EQ(spectralFunction.getSize(), 3*1000);
	for(unsigned int n = 0; n < spectralFunction.getResolution(); n++){
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 1}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 1}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 2}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 2}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 1}, {0, 3}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 1}, {0, 3}}, n)),
			0
		);
	}
	EXPECT_DOUBLE_EQ(spectralFunction.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spectralFunction.getUpperBound(), 10);
}

//TBTKFeature Property.SpectralFunction.construction.3 2020-02-08
TEST(GreensFunction, Constructor3){
	IndexTree indexTree;
	indexTree.add({{0, 1}, {0, 1}});
	indexTree.add({{0, 2}, {0, 1}});
	indexTree.add({{0, 1}, {0, 3}});
	indexTree.generateLinearMap();
	complex<double> data[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data[n] = n;
	SpectralFunction spectralFunction(
		indexTree,
		-10,
		10,
		1000,
		data
	);
	ASSERT_EQ(spectralFunction.getBlockSize(), 1000);
	ASSERT_EQ(spectralFunction.getResolution(), 1000);
	ASSERT_EQ(spectralFunction.getSize(), 3*1000);
	for(unsigned int n = 0; n < spectralFunction.getResolution(); n++){
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 1}, {0, 1}}, n)),
			n
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 1}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 2}, {0, 1}}, n)),
			n + 1000
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 2}, {0, 1}}, n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(spectralFunction({{0, 1}, {0, 3}}, n)),
			n + 2000
		);
		EXPECT_DOUBLE_EQ(
			imag(spectralFunction({{0, 1}, {0, 3}}, n)),
			0
		);
	}
	EXPECT_DOUBLE_EQ(spectralFunction.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spectralFunction.getUpperBound(), 10);
}

};	//End of namespace Property
};	//End of namespace TBTK
