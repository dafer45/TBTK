#include "TBTK/Streams.h"
#include "TBTK/VectorNd.h"

#include "gtest/gtest.h"

namespace TBTK{

class VectorNdTest : public ::testing::Test{
protected:
	VectorNd u, v, w;

	void SetUp() override{
		u = VectorNd({2, 3});
		v = VectorNd({4, 7});
		w = VectorNd({1, 2, 3});
	}
};

//TBTKFeature Utilities.VectorNd.construction.1 2019-11-14
TEST(VectorNd, construction1){
	VectorNd u(2);
	VectorNd v(3);

	EXPECT_EQ(u.getSize(), 2);
	EXPECT_EQ(v.getSize(), 3);
}

//TBTKFeature Utilities.VectorNd.construction.2 2019-11-14
TEST(VectorNd, construction2){
	VectorNd v({10, 20});

	EXPECT_EQ(v.getSize(), 2);
	EXPECT_EQ(v[0], 10);
	EXPECT_EQ(v[1], 20);
}

//TBTKFeature Utilities.VectorNd.construction.3 2019-11-14
TEST(VectorNd, construction3){
	VectorNd v(std::vector<double>({10, 20}));

	EXPECT_EQ(v.getSize(), 2);
	EXPECT_EQ(v[0], 10);
	EXPECT_EQ(v[1], 20);
}

//TBTKFeature Utilities.VectorNd.operatorArraySubscript.1 2019-11-14
TEST_F(VectorNdTest, operatorArraySubscript1){
	EXPECT_EQ(u[0], 2);
	EXPECT_EQ(u[1], 3);
	EXPECT_EQ(v[0], 4);
	EXPECT_EQ(v[1], 7);
	EXPECT_EQ(w[0], 1);
	EXPECT_EQ(w[1], 2);
	EXPECT_EQ(w[2], 3);

	w[1] = 10;
	EXPECT_EQ(w[1], 10);
}

//TBTKFeature Utilities.VectorNd.operatorArraySubscript.2 2019-11-14
TEST_F(VectorNdTest, operatorArraySubscript2){
	const VectorNd &U = u;
	const VectorNd &V = v;
	const VectorNd &W = w;
	EXPECT_EQ(U[0], 2);
	EXPECT_EQ(U[1], 3);
	EXPECT_EQ(V[0], 4);
	EXPECT_EQ(V[1], 7);
	EXPECT_EQ(W[0], 1);
	EXPECT_EQ(W[1], 2);
	EXPECT_EQ(W[2], 3);
}

//TBTKFeature Utilities.VectorNd.operatorAddition.1 2019-11-14
TEST_F(VectorNdTest, operatorAddition1){
	VectorNd result = u + v;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], 6);
	EXPECT_EQ(result[1], 10);
}

//TBTKFeature Utilities.VectorNd.operatorAddition.2 2019-11-14
TEST_F(VectorNdTest, operatorAddition2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			VectorNd result = u + w;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.VectorNd.operatorSubtraction.1 2019-11-14
TEST_F(VectorNdTest, operatorSubtraction1){
	VectorNd result = u - v;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], -2);
	EXPECT_EQ(result[1], -4);
}

//TBTKFeature Utilities.VectorNd.operatorSubtraction.2 2019-11-14
TEST_F(VectorNdTest, operatorSubtraction2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			VectorNd result = u - w;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.VectorNd.operatorInversion.1 2019-11-14
TEST_F(VectorNdTest, operatorInversion1){
	VectorNd result = -u;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], -2);
	EXPECT_EQ(result[1], -3);
}

//TBTKFeature Utilities.VectorNd.operatorMultiplication.1 2019-11-14
TEST_F(VectorNdTest, operatorMultiplication1){
	VectorNd result = 2*u;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], 4);
	EXPECT_EQ(result[1], 6);
}

//TBTKFeature Utilities.VectorNd.operatorMultiplication.2 2019-11-14
TEST_F(VectorNdTest, operatorMultiplication2){
	VectorNd result = u*2;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], 4);
	EXPECT_EQ(result[1], 6);
}

//TBTKFeature Utilities.VectorNd.operatorDivision.1 2019-11-14
TEST_F(VectorNdTest, operatorDivision1){
	VectorNd result = u/2;

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_EQ(result[0], 1);
	EXPECT_EQ(result[1], 1.5);
}

//TBTKFeature Utilities.VectorNd.unit.1 2019-11-14
TEST_F(VectorNdTest, unit1){
	VectorNd result = u.unit();

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_FLOAT_EQ(result[0], 1/sqrt(3.25));
	EXPECT_FLOAT_EQ(result[1], 1.5/sqrt(3.25));
}

//TBTKFeature Utilities.VectorNd.perpendicular.1 2019-11-14
/*TEST_F(VectorNdTest, perpendicular1){
	VectorNd result = u.perpendicular(v);

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_FLOAT_EQ(result[0], 2 - (29/65.)*4);
	EXPECT_FLOAT_EQ(result[1], 3 - (29/65.)*7);
}

//TBTKFeature Utilities.VectorNd.perpendicular.2 2019-11-14
TEST_F(VectorNdTest, perpendicular2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			VectorNd result = u.perpendicular(w);
		},
		::testing::ExitedWithCode(1),
		""
	);
}*/

//TBTKFeature Utilities.VectorNd.parallel.1 2019-11-14
TEST_F(VectorNdTest, parallel1){
	VectorNd result = u.parallel(v);

	EXPECT_EQ(result.getSize(), 2);
	EXPECT_FLOAT_EQ(result[0], (29/65.)*4);
	EXPECT_FLOAT_EQ(result[1], (29/65.)*7);
}

//TBTKFeature Utilities.VectorNd.parallel.2 2019-11-14
TEST_F(VectorNdTest, parallel2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			VectorNd result = u.parallel(w);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.VectorNd.norm.1 2019-11-14
TEST_F(VectorNdTest, norm1){
	EXPECT_FLOAT_EQ(u.norm(), sqrt(13));
}

//TBTKFeature Utilities.VectorNd.dotProduct.1 2019-11-14
TEST_F(VectorNdTest, dotProduct1){
	EXPECT_FLOAT_EQ(VectorNd::dotProduct(u, v), 2*4 + 3*7);
}

//TBTKFeature Utilities.VectorNd.dotProduct.2 2019-11-14
TEST_F(VectorNdTest, dotProduct2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			VectorNd::dotProduct(u, w);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.VectorNd.getStdVector.1 2019-11-14
TEST_F(VectorNdTest, getStdVector1){
	std::vector<double> U = u.getStdVector();

	EXPECT_EQ(U.size(), 2);
	EXPECT_FLOAT_EQ(U[0], 2);
	EXPECT_FLOAT_EQ(U[1], 3);
}

//TBTKFeature Utilities.VectorNd.getSize.1 2019-11-14
TEST_F(VectorNdTest, getSize){
	EXPECT_EQ(u.getSize(), 2);
	EXPECT_EQ(v.getSize(), 2);
	EXPECT_EQ(w.getSize(), 3);
}

};
