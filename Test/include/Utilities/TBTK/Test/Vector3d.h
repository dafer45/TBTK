#include "TBTK/Streams.h"
#include "TBTK/Vector3d.h"

#include "gtest/gtest.h"

namespace TBTK{

class Vector3dTest : public ::testing::Test{
protected:
	Vector3d u, v;

	void SetUp() override{
		u = Vector3d({2, 3, 4});
		v = Vector3d({5, 7, 11});
	}
};

//TBTKFeature Utilities.Vector3d.construction.1 2019-11-13
TEST(Vector3d, construction1){
	Vector3d v({10, 20, 30});

	EXPECT_EQ(v.x, 10);
	EXPECT_EQ(v.y, 20);
	EXPECT_EQ(v.z, 30);
}

//TBTKFeature Utilities.Vector3d.construction.2 2019-11-13
TEST(Vector3d, construction2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Vector3d v({10, 20});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Vector3d.construction.3 2019-11-13
TEST(Vector3d, construction3){
	Vector3d v(std::vector<double>({10, 20, 30}));

	EXPECT_EQ(v.x, 10);
	EXPECT_EQ(v.y, 20);
	EXPECT_EQ(v.z, 30);
}

//TBTKFeature Utilities.Vector3d.construction.4 2019-11-13
TEST(Vector3d, construction4){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Vector3d v(std::vector<double>({10, 20}));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Vector3d.operatorAddition.1 2019-11-13
TEST_F(Vector3dTest, operatorAddition1){
	Vector3d result = u + v;

	EXPECT_EQ(result.x, 2 + 5);
	EXPECT_EQ(result.y, 3 + 7);
	EXPECT_EQ(result.z, 4 + 11);
}

//TBTKFeature Utilities.Vector3d.operatorSubtraction.1 2019-11-13
TEST_F(Vector3dTest, operatorSubtraction1){
	Vector3d result = u - v;

	EXPECT_EQ(result.x, 2 - 5);
	EXPECT_EQ(result.y, 3 - 7);
	EXPECT_EQ(result.z, 4 - 11);
}

//TBTKFeature Utilities.Vector3d.operatorInversion.1 2019-11-13
TEST_F(Vector3dTest, operatorInversion1){
	Vector3d result = -u;

	EXPECT_EQ(result.x, -2);
	EXPECT_EQ(result.y, -3);
	EXPECT_EQ(result.z, -4);
}

//TBTKFeature Utilities.Vector3d.operatorMultiplication.1 2019-11-13
TEST_F(Vector3dTest, operatorMultiplication1){
	Vector3d result = u*v;

	EXPECT_EQ(result.x, 3*11 - 4*7);
	EXPECT_EQ(result.y, 4*5 - 2*11);
	EXPECT_EQ(result.z, 2*7 - 3*5);
}

//TBTKFeature Utilities.Vector3d.operatorMultiplication.2 2019-11-13
TEST_F(Vector3dTest, operatorMultiplication2){
	Vector3d result = 2*u;

	EXPECT_EQ(result.x, 4);
	EXPECT_EQ(result.y, 6);
	EXPECT_EQ(result.z, 8);
}

//TBTKFeature Utilities.Vector3d.operatorMultiplication.3 2019-11-13
TEST_F(Vector3dTest, operatorMultiplication3){
	Vector3d result = u*2;

	EXPECT_EQ(result.x, 4);
	EXPECT_EQ(result.y, 6);
	EXPECT_EQ(result.z, 8);
}

//TBTKFeature Utilities.Vector3d.operatorDivision.1 2019-11-13
TEST_F(Vector3dTest, operatorDivision1){
	Vector3d result = u/2;

	EXPECT_EQ(result.x, 1);
	EXPECT_EQ(result.y, 1.5);
	EXPECT_EQ(result.z, 2);
}

//TBTKFeature Utilities.Vector3d.unit.1 2019-11-13
TEST_F(Vector3dTest, unit1){
	Vector3d result = u.unit();

	EXPECT_FLOAT_EQ(result.x, 2/sqrt(2*2 + 3*3 + 4*4));
	EXPECT_FLOAT_EQ(result.y, 3/sqrt(2*2 + 3*3 + 4*4));
	EXPECT_FLOAT_EQ(result.z, 4/sqrt(2*2 + 3*3 + 4*4));
}

//TBTKFeature Utilities.Vector3d.perpendicular.1 2019-11-13
TEST_F(Vector3dTest, perpendicular1){
	Vector3d result = u.perpendicular(v);

	EXPECT_FLOAT_EQ(result.x, 2 - ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*5);
	EXPECT_FLOAT_EQ(result.y, 3 - ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*7);
	EXPECT_FLOAT_EQ(result.z, 4 - ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*11);
}

//TBTKFeature Utilities.Vector3d.parallel.1 2019-11-13
TEST_F(Vector3dTest, parallel1){
	Vector3d result = u.parallel(v);

	EXPECT_FLOAT_EQ(result.x, ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*5);
	EXPECT_FLOAT_EQ(result.y, ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*7);
	EXPECT_FLOAT_EQ(result.z, ((2*5 + 3*7 + 4*11)/(5*5. + 7*7 + 11*11))*11);
}

//TBTKFeature Utilities.Vector3d.norm.1 2019-11-13
TEST_F(Vector3dTest, norm1){
	EXPECT_FLOAT_EQ(u.norm(), sqrt(2*2 + 3*3 + 4*4));
}

//TBTKFeature Utilities.Vector3d.dotProduct.1 2019-11-13
TEST_F(Vector3dTest, dotProduct1){
	EXPECT_FLOAT_EQ(Vector3d::dotProduct(u, v), 2*5 + 3*7 + 4*11);
}

//TBTKFeature Utilities.Vector3d.getStdVector.1 2019-11-13
TEST_F(Vector3dTest, getStdVector1){
	std::vector<double> w = u.getStdVector();

	EXPECT_EQ(w.size(), 3);
	EXPECT_FLOAT_EQ(w[0], 2);
	EXPECT_FLOAT_EQ(w[1], 3);
	EXPECT_FLOAT_EQ(w[2], 4);
}

};
