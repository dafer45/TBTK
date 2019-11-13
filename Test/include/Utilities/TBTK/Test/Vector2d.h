#include "TBTK/Streams.h"
#include "TBTK/Vector2d.h"

#include "gtest/gtest.h"

namespace TBTK{

class Vector2dTest : public ::testing::Test{
protected:
	Vector2d u, v;

	void SetUp() override{
		u = Vector2d({2, 3});
		v = Vector2d({4, 7});
	}
};

//TBTKFeature Utilities.Vector2d.construction.1 2019-11-13
TEST(Vector2d, construction1){
	Vector2d v({10, 20});

	EXPECT_EQ(v.x, 10);
	EXPECT_EQ(v.y, 20);
}

//TBTKFeature Utilities.Vector2d.construction.2 2019-11-13
TEST(Vector2d, construction2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Vector2d v({10, 20, 30});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Vector2d.construction.3 2019-11-13
TEST(Vector2d, construction3){
	Vector2d v(std::vector<double>({10, 20}));

	EXPECT_EQ(v.x, 10);
	EXPECT_EQ(v.y, 20);
}

//TBTKFeature Utilities.Vector2d.construction.4 2019-11-13
TEST(Vector2d, construction4){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Vector2d v(std::vector<double>({10, 20, 30}));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Vector2d.operatorAddition.1 2019-11-13
TEST_F(Vector2dTest, operatorAddition1){
	Vector2d result = u + v;

	EXPECT_EQ(result.x, 6);
	EXPECT_EQ(result.y, 10);
}

//TBTKFeature Utilities.Vector2d.operatorSubtraction.1 2019-11-13
TEST_F(Vector2dTest, operatorSubtraction1){
	Vector2d result = u - v;

	EXPECT_EQ(result.x, -2);
	EXPECT_EQ(result.y, -4);
}

//TBTKFeature Utilities.Vector2d.operatorInversion.1 2019-11-13
TEST_F(Vector2dTest, operatorInversion1){
	Vector2d result = -u;

	EXPECT_EQ(result.x, -2);
	EXPECT_EQ(result.y, -3);
}

//TBTKFeature Utilities.Vector2d.operatorMultiplication.1 2019-11-13
TEST_F(Vector2dTest, operatorMultiplication1){
	Vector2d result = 2*u;

	EXPECT_EQ(result.x, 4);
	EXPECT_EQ(result.y, 6);
}

//TBTKFeature Utilities.Vector2d.operatorMultiplication.2 2019-11-13
TEST_F(Vector2dTest, operatorMultiplication2){
	Vector2d result = u*2;

	EXPECT_EQ(result.x, 4);
	EXPECT_EQ(result.y, 6);
}

//TBTKFeature Utilities.Vector2d.operatorDivision.1 2019-11-13
TEST_F(Vector2dTest, operatorDivision1){
	Vector2d result = u/2;

	EXPECT_EQ(result.x, 1);
	EXPECT_EQ(result.y, 1.5);
}

//TBTKFeature Utilities.Vector2d.unit.1 2019-11-13
TEST_F(Vector2dTest, unit1){
	Vector2d result = u.unit();

	EXPECT_FLOAT_EQ(result.x, 1/sqrt(3.25));
	EXPECT_FLOAT_EQ(result.y, 1.5/sqrt(3.25));
}

//TBTKFeature Utilities.Vector2d.parallel.1 2019-11-13
TEST_F(Vector2dTest, parallel1){
	Vector2d result = u.parallel(v);

	EXPECT_FLOAT_EQ(result.x, (29/65.)*4);
	EXPECT_FLOAT_EQ(result.y, (29/65.)*7);
}

//TBTKFeature Utilities.Vector2d.norm.1 2019-11-13
TEST_F(Vector2dTest, norm1){
	EXPECT_FLOAT_EQ(u.norm(), sqrt(13));
}

//TBTKFeature Utilities.Vector2d.dotProduct.1 2019-11-13
TEST_F(Vector2dTest, dotProduct1){
	EXPECT_FLOAT_EQ(Vector2d::dotProduct(u, v), 2*4 + 3*7);
}

//TBTKFeature Utilities.Vector2d.getStdVector.1 2019-11-13
TEST_F(Vector2dTest, getStdVector1){
	std::vector<double> w = u.getStdVector();

	EXPECT_EQ(w.size(), 2);
	EXPECT_FLOAT_EQ(w[0], 2);
	EXPECT_FLOAT_EQ(w[1], 3);
}

};
