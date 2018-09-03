#include "TBTK/Geometry.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Geometry, Constructor){
	//Not testable on its own.
}

TEST(Geometry, serializeToJSON){
	Geometry geometry0;
	geometry0.setCoordinate({0, 1, 2, 3}, {0, 1, 2});
	geometry0.setCoordinate({0, 1, 3, 3}, {0, 1, 3});

	Geometry geometry1(
		geometry0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(geometry1.getDimensions(), 3);

	//Retrieve the first coordinate.
	const std::vector<double> &coordinate0
		= geometry1.getCoordinate({0, 1, 2, 3});
	EXPECT_EQ(coordinate0.size(), 3);
	EXPECT_EQ(coordinate0[0], 0);
	EXPECT_EQ(coordinate0[1], 1);
	EXPECT_EQ(coordinate0[2], 2);

	//Retrieve the second coordinate.
	const std::vector<double> &coordinate1
		= geometry1.getCoordinate({0, 1, 3, 3});
	EXPECT_EQ(coordinate1.size(), 3);
	EXPECT_EQ(coordinate1[0], 0);
	EXPECT_EQ(coordinate1[1], 1);
	EXPECT_EQ(coordinate1[2], 3);
}

TEST(Geometry, Destructor){
	//Not testable on its own.
}

TEST(Geometry, setCoordinate){
	Geometry geometry0;

	//Set three-dimensional coordinates.
	geometry0.setCoordinate({0, 1, 2, 3}, {0, 1, 2});
	geometry0.setCoordinate({0, 1, 3, 3}, {0, 1, 3});

	//Fail to set two-dimensional coordinate if a three-dimensional
	//coordinate has already ben set.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			geometry0.setCoordinate({0, 1, 4, 3}, {0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Set two-dimensional coordinate.
	Geometry geometry1;
	geometry1.setCoordinate({0, 1, 4, 3}, {0, 1});
}

TEST(Geometry, getCoordinate){
	Geometry geometry;
	geometry.setCoordinate({0, 1, 2, 3}, {0, 1, 2});
	geometry.setCoordinate({0, 1, 3, 3}, {0, 1, 3});

	//Retrieve the first coordinate.
	const std::vector<double> &coordinate0
		= geometry.getCoordinate({0, 1, 2, 3});
	EXPECT_EQ(coordinate0.size(), 3);
	EXPECT_EQ(coordinate0[0], 0);
	EXPECT_EQ(coordinate0[1], 1);
	EXPECT_EQ(coordinate0[2], 2);

	//Retrieve the second coordinate.
	const std::vector<double> &coordinate1
		= geometry.getCoordinate({0, 1, 3, 3});
	EXPECT_EQ(coordinate1.size(), 3);
	EXPECT_EQ(coordinate1[0], 0);
	EXPECT_EQ(coordinate1[1], 1);
	EXPECT_EQ(coordinate1[2], 3);
}

TEST(Geometry, getDimensions){
	Geometry geometry;

	//Test the initial value.
	EXPECT_EQ(geometry.getDimensions(), -1);

	//Test after having set a coordinate.
	geometry.setCoordinate({0, 1, 2, 3}, {0, 1, 2});
	EXPECT_EQ(geometry.getDimensions(), 3);
}

TEST(Geometry, translate){
	Geometry geometry;
	geometry.setCoordinate({0, 1, 2, 3}, {0, 1, 2});
	geometry.setCoordinate({0, 1, 3, 3}, {0, 1, 3});
	geometry.translate({1, 2, 3});

	//Retrieve the first coordinate.
	const std::vector<double> &coordinate0
		= geometry.getCoordinate({0, 1, 2, 3});
	EXPECT_EQ(coordinate0.size(), 3);
	EXPECT_EQ(coordinate0[0], 1);
	EXPECT_EQ(coordinate0[1], 3);
	EXPECT_EQ(coordinate0[2], 5);

	//Retrieve the second coordinate.
	const std::vector<double> &coordinate1
		= geometry.getCoordinate({0, 1, 3, 3});
	EXPECT_EQ(coordinate1.size(), 3);
	EXPECT_EQ(coordinate1[0], 1);
	EXPECT_EQ(coordinate1[1], 3);
	EXPECT_EQ(coordinate1[2], 6);

	//Fail to translate with a vector of the wrong dimension,
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			geometry.translate({0, 1});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

}; //End of namespace TBTK
