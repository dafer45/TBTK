#include "TBTK/HoppingAmplitude.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(HoppingAmplitude, ConstructorAmplitude){
	std::string errorMessage = "Constructor failed.";

	HoppingAmplitude hoppingAmplitude(std::complex<double>(1, 2), {1, 2, 3}, {4, 5});
	EXPECT_EQ(hoppingAmplitude.getAmplitude(), std::complex<double>(1, 2)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({1, 2, 3})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({4, 5})) << errorMessage;
}

std::complex<double> amplitudeCallback(const Index &to, const Index &from){
	return std::complex<double>(3, 4);
}

TEST(HoppingAmplitude, ConstructorAmplitudeCallback){
	std::string errorMessage = "Callback constructor failed.";

	HoppingAmplitude hoppingAmplitude(amplitudeCallback, {1, 2, 3}, {4, 5});
	EXPECT_EQ(hoppingAmplitude.getAmplitude(), std::complex<double>(3, 4)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({1, 2, 3})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({4, 5})) << errorMessage;
}

TEST(HoppingAmplitude, CopyConstructor){
	std::string errorMessage = "Copy constructor failed.";

	HoppingAmplitude hoppingAmplitude0(std::complex<double>(1, 2), {1, 2, 3}, {4, 5});
	HoppingAmplitude hoppingAmplitude1(amplitudeCallback, {1, 2}, {3, 4, 5});

	HoppingAmplitude hoppingAmplitude2 = hoppingAmplitude0;
	EXPECT_EQ(hoppingAmplitude2.getAmplitude(), std::complex<double>(1, 2)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude2.getToIndex().equals({1, 2, 3})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude2.getFromIndex().equals({4, 5})) << errorMessage;

	HoppingAmplitude hoppingAmplitude3 = hoppingAmplitude1;
	EXPECT_EQ(hoppingAmplitude3.getAmplitude(), std::complex<double>(3, 4)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude3.getToIndex().equals({1, 2})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude3.getFromIndex().equals({3, 4, 5})) << errorMessage;
}

TEST(HoppingAmplitude, SerializeToJSON){
	std::string errorMessage = "JSON serialization failed.";

	HoppingAmplitude hoppingAmplitude0(std::complex<double>(1, 2), {1, 2, 3}, {4, 5});
	HoppingAmplitude hoppingAmplitude1(
		hoppingAmplitude0.serialize(Serializeable::Mode::JSON),
		Serializeable::Mode::JSON
	);
	EXPECT_EQ(hoppingAmplitude0.getAmplitude(), hoppingAmplitude1.getAmplitude()) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude0.getToIndex().equals(hoppingAmplitude1.getToIndex())) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude0.getFromIndex().equals(hoppingAmplitude1.getFromIndex())) << errorMessage;
}

TEST(HoppingAmplitude, getHermitianConjugate){
	std::string errorMessage = "getHermitianConjugate() failed.";

//	HoppingAmplitude hoppingAmplit
}

};
