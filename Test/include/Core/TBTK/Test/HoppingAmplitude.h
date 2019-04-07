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

class AmplitudeCallback : public HoppingAmplitude::AmplitudeCallback{
	virtual std::complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		if(to.equals({1, 2}))
			return std::complex<double>(3, 4);
		else if(from.equals({1, 2}))
			return std::complex<double>(3, -4);
		else
			return -1;
	}
} amplitudeCallback;

TEST(HoppingAmplitude, ConstructorAmplitudeCallback){
	std::string errorMessage = "Callback constructor failed.";

	HoppingAmplitude hoppingAmplitude(amplitudeCallback, {1, 2}, {3, 4, 5});
	EXPECT_EQ(hoppingAmplitude.getAmplitude(), std::complex<double>(3, 4)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getToIndex().equals({1, 2})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude.getFromIndex().equals({3, 4, 5})) << errorMessage;
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
		hoppingAmplitude0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(hoppingAmplitude0.getAmplitude(), hoppingAmplitude1.getAmplitude()) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude0.getToIndex().equals(hoppingAmplitude1.getToIndex())) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude0.getFromIndex().equals(hoppingAmplitude1.getFromIndex())) << errorMessage;
}

TEST(HoppingAmplitude, getHermitianConjugate){
	std::string errorMessage = "getHermitianConjugate() failed.";

	HoppingAmplitude hoppingAmplitude0(std::complex<double>(1, 2), {1, 2, 3}, {4, 5});
	HoppingAmplitude hoppingAmplitude1 = hoppingAmplitude0.getHermitianConjugate();
	EXPECT_EQ(hoppingAmplitude1.getAmplitude(), std::complex<double>(1, -2)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude1.getToIndex().equals({4, 5})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude1.getFromIndex().equals({1, 2, 3})) << errorMessage;

	HoppingAmplitude hoppingAmplitude2(amplitudeCallback, {1, 2}, {3, 4, 5});
	HoppingAmplitude hoppingAmplitude3 = hoppingAmplitude2.getHermitianConjugate();
	EXPECT_EQ(hoppingAmplitude3.getAmplitude(), std::complex<double>(3, -4)) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude3.getToIndex().equals({3, 4, 5})) << errorMessage;
	EXPECT_TRUE(hoppingAmplitude3.getFromIndex().equals({1, 2})) << errorMessage;
}

TEST(HoppingAmplitude, getAmplitude){
	//Extensively tested through other tests.
}

TEST(HoppingAmplitude, operatorAddition){
	std::string errorMessage = "operator+() failed.";

	HoppingAmplitude hoppingAmplitude(std::complex<double>(1, 2), {1, 2, 3}, {4, 5});
	std::tuple<HoppingAmplitude, HoppingAmplitude> pair = hoppingAmplitude + HC;
	EXPECT_EQ(std::get<0>(pair).getAmplitude(), std::complex<double>(1, 2)) << errorMessage;
	EXPECT_EQ(std::get<1>(pair).getAmplitude(), std::complex<double>(1, -2)) << errorMessage;
	EXPECT_TRUE(std::get<0>(pair).getToIndex().equals({1, 2, 3})) << errorMessage;
	EXPECT_TRUE(std::get<1>(pair).getToIndex().equals({4, 5})) << errorMessage;
	EXPECT_TRUE(std::get<0>(pair).getFromIndex().equals({4, 5})) << errorMessage;
	EXPECT_TRUE(std::get<1>(pair).getFromIndex().equals({1, 2, 3})) << errorMessage;
}

TEST(HoppingAmplitude, getToIndex){
	//Extensively tested through other tests.
}

TEST(HoppingAmplitude, getFromIndex){
	//Extensively tested through other tests.
}

TEST(HoppingAmplitude, getIsCallbackDependent){
	HoppingAmplitude hoppingAmplitude0(1, {0}, {0});
	EXPECT_FALSE(hoppingAmplitude0.getIsCallbackDependent());

	HoppingAmplitude hoppingAmplitude1(amplitudeCallback, {0}, {0});
	EXPECT_TRUE(hoppingAmplitude1.getIsCallbackDependent());
}

TEST(HoppingAmplitude, getAmplitudeCallback){
	HoppingAmplitude hoppingAmplitude0(1, {0}, {0});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			hoppingAmplitude0.getAmplitudeCallback();
		},
		::testing::ExitedWithCode(1),
		""
	);

	HoppingAmplitude hoppingAmplitude1(amplitudeCallback, {0}, {0});
	EXPECT_EQ(&hoppingAmplitude1.getAmplitudeCallback(), &amplitudeCallback);
}

TEST(HoppingAmplitude, toString){
	//Not tested due to insuficient control of number formating.
}

TEST(HoppingAmplitude, getSizeInBytes){
	EXPECT_TRUE(HoppingAmplitude(std::complex<double>(1, 2), {1, 2, 3}, {4, 5}).getSizeInBytes() > 0) << "getSizeInBytes() failed.";
}

};
