#include "TBTK/IndexException.h"
#include "TBTK/Property/WaveFunctions.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(WaveFunctions, Constructor0){
	WaveFunctions waveFunctions;
}

TEST(WaveFunctions, Constructor1){
	IndexTree indexTree;
	indexTree.add({1, 1});
	indexTree.add({1, 2});
	indexTree.add({2, 5});
	indexTree.generateLinearMap();

	//Continuous state indices.
	WaveFunctions waveFunctions0(indexTree, {3, 4, 5});
	const WaveFunctions &waveFunctions0C = waveFunctions0;
	for(unsigned int n = 3; n < 6; n++){
		//Non const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions0({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0({2, 5}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({2, 5}, n)), 0);

		//Const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({2, 5}, n)), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({2, 5}, n)), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions0({1, 3}, 3), IndexException);
	EXPECT_THROW(waveFunctions0C({1, 3}, 3), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0({1, 1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0C({1, 1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0({1, 1}, 6);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0C({1, 1}, 6);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Non continuous state indices.
	WaveFunctions waveFunctions1(indexTree, {1, 3, 7});
	const WaveFunctions &waveFunctions1C = waveFunctions1;
	int states[3] = {1, 3, 7};
	for(unsigned int n = 0; n < 3; n++){
		//Non const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({2, 5}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({2, 5}, states[n])), 0);

		//Const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({2, 5}, states[n])), 0);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({2, 5}, states[n])), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions1({1, 3}, 1), IndexException);
	EXPECT_THROW(waveFunctions1C({1, 3}, 1), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({1, 1}, 2);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1C({1, 1}, 2);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(WaveFunctions, Constructor2){
	IndexTree indexTree;
	indexTree.add({1, 1});
	indexTree.add({1, 2});
	indexTree.add({2, 5});
	indexTree.generateLinearMap();

	std::complex<double> data[3*3];
	for(unsigned int n = 0; n < 3*3; n++)
		data[n] = n;

	//Continuous state indices.
	WaveFunctions waveFunctions0(indexTree, {3, 4, 5}, data);
	const WaveFunctions &waveFunctions0C = waveFunctions0;
	for(unsigned int n = 3; n < 6; n++){
		//Non const versions.
		EXPECT_DOUBLE_EQ(real(waveFunctions0({1, 1}, n)), 0 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0({1, 2}, n)), 3 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0({2, 5}, n)), 6 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0({2, 5}, n)), 0);

		//Const versions.
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({1, 1}, n)), 0 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({1, 2}, n)), 3 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions0C({2, 5}, n)), 6 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions0C({2, 5}, n)), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions0({1, 3}, 3), IndexException);
	EXPECT_THROW(waveFunctions0C({1, 3}, 3), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0({1, 1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0C({1, 1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0({1, 1}, 6);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions0C({1, 1}, 6);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Non continuous state indices.
	WaveFunctions waveFunctions1(indexTree, {1, 3, 7}, data);
	const WaveFunctions &waveFunctions1C = waveFunctions1;
	int states[3] = {1, 3, 7};
	for(unsigned int n = 0; n < 3; n++){
		//Non const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 1}, states[n])), 0 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 2}, states[n])), 3 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({2, 5}, states[n])), 6 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({2, 5}, states[n])), 0);

		//Const version.
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({1, 1}, states[n])), 0 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({1, 2}, states[n])), 3 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1C({2, 5}, states[n])), 6 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1C({2, 5}, states[n])), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions1({1, 3}, 1), IndexException);
	EXPECT_THROW(waveFunctions1C({1, 3}, 1), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({1, 1}, 2);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1C({1, 1}, 2);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(WaveFunctions, SerializeToJSON){
	IndexTree indexTree;
	indexTree.add({1, 1});
	indexTree.add({1, 2});
	indexTree.add({2, 5});
	indexTree.generateLinearMap();

	std::complex<double> data[3*3];
	for(unsigned int n = 0; n < 3*3; n++)
		data[n] = n;

	//Continuous state indices.
	WaveFunctions waveFunctions0(indexTree, {3, 4, 5}, data);
	WaveFunctions waveFunctions1(
		waveFunctions0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	for(unsigned int n = 3; n < 6; n++){
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 1}, n)), 0 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 1}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({1, 2}, n)), 3 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({1, 2}, n)), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions1({2, 5}, n)), 6 + n - 3);
		EXPECT_DOUBLE_EQ(imag(waveFunctions1({2, 5}, n)), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions1({1, 3}, 3), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({1, 1}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({1, 1}, 6);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Non continuous state indices.
	WaveFunctions waveFunctions2(indexTree, {1, 3, 7}, data);
	WaveFunctions waveFunctions3(
		waveFunctions2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	int states[3] = {1, 3, 7};
	for(unsigned int n = 0; n < 3; n++){
		EXPECT_DOUBLE_EQ(real(waveFunctions3({1, 1}, states[n])), 0 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions3({1, 1}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions3({1, 2}, states[n])), 3 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions3({1, 2}, states[n])), 0);
		EXPECT_DOUBLE_EQ(real(waveFunctions3({2, 5}, states[n])), 6 + n);
		EXPECT_DOUBLE_EQ(imag(waveFunctions3({2, 5}, states[n])), 0);
	}
	//Throw exception for invalid Index.
	EXPECT_THROW(waveFunctions3({1, 3}, 1), IndexException);
	//Fail for invalid state index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions3({1, 1}, 2);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(WaveFunctions, getStates){
	IndexTree indexTree;
	indexTree.add({1, 1});
	indexTree.add({1, 2});
	indexTree.add({2, 5});
	indexTree.generateLinearMap();

	WaveFunctions waveFunctions0(indexTree, {3, 4, 5});
	const std::vector<unsigned int> &states0 = waveFunctions0.getStates();
	ASSERT_EQ(states0.size(), 3);
	EXPECT_EQ(states0[0], 3);
	EXPECT_EQ(states0[1], 4);
	EXPECT_EQ(states0[2], 5);

	WaveFunctions waveFunctions1(indexTree, {1, 3, 7});
	const std::vector<unsigned int> &states1 = waveFunctions1.getStates();
	ASSERT_EQ(states1.size(), 3);
	EXPECT_EQ(states1[0], 1);
	EXPECT_EQ(states1[1], 3);
	EXPECT_EQ(states1[2], 7);
}

TEST(WaveFunctions, operatorFunction){
	//Already tested through
	//WaveFunctions::Constructor0
	//WaveFunctions::Constructor1
	//WaveFunctions::SerializeToJSON
}

//TODO
//It is not clear that this function actually should be part of the
//WaveFunctions interface.
TEST(WaveFunctions, getMinAbs){
}

//TODO
//It is not clear that this function actually should be part of the
//WaveFunctions interface.
TEST(WaveFunctions, getMaxAbs){
}

//TODO
//It is not clear that this function actually should be part of the
//WaveFunctions interface.
TEST(WaveFunctions, getMinArg){
}

//TODO
//It is not clear that this function actually should be part of the
//WaveFunctions interface.
TEST(WaveFunctions, getMaxArg){
}

TEST(WaveFunctions, serialize){
	//Already tested through WaveFunctions::SerializeToJSON.
}

};	//End of namespace Property
};	//End of namespace TBTK
