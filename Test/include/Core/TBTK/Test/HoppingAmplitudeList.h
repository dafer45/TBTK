#include "TBTK/HoppingAmplitudeList.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(HoppingAmplitudeList, Constructor){
	HoppingAmplitudeList hoppingAmplitudeList;
	EXPECT_EQ(hoppingAmplitudeList.getSize(), 0);
}

TEST(HoppingAmplitudeList, SerializeToJSON){
	HoppingAmplitudeList hoppingAmplitudeList0;
	hoppingAmplitudeList0.pushBack(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeList0.pushBack(HoppingAmplitude(1, {2, 1}, {3, 4}));
	hoppingAmplitudeList0.pushBack(HoppingAmplitude(1, {2, 1}, {4, 3}));

	HoppingAmplitudeList hoppingAmplitudeList1(
		hoppingAmplitudeList0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(hoppingAmplitudeList1.getSize(), 3);
	EXPECT_TRUE(hoppingAmplitudeList1[0].getToIndex().equals({1, 2}));
	EXPECT_TRUE(hoppingAmplitudeList1[0].getFromIndex().equals({3, 4}));
	EXPECT_TRUE(hoppingAmplitudeList1[1].getToIndex().equals({2, 1}));
	EXPECT_TRUE(hoppingAmplitudeList1[1].getFromIndex().equals({3, 4}));
	EXPECT_TRUE(hoppingAmplitudeList1[2].getToIndex().equals({2, 1}));
	EXPECT_TRUE(hoppingAmplitudeList1[2].getFromIndex().equals({4, 3}));
};

TEST(HoppingAmplitudeList, getSize){
	HoppingAmplitudeList hoppingAmplitudeList;
	EXPECT_EQ(hoppingAmplitudeList.getSize(), 0);

	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {2, 1}, {3, 4}));
	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {2, 1}, {4, 3}));
	EXPECT_EQ(hoppingAmplitudeList.getSize(), 3);
}

TEST(HoppingAmplitudeList, operatorSubscript){
	HoppingAmplitudeList hoppingAmplitudeList;

	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {1, 2}, {3, 4}));
	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {2, 1}, {3, 4}));
	hoppingAmplitudeList.pushBack(HoppingAmplitude(1, {2, 1}, {4, 3}));

	EXPECT_TRUE(hoppingAmplitudeList[0].getToIndex().equals({1, 2}));
	EXPECT_TRUE(hoppingAmplitudeList[0].getFromIndex().equals({3, 4}));
	EXPECT_TRUE(hoppingAmplitudeList[1].getToIndex().equals({2, 1}));
	EXPECT_TRUE(hoppingAmplitudeList[1].getFromIndex().equals({3, 4}));
	EXPECT_TRUE(hoppingAmplitudeList[2].getToIndex().equals({2, 1}));
	EXPECT_TRUE(hoppingAmplitudeList[2].getFromIndex().equals({4, 3}));
}

TEST(HoppingAmplitudeList, serialize){
	//Already tested through SerializeToJSON.
}

TEST(HoppingAmplitudeList, getSizeInBytes){
	HoppingAmplitudeList hoppingAmplitudeList;
	EXPECT_TRUE(hoppingAmplitudeList.getSizeInBytes() > 0);
}

}; //End of namespace TBTK
