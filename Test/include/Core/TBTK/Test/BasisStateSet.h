#include "TBTK/BasisStateSet.h"

#include "TBTK/BasicState.h"

#include "gtest/gtest.h"

namespace TBTK{

class TestState : public AbstractState{
public:
	TestState(
		const Index &index
	) : AbstractState((AbstractState::StateID)-1)
	{
		setIndex(index);
	}

	virtual AbstractState* clone() const{
		return new TestState(getIndex());
	}

	virtual std::complex<double> getOverlap(
		const AbstractState &ket
	) const{
		return 0;
	}

	virtual std::complex<double> getMatrixElement(
		const AbstractState &ket,
		const AbstractOperator &o
	) const{
		return 0;
	}
};

TEST(BasisStateSet, Constructor){
	//Not testable on its own.
}

TEST(BasisStateSet, SerializeToJSON){
	//TODO: Not yet implemented.
}

TEST(BasisStateSet, add){
	BasisStateSet basisStateSet;
	basisStateSet.add(TestState({1, 2}));
	basisStateSet.add(TestState({2, 1}));
	basisStateSet.add(TestState({3, 2}));

	//Fail to add basis state with conflicting Index structure (shorter).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			basisStateSet.add(TestState({1}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add basis state with conflicting Index structure (longer).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			basisStateSet.add(
				TestState({1, 2, 3})
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(BasisStateSet, get0){
	BasisStateSet basisStateSet;
	basisStateSet.add(TestState({1, 2}));
	basisStateSet.add(TestState({2, 1}));
	basisStateSet.add(TestState({3, 2}));

	//Check that all added basis state are possible to get.
	EXPECT_TRUE(basisStateSet.get({1, 2}).getIndex().equals({1, 2}));
	EXPECT_TRUE(basisStateSet.get({2, 1}).getIndex().equals({2, 1}));
	EXPECT_TRUE(basisStateSet.get({3, 2}).getIndex().equals({3, 2}));
	EXPECT_EQ(basisStateSet.get({1, 2}).getStateID(), -1);
	EXPECT_EQ(basisStateSet.get({2, 1}).getStateID(), -1);
	EXPECT_EQ(basisStateSet.get({3, 2}).getStateID(), -1);

	//Throw ElementNotFOundException for non-existing elements.
	EXPECT_THROW(basisStateSet.get({1, 2, 5}), ElementNotFoundException);

	//Fail for invalid Index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			basisStateSet.get({1, -1, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(BasisStateSet, get1){
	BasisStateSet basisStateSet;
	basisStateSet.add(TestState({1, 2}));
	basisStateSet.add(TestState({2, 1}));
	basisStateSet.add(TestState({3, 2}));

	//Check that all added basis state are possible to get.
	EXPECT_EQ(
		&((const BasisStateSet&)basisStateSet).get({1, 2}),
		&basisStateSet.get({1, 2})
	);
}

TEST(BasisStateSet, getSizeInBytes){
	//TODO: Not yet implemented.
}

TEST(BasisStateSet, Iterator){
	BasisStateSet basisStateSet;
	basisStateSet.add(TestState({1, 2}));
	basisStateSet.add(TestState({2, 1}));
	basisStateSet.add(TestState({3, 2}));

	BasisStateSet::Iterator iterator = basisStateSet.begin();
	EXPECT_FALSE(iterator == basisStateSet.end());
	EXPECT_TRUE(iterator != basisStateSet.end());
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_FALSE(iterator == basisStateSet.end());
	EXPECT_TRUE(iterator != basisStateSet.end());
	EXPECT_TRUE((*iterator).getIndex().equals({2, 1}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_FALSE(iterator == basisStateSet.end());
	EXPECT_TRUE(iterator != basisStateSet.end());
	EXPECT_TRUE((*iterator).getIndex().equals({3, 2}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_TRUE(iterator == basisStateSet.end());
	EXPECT_FALSE(iterator != basisStateSet.end());
}

TEST(BasisStateSet, ConstIterator){
	BasisStateSet basisStateSet;
	basisStateSet.add(TestState({1, 2}));
	basisStateSet.add(TestState({2, 1}));
	basisStateSet.add(TestState({3, 2}));

	BasisStateSet::ConstIterator iterator = basisStateSet.cbegin();
	EXPECT_FALSE(iterator == basisStateSet.cend());
	EXPECT_TRUE(iterator != basisStateSet.cend());
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_FALSE(iterator == basisStateSet.cend());
	EXPECT_TRUE(iterator != basisStateSet.cend());
	EXPECT_TRUE((*iterator).getIndex().equals({2, 1}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_FALSE(iterator == basisStateSet.cend());
	EXPECT_TRUE(iterator != basisStateSet.cend());
	EXPECT_TRUE((*iterator).getIndex().equals({3, 2}));
	EXPECT_EQ((*iterator).getStateID(), -1);

	++iterator;
	EXPECT_TRUE(iterator == basisStateSet.cend());
	EXPECT_FALSE(iterator != basisStateSet.cend());

	//Verify that begin() and end() return ConstIterator for const
	//SourceAmplitudeSet.
	iterator = const_cast<const BasisStateSet&>(basisStateSet).begin();
	iterator = const_cast<const BasisStateSet&>(basisStateSet).end();
}

};
