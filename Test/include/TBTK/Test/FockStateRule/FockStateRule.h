#include "TBTK/FockStateRule/FockStateRule.h"
#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateRule{

class ImplementedFockStateRule : public FockStateRule{
public:
	ImplementedFockStateRule(
		FockStateRuleID fockStateRuleID
	) : FockStateRule(fockStateRuleID){}

	//Dummy implementation to make the class non-abstract.
	virtual FockStateRule* clone() const{
		return new ImplementedFockStateRule(getFockStateRuleID());
	}

	//Dummy implementation to make the class non-abstract.
	virtual WrapperRule createNewRule(
		const LadderOperator<BitRegister> &ladderOperator
	) const{
		return WrapperRule(
			ImplementedFockStateRule(getFockStateRuleID())
		);
	}

	//Dummy implementation to make the class non-abstract.
	virtual WrapperRule createNewRule(
		const LadderOperator<ExtensiveBitRegister> &ladderOperator
	) const{
		return ImplementedFockStateRule(getFockStateRuleID());
	}

	//Dummy implementation to make the class non-abstract.
	virtual bool isSatisfied(
		const FockSpace<BitRegister> &fockSpace,
		const FockState<BitRegister> &fockState
	) const{
		return true;
	}

	//Dummy implementation to make the class non-abstract.
	virtual bool isSatisfied(
		const FockSpace<ExtensiveBitRegister> &fockSpace,
		const FockState<ExtensiveBitRegister> &fockState
	) const{
		return true;
	}

	//Dummy implementation to make the class non-abstract.
	virtual bool operator==(const FockStateRule &rhs) const{
		return false;
	}

	//Dummy implementation to make the class non-abstract.
	virtual void print() const{
	}
};

//TBTKFeature FockStateRule.FockStateRule.getFockStateID.1 2019-11-05
TEST(FockStateRule, getFockStateID1){
	ImplementedFockStateRule implementedFockStateRule(
		FockStateRule::FockStateRule::FockStateRuleID::DifferenceRule
	);
	EXPECT_EQ(
		implementedFockStateRule.getFockStateRuleID(),
		FockStateRule::FockStateRule::FockStateRuleID::DifferenceRule
	);
}

//TBTKFeature FockStateRule.FockStateRule.operatorMultiplication.1 2019-11-05
TEST(FockStateRule, operatorMultiplication1){
	//This test is very convoluted because the structure of the
	//FockStateRules is strange. The FockStateRules should be rewritten to
	//not mix the WrapperRule class and FockStateRule class. In particular,
	//operator*() is defined in FockStateRule, but implemented in WrapperRule.
	Model model;
	model << HoppingAmplitude(1, {0}, {0});
	model.construct();
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize() + 1);

	LadderOperator<BitRegister> ladderOperator(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	SumRule sumRule0(
		{{0}},
		0
	);
	SumRule sumRule1(
		{{0}},
		1
	);

	WrapperRule wrapperRule = ladderOperator*sumRule0;
	EXPECT_TRUE(wrapperRule == sumRule1);
}

//TBTKFeature FockStateRule.FockStateRule.operatorMultiplication.2 2019-11-05
TEST(FockStateRule, operatorMultiplication2){
	//This test is very convoluted because the structure of the
	//FockStateRules is strange. The FockStateRules should be rewritten to
	//not mix the WrapperRule class and FockStateRule class. In particular,
	//operator*() is defined in FockStateRule, but implemented in WrapperRule.
	Model model;
	model << HoppingAmplitude(1, {0}, {0});
	model.construct();
	FockState<ExtensiveBitRegister> templateState(model.getBasisSize());
	ExtensiveBitRegister fermionMask(model.getBasisSize()+1);

	LadderOperator<ExtensiveBitRegister> ladderOperator(
		LadderOperator<ExtensiveBitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	SumRule sumRule0(
		{{0}},
		0
	);
	SumRule sumRule1(
		{{0}},
		1
	);

	WrapperRule wrapperRule = ladderOperator*sumRule0;
	EXPECT_TRUE(wrapperRule == sumRule1);
}

};
};
