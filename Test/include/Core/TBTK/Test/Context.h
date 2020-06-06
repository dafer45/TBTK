#include "TBTK/Context.h"
#include "TBTK/PersistentObject.h"
#include "TBTK/DynamicTypeInformation.h"

#include "gtest/gtest.h"

namespace TBTK{

class BaseA : virtual public PersistentObject{
	TBTK_DYNAMIC_TYPE_INFORMATION(BaseA)
public:
	std::string serialize(Mode mode) const{
		return "";
	}
private:
	int a;
};

class BaseB : virtual public PersistentObject{
	TBTK_DYNAMIC_TYPE_INFORMATION(BaseB)
public:
	std::string serialize(Mode mode) const{
		return "";
	}
private:
	double b;
};

class Derived : public BaseA, public BaseB{
	TBTK_DYNAMIC_TYPE_INFORMATION(Derived)
public:
	std::string serialize(Mode mode) const{
		return "";
	}
private:
	char d;
};

class NonPersistent{
private:
	unsigned int n;
};

class MissingDynamicTypeInformation : virtual public PersistentObject{
public:
	std::string serialize(Mode mode) const{
		return "";
	}
private:
	float m;
};

DynamicTypeInformation BaseA::dynamicTypeInformation("BaseA", {});
DynamicTypeInformation BaseB::dynamicTypeInformation("BaseB", {});
DynamicTypeInformation Derived::dynamicTypeInformation(
	"Derived",
	{&BaseA::dynamicTypeInformation, &BaseB::dynamicTypeInformation}
);

bool isInitialized = false;

class ContextTest : public ::testing::Test{
protected:
	void SetUp() override{
		if(!isInitialized){
			Context &context = Context::getContext();
			context.create<BaseA>("MyBaseA");
			context.create<BaseB>("MyBaseB");
			context.create<Derived>("MyDerived");
			isInitialized = true;
		}
	}
};

//TBTKFeature Core.Context.getContext.0 2020-06-04
TEST_F(ContextTest, getContext0){
	Context &context0 = Context::getContext();
	Context &context1 = Context::getContext();
	EXPECT_EQ(&context0, &context1);
}

//TBTKFeature Core.Context.create.0 2020-06-06
TEST_F(ContextTest, create0){
	Context &context = Context::getContext();
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			context.create<BaseA>("MyBaseA");
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Core.Context.create.1 2020-06-06
TEST_F(ContextTest, create1){
	Context &context = Context::getContext();
	BaseA &baseA = context.create<BaseA>("TempBaseA");
	EXPECT_EQ(baseA.getDynamicTypeInformation().getName(), "BaseA");
	context.erase("TempBaseA");
}

//TBTKFeature Core.Context.create.2 2020-06-06
TEST_F(ContextTest, create2){
	Context &context = Context::getContext();
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			context.create<MissingDynamicTypeInformation>(
				"TempBaseA"
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Core.Context.erase.0 2020-06-06
TEST_F(ContextTest, erase0){
	Context &context = Context::getContext();
	context.create<Derived>("ToBeErased");
	Derived &derived = context.get<Derived>("ToBeErased");
	EXPECT_EQ(derived.getDynamicTypeInformation().getName(), "Derived");
	context.erase("ToBeErased");
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			context.get<Derived>("ToBeErased");
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Core.Context.get.0 2020-06-06
TEST_F(ContextTest, get0){
	Context &context = Context::getContext();
	EXPECT_EQ(
		context.get<BaseA>(
			"MyBaseA"
		).getDynamicTypeInformation().getName(),
		"BaseA"
	);
	EXPECT_EQ(
		context.get<BaseB>(
			"MyBaseB"
		).getDynamicTypeInformation().getName(),
		"BaseB"
	);
	EXPECT_EQ(
		context.get<Derived>(
			"MyDerived"
		).getDynamicTypeInformation().getName(),
		"Derived"
	);
}

//TBTKFeature Core.Context.get.1 2020-06-06
TEST_F(ContextTest, get1){
	Context &context = Context::getContext();
	EXPECT_EQ(
		context.get<BaseA>(
			"MyDerived"
		).getDynamicTypeInformation().getName(),
		"Derived"
	);
}

//TBTKFeature Core.Context.get.2 2020-06-06
TEST_F(ContextTest, get2){
	Context &context = Context::getContext();
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			context.get<Derived>("BaseA");
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Core.Context.get.3 2020-06-06
TEST_F(ContextTest, get3){
	Context &context = Context::getContext();
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			context.get<Derived>("NonExistent");
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
