#include "TBTK/DynamicTypeInformation.h"
#include "TBTK/Streams.h"

#include "gtest/gtest.h"

#include <typeinfo>

namespace TBTK{

class BaseA{
public:
	TBTK_DYNAMIC_TYPE_INFORMATION
private:
};

class BaseB{
public:
	TBTK_DYNAMIC_TYPE_INFORMATION
private:
};

class Derived : public BaseA, public BaseB{
	TBTK_DYNAMIC_TYPE_INFORMATION
};

DynamicTypeInformation BaseA::dynamicTypeInformation("BaseA", {});
DynamicTypeInformation BaseB::dynamicTypeInformation("BaseB", {});
DynamicTypeInformation Derived::dynamicTypeInformation(
	"Derived",
	{&BaseA::dynamicTypeInformation, &BaseB::dynamicTypeInformation}
);

//TBTKFeature Utilities.DynamicTypeInformation.getName.0 2020-06-05
TEST(DynamicTypeInformation, getName0){
	EXPECT_EQ(Derived::dynamicTypeInformation.getName(), "Derived");
}

//TBTKFeature Utilities.DynamicTypeInformation.getName.1 2020-06-05
TEST(DynamicTypeInformation, getName1){
	Derived derived;
	EXPECT_EQ(derived.dynamicTypeInformation.getName(), "Derived");
	EXPECT_EQ(((BaseA&)derived).dynamicTypeInformation.getName(), "BaseA");
}

//TBTKFeature Utilities.DynamicTypeInformation.getDynamicTypeInformation.0 2020-06-05
TEST(DynamicTypeInformation, getDynamicTypeInformation0){
	Derived derived;
	EXPECT_EQ(derived.getDynamicTypeInformation().getName(), "Derived");
	EXPECT_EQ(
		((BaseA&)derived).getDynamicTypeInformation().getName(),
		"Derived"
	);
	EXPECT_EQ(
		((BaseA)derived).getDynamicTypeInformation().getName(),
		"BaseA"
	);
}

//TBTKFeature Utilities.DynamicTypeInformation.getNumParents.0 2020-06-05
TEST(DynamicTypeInformation, getNumParents0){
	EXPECT_EQ(BaseA::dynamicTypeInformation.getNumParents(), 0);
	EXPECT_EQ(BaseB::dynamicTypeInformation.getNumParents(), 0);
	EXPECT_EQ(Derived::dynamicTypeInformation.getNumParents(), 2);
}

//TBTKFeature Utilities.DynamicTypeInformation.getParent.0 2020-06-05
TEST(DynamicTypeInformation, getParent0){
	EXPECT_EQ(
		Derived::dynamicTypeInformation.getParent(0).getName(),
		"BaseA"
	);
	EXPECT_EQ(
		Derived::dynamicTypeInformation.getParent(1).getName(),
		"BaseB"
	);
}

};
