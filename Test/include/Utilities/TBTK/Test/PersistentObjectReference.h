#include "TBTK/PersistentObjectReference.h"

#include "gtest/gtest.h"

namespace TBTK{

class Base : virtual public PersistentObject{
	TBTK_DYNAMIC_TYPE_INFORMATION(Base)
public:
	virtual std::string serialize(Mode mode) const{
		return "";
	}
};

class Derived : public Base{
	TBTK_DYNAMIC_TYPE_INFORMATION(Derived)
public:
	virtual std::string serialize(Mode mode) const{
		return "";
	}
};

class Other : virtual public PersistentObject{
	TBTK_DYNAMIC_TYPE_INFORMATION(Other)
public:
	virtual std::string serialize(Mode mode) const{
		return "";
	}
};

DynamicTypeInformation Base::dynamicTypeInformation("Base", {});
DynamicTypeInformation Derived::dynamicTypeInformation(
	"Derived",
	{&Base::dynamicTypeInformation}
);
DynamicTypeInformation Other::dynamicTypeInformation("Other", {});

//TBTKFeature Utilities.PersistentObjectReference.construction.0 2020-06-07
TEST(PersistentObject, construction0){
	PersistentObjectReference<Base> reference;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			reference.get<Base>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PersistentObjectReference.set.0 2020-06-07
TEST(PersistentObject, set0){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	reference.get<Base>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.0 2020-06-07
TEST(PersistentObject, get0){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	reference.get<Base>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.1 2020-06-07
TEST(PersistentObject, get1){
	Derived derived;
	PersistentObjectReference<Derived> reference;
	reference.set(derived);
	reference.get<Derived>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.2 2020-06-07
TEST(PersistentObject, get2){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			reference.get<Derived>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PersistentObjectReference.get.3 2020-06-07
TEST(PersistentObject, get3){
	Derived derived;
	PersistentObjectReference<Derived> reference;
	reference.set(derived);
	reference.get<Base>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.4 2020-06-07
TEST(PersistentObject, get4){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			reference.get<Other>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PersistentObjectReference.get.5 2020-06-07
TEST(PersistentObject, get5){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	((const PersistentObjectReference<Base>&)reference).get<Base>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.6 2020-06-07
TEST(PersistentObject, get6){
	Derived derived;
	PersistentObjectReference<Derived> reference;
	reference.set(derived);
	((const PersistentObjectReference<Derived>&)reference).get<Derived>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.7 2020-06-07
TEST(PersistentObject, get7){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			((const PersistentObjectReference<Base>&)reference).get<Derived>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PersistentObjectReference.get.8 2020-06-07
TEST(PersistentObject, get8){
	Derived derived;
	PersistentObjectReference<Derived> reference;
	reference.set(derived);
	((const PersistentObjectReference<Derived>&)reference).get<Base>();
}

//TBTKFeature Utilities.PersistentObjectReference.get.9 2020-06-07
TEST(PersistentObject, get9){
	Base base;
	PersistentObjectReference<Base> reference;
	reference.set(base);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			((const PersistentObjectReference<Base>&)reference).get<Other>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
