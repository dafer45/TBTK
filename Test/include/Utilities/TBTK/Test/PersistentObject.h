#include "TBTK/PersistentObject.h"

#include "gtest/gtest.h"

namespace TBTK{

class ImplementedPersistentObject : public PersistentObject{
public:
	std::string serialize(Mode mode) const{
		return "";
	}
};

//TBTKFeature Utilities.PersistentObject.construction.0 2020-06-07
TEST(PersistentObject, construction0){
	ImplementedPersistentObject persistentObject;
}

//TBTKFeature Utilities.PersistentObject.copyConstruction.0 2020-06-07
TEST(PersistentObject, copyConstruction0){
	ImplementedPersistentObject persistentObject;
	ImplementedPersistentObject copy(persistentObject);
}

//TBTKFeature Utilities.PersistentObject.moveConstruction.0 2020-06-07
TEST(PersistentObject, moveConstruction0){
	ImplementedPersistentObject persistentObject;
	ImplementedPersistentObject moved = std::move(persistentObject);
}

//TBTKFeature Utilities.PersistentObject.operatorAssignment.0 2020-06-07
TEST(PersistentObject, operatorAssignment0){
	ImplementedPersistentObject persistentObject;
	ImplementedPersistentObject copy;
	copy = persistentObject;
}

//TBTKFeature Utilities.PersistentObject.operatorAssignment.1 2020-06-07
TEST(PersistentObject, operatorAssignment1){
	ImplementedPersistentObject persistentObject;
	persistentObject = persistentObject;
}

//TBTKFeature Utilities.PersistentObject.operatorMoveAssignment.0 2020-06-07
TEST(PersistentObject, operatorMoveAssignment0){
	ImplementedPersistentObject persistentObject;
	ImplementedPersistentObject moved;
	moved = std::move(persistentObject);
}

//TBTKFeature Utilities.PersistentObject.operatorMoveAssignment.1 2020-06-07
TEST(PersistentObject, operatorMoveAssignment1){
	ImplementedPersistentObject persistentObject;
	persistentObject = std::move(persistentObject);
}

};
