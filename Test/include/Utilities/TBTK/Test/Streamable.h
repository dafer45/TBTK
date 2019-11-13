#include "TBTK/Streamable.h"

#include "gtest/gtest.h"

namespace TBTK{

class ImplementedStreamable : public Streamable{
public:
	virtual std::string toString() const{
		return "ReplyString";
	}
};

//TBTKFeature Utilities.Streamable.operatorOstream.1 2019-11-13
TEST(Streamable, operatorOstream1){
	EXPECT_EQ(
		ImplementedStreamable().toString().compare("ReplyString"),
		0
	);
}

};
